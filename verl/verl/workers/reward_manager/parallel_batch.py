# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Parallel Batch Reward Manager - 多进程并行计算 reward
"""

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Process, Queue, cpu_count
from typing import Any, Dict, List, Tuple
import time

import torch

from verl import DataProto
from verl.workers.reward_manager import register


def _compute_score_impl(index: int, data_source: str, solution_str: str, ground_truth: str, 
                        extra_info: Dict[str, Any], timeout_seconds: int) -> Tuple[int, Any, str]:
    """实际的计算逻辑"""
    try:
        from verl.utils.reward_score import default_compute_score
        
        score = default_compute_score(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
            timeout_seconds=timeout_seconds,
        )
        return (index, score, None)
    except Exception as e:
        return (index, 0.0, str(e))


def _compute_score_with_timeout(args: Tuple, result_queue: Queue, timeout_seconds: int):
    """在子进程中运行，结果放入队列"""
    index, data_source, solution_str, ground_truth, extra_info, _ = args
    result = _compute_score_impl(index, data_source, solution_str, ground_truth, extra_info, timeout_seconds)
    result_queue.put(result)


def _compute_score_worker(args: Tuple[int, str, str, str, Dict[str, Any], Any, int]) -> Tuple[int, Any, str]:
    """
    多进程 worker 函数，用于并行计算单个样本的得分
    使用子进程+超时来确保不会卡住
    
    Args:
        args: (index, data_source, solution_str, ground_truth, extra_info, timeout_seconds)
        
    Returns:
        (index, score, error_message)
    """
    index, data_source, solution_str, ground_truth, extra_info, timeout_seconds = args
    
    # 创建一个队列来接收结果
    result_queue = Queue()
    
    # 创建子进程
    process = Process(
        target=_compute_score_with_timeout,
        args=(args, result_queue, timeout_seconds)
    )
    process.start()
    
    # 等待进程完成，带超时
    process.join(timeout=timeout_seconds + 5)  # 额外给 5 秒缓冲
    
    if process.is_alive():
        # 超时，强制终止进程
        process.terminate()
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join()
        return (index, 0.0, f"Timeout after {timeout_seconds}s (process killed)")
    
    # 获取结果
    if not result_queue.empty():
        return result_queue.get()
    else:
        return (index, 0.0, "No result returned")


@register("parallel_batch")
class ParallelBatchRewardManager:
    """
    A parallel batch reward manager that computes rewards for a batch of data using multiprocessing.

    Args:
        tokenizer (Tokenizer): The tokenizer to use for decoding the responses.
        num_examine (int): The number of responses to examine.
        compute_score (callable): The function to compute the rewards (used for fallback).
        reward_fn_key (str): The key to use for the reward function.
        num_workers (int): Number of parallel workers (default: min(cpu_count(), 32)).
        timeout_seconds (int): Timeout for each score computation (default: 300).
        reward_kwargs (dict): The keyword arguments to pass to the reward function.
    """

    def __init__(
        self, 
        tokenizer, 
        num_examine, 
        compute_score, 
        reward_fn_key="data_source",
        num_workers: int = None,
        timeout_seconds: int = 300,
        **reward_kwargs
    ):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key
        self.reward_kwargs = reward_kwargs
        self.num_workers = num_workers if num_workers is not None else min(cpu_count(), 32)
        self.timeout_seconds = timeout_seconds
        print(f"[ParallelBatchRewardManager] Initialized with {self.num_workers} workers, timeout={self.timeout_seconds}s")

    def verify_parallel(self, data) -> List[Any]:
        """
        并行计算所有样本的得分
        
        Args:
            data: DataProto object
            
        Returns:
            List of scores
        """
        prompt_ids = data.batch["prompts"]
        response_ids = data.batch["responses"]
        attention_mask = data.batch["attention_mask"]

        prompt_len = prompt_ids.shape[-1]
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)

        # 准备所有样本的数据
        args_list = []
        for i in range(len(data)):
            valid_len = valid_response_lengths[i].item()
            valid_response_ids = response_ids[i][:int(valid_len)]
            solution_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            
            ground_truth = data[i].non_tensor_batch["reward_model"].get("ground_truth", None)
            data_source = data.non_tensor_batch[self.reward_fn_key][i]
            extra_info = data.non_tensor_batch.get("extra_info", [None] * len(data))[i]
            
            args_list.append((
                i,
                data_source,
                solution_str,
                ground_truth,
                extra_info,
                self.timeout_seconds,
            ))
        
        # 初始化结果列表
        scores = [0.0] * len(data)
        errors = [None] * len(data)
        
        # 使用 ProcessPoolExecutor 并行计算
        # 每个 worker 内部有自己的子进程超时控制，可以强制终止卡住的任务
        start_time = time.time()
        print(f"[ParallelBatchRewardManager] Starting parallel reward computation for {len(args_list)} samples...")
        
        try:
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                # 使用 map 并行执行所有任务
                results = list(executor.map(_compute_score_worker, args_list))
                
            for index, score, error_msg in results:
                if error_msg is not None:
                    print(f"[ParallelBatchRewardManager] Error computing score for sample {index}: {error_msg}")
                    scores[index] = 0.0
                    errors[index] = error_msg
                else:
                    scores[index] = score
                    
        except Exception as e:
            print(f"[ParallelBatchRewardManager] Pool error: {e}, falling back to sequential computation")
            # Fallback to sequential computation
            for args in args_list:
                try:
                    index, score, error_msg = _compute_score_worker(args)
                    scores[index] = score if error_msg is None else 0.0
                except Exception as ex:
                    scores[args[0]] = 0.0
        
        elapsed = time.time() - start_time
        print(f"[ParallelBatchRewardManager] Finished computing {len(args_list)} scores in {elapsed:.2f}s ({elapsed/len(args_list):.3f}s/sample)")
        
        return scores

    def __call__(self, data: DataProto, return_dict=False):
        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        prompt_ids = data.batch["prompts"]
        prompt_len = prompt_ids.shape[-1]
        attention_mask = data.batch["attention_mask"]
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)
        data_sources = data.non_tensor_batch[self.reward_fn_key]

        # 使用并行计算
        scores = self.verify_parallel(data)
        rewards = []
        already_printed = {}

        for i in range(len(data)):
            length = valid_response_lengths[i].item()
            score = scores[i]

            if isinstance(score, dict):
                reward = score["score"]
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            rewards.append(reward)
            reward_tensor[i, int(length) - 1] = reward

            data_source = data_sources[i]
            if already_printed.get(data_source, 0) < self.num_examine:
                response_str = self.tokenizer.decode(data.batch["responses"][i][:int(length)], skip_special_tokens=True)
                prompt_str = self.tokenizer.decode(data.batch["prompts"][i], skip_special_tokens=True)
                ground_truth = data[i].non_tensor_batch["reward_model"].get("ground_truth", None)
                print("[prompt]", prompt_str[:500] + "..." if len(prompt_str) > 500 else prompt_str)
                print("[response]", response_str[:500] + "..." if len(response_str) > 500 else response_str)
                print("[ground_truth]", ground_truth)
                print("[score]", scores[i])
                already_printed[data_source] = already_printed.get(data_source, 0) + 1

        data.batch["acc"] = torch.tensor(rewards, dtype=torch.float32, device=prompt_ids.device)

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        else:
            return reward_tensor

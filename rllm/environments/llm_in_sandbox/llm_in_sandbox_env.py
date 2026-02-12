"""
LLM-in-Sandbox Environment for rllm.

This environment wraps llm-in-sandbox's DockerRuntime to provide a standard
Gym-like interface for RL training.
"""

import json
import logging
import os
import tempfile
import shutil
from concurrent.futures import ProcessPoolExecutor
from typing import Any

from rllm.environments.base.base_env import BaseEnv

# Import from llm-in-sandbox
from llm_in_sandbox import DockerRuntime, Action
from llm_in_sandbox.agent import Agent as LLMSandboxAgentImpl
from llm_in_sandbox.benchmark.runner import load_reward_function

logger = logging.getLogger(__name__)

DEFAULT_WORKDIR = "/testbed"


def _compute_score_in_process(task_name: str, agent_answer: str, ground_truth: str, entry_kwargs: dict) -> float:
    """Wrapper function that can be pickled for multiprocessing.
    
    Loads the reward function in the subprocess and computes the score.
    """
    from llm_in_sandbox.benchmark.runner import load_reward_function
    compute_score_fn = load_reward_function(task_name)
    return compute_score_fn(agent_answer, ground_truth, **entry_kwargs)


class LLMinSandboxEnv(BaseEnv):
    """
    LLM-in-Sandbox Environment.
    
    Wraps DockerRuntime and reuses llm-in-sandbox's Agent for action execution.
    """

    def __init__(
        self,
        entry: dict,
        docker_image: str = "cdx123/llm-in-sandbox:v0.1",
        step_timeout: int = 120,
        verbose: bool = False,
        working_dir: str = None,
        input_dir: str = None,
        output_dir: str = None,
    ):
        self.entry = entry
        self.docker_image = docker_image
        self.step_timeout = step_timeout
        self.verbose = verbose
        
        # Directory configs (use entry values or defaults)
        self.working_dir = working_dir or entry.get("working_dir", DEFAULT_WORKDIR)
        self.input_dir = input_dir or entry.get("input_dir", f"{self.working_dir}/documents")
        self.output_dir = output_dir or entry.get("output_dir", f"{self.working_dir}/")
        
        self.runtime = None
        self.total_steps = 0
        self._temp_dir = None
        
        # Will be used to execute actions (reuse llm-in-sandbox's implementation)
        self._agent_impl = LLMSandboxAgentImpl.__new__(LLMSandboxAgentImpl)

    def reset(self) -> tuple[str, dict]:
        """Reset the environment."""
        if self.runtime is not None:
            self.close()

        self.runtime = DockerRuntime(
            docker_image=self.docker_image,
            repo_path=self.working_dir,
            logger=logger if self.verbose else False,
        )

        # Copy input files if provided
        input_files = self.entry.get("input_files")
        if input_files:
            self._copy_input_files(input_files)

        self.total_steps = 0
        problem_statement = self.entry.get("problem_statement", "")
        
        return problem_statement, {"extra_info": self.entry}

    def _copy_input_files(self, input_files: str | dict):
        """Copy input files to the container."""
        if isinstance(input_files, str):
            input_files = json.loads(input_files) if input_files else {}
        
        if not input_files:
            return
            
        self._temp_dir = tempfile.mkdtemp()
        
        for filename, content in input_files.items():
            if content is None:
                continue
            temp_path = os.path.join(self._temp_dir, filename)
            os.makedirs(os.path.dirname(temp_path) if os.path.dirname(temp_path) else self._temp_dir, exist_ok=True)
            with open(temp_path, "w", encoding="utf-8") as f:
                f.write(content)
        
        self.runtime.copy_dir_to_container(self._temp_dir, self.input_dir)

    def step(self, action: Any) -> tuple[str, float, bool, dict]:
        """Execute an action using llm-in-sandbox's implementation."""
        self.total_steps += 1
        
        # Convert to llm-in-sandbox Action if needed
        if isinstance(action, str):
            action = Action.from_string(action)
        elif hasattr(action, 'function_name') and not isinstance(action, Action):
            # Convert from r2egym Action to llm-in-sandbox Action
            action = Action(
                function_name=action.function_name,
                parameters=action.parameters if hasattr(action, 'parameters') else {}
            )

        done = action.function_name == "submit"
        
        # Directly call llm-in-sandbox Agent's _execute_action method
        observation = self._agent_impl._execute_action(action, self.runtime)

        return observation, 0.0, done, {}

    def compute_final_reward(self) -> float:
        """Compute reward using llm-in-sandbox's reward functions."""
        answer_output, _ = self.runtime.run(
            f"cat {self.output_dir}/answer.txt 2>/dev/null || echo ''"
        )
        agent_answer = answer_output.strip()
        
        ground_truth = self.entry["ground_truth"]
        domain = self.entry["domain"]
        
        try:
            # For training, always use instruct_pretrain reward which routes by domain
            task_name = "instruct_pretrain"
            # Pass all entry fields as kwargs (for qa_type, domain, etc.)
            entry_kwargs = {k: v for k, v in self.entry.items() if k != 'ground_truth'}
            
            # Run reward computation in separate process to avoid thread conflicts
            # (math_verify uses threading internally for timeout)
            with ProcessPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_compute_score_in_process, task_name, agent_answer, ground_truth, entry_kwargs)
                reward = future.result(timeout=60)  # 60s timeout for reward computation
            
            # Debug print
            answer_preview = agent_answer[-50:] if len(agent_answer) > 50 else agent_answer
            gt_preview = ground_truth[:100] if len(ground_truth) > 100 else ground_truth
            print(f"[Score] domain={domain} | score={reward:.2f} | output[-50:]={repr(answer_preview)} | gt={repr(gt_preview)}")
            
            return reward
        except Exception as e:
            raise RuntimeError(f"Error computing reward for {domain}: {e}") from e

    def close(self) -> None:
        """Close the environment."""
        if self.runtime is not None:
            try:
                self.runtime.close()
            except Exception as e:
                logger.warning(f"Error closing runtime: {e}")
            self.runtime = None
        
        if self._temp_dir and os.path.exists(self._temp_dir):
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._temp_dir = None

    @staticmethod
    def from_dict(extra_info: dict | str) -> "LLMSandboxEnv":
        """Create an environment instance from dictionary."""
        if isinstance(extra_info, str):
            extra_info = json.loads(extra_info)
        
        docker_image = extra_info.pop("docker_image", "cdx123/llm-in-sandbox:v0.1")
        step_timeout = extra_info.pop("step_timeout", 120)
        verbose = extra_info.pop("verbose", False)
        working_dir = extra_info.pop("working_dir", None)
        input_dir = extra_info.pop("input_dir", None)
        output_dir = extra_info.pop("output_dir", None)
        
        return LLMinSandboxEnv(
            entry=extra_info,
            docker_image=docker_image,
            step_timeout=step_timeout,
            verbose=verbose,
            working_dir=working_dir,
            input_dir=input_dir,
            output_dir=output_dir,
        )

    @staticmethod
    def is_multithread_safe() -> bool:
        return True

# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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

import copy
import glob
import logging
import os
import re
from collections import defaultdict
from typing import Optional

import datasets
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask

logger = logging.getLogger(__name__)


def collate_fn(data_list: list[dict]) -> dict:
    """
    Collate a batch of sample dicts into batched tensors and arrays.

    Args:
        data_list: List of dicts mapping feature names to torch.Tensor or other values.

    Returns:
        Dict where tensor entries are stacked into a torch.Tensor of shape
        (batch_size, \*dims) and non-tensor entries are converted to
        np.ndarray of dtype object with shape (batch_size,).
    """
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    return {**tensors, **non_tensors}


class RLHFDataset(Dataset):
    """
    Load and preprocess RLHF data from Parquet or JSON files.

    - Caches files locally.
    - Reads into a HuggingFace Dataset and tokenizes prompts.
    - Optionally handles images/videos via a ProcessorMixin.
    - Filters prompts over a max length.
    - Supports resuming from checkpoints.
    - Automatically detects file format (.parquet or .json) based on extension.

    Args:
        data_files (str or list): Path(s) to Parquet or JSON file(s).
        tokenizer (PreTrainedTokenizer): For the tokenization of text to token IDs.
        config (DictConfig): Options like cache_dir, prompt_key, max_prompt_length, truncation, etc.
        processor (ProcessorMixin, optional): Multimodal preprocessor for images/videos.
    """

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        if not isinstance(data_files, list | ListConfig):
            data_files = [data_files]

        # Expand glob patterns in data_files
        expanded_files = []
        for file_pattern in data_files:
            # Check if the pattern contains wildcards
            if '*' in file_pattern or '?' in file_pattern:
                matched_files = glob.glob(file_pattern)
                if not matched_files:
                    raise FileNotFoundError(f"No files matched the pattern: {file_pattern}")
                expanded_files.extend(sorted(matched_files))
            else:
                expanded_files.append(file_pattern)
        
        self.data_files = copy.deepcopy(expanded_files)
        self.original_data_files = copy.deepcopy(expanded_files)  # use for resume
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.prompt_key = config.get("prompt_key", "prompt")
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.return_full_prompt = config.get("return_full_prompt", False)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)

        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        self.num_workers = min(self.num_workers, os.cpu_count())
        self.use_shm = config.get("use_shm", False)
        self.chat_template_func = config.get("chat_template_func", None)
        self.need_tools_kwargs = config.get("need_tools_kwargs", False)
        self.filter_prompts = config.get("filter_prompts", True)
        self.serialize_dataset = False
        self.return_multi_modal_inputs = config.get("return_multi_modal_inputs", True)

        self._download()
        self._read_files_and_tokenize()

    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_to_local

        data_files = self.data_files if not use_origin_parquet else self.original_data_files
        for i, data_file in enumerate(data_files):
            self.data_files[i] = copy_to_local(src=data_file, cache_dir=self.cache_dir, use_shm=self.use_shm)

    def _read_files_and_tokenize(self):
        import json
        
        dataframes = []
        self._use_raw_data = False  # Track if we're using raw Python list
        
        for data_file in self.data_files:
            # Try to detect file format and load accordingly
            if data_file.endswith('.json') or data_file.endswith('.jsonl'):
                try:
                    print(f"Loading JSON file: {data_file}")
                    
                    # Check if it's a JSON array file by reading the first few bytes
                    with open(data_file, 'r') as f:
                        first_char = f.read(1).strip()
                    
                    # If file starts with '[', it's likely a JSON array, not JSONL
                    if first_char == '[':
                        # HuggingFace datasets merges dict schemas across samples, which causes
                        # issues when samples have dicts with different keys (e.g., different documents).
                        # Solution: Keep data as plain Python list
                        import json
                        
                        with open(data_file, 'r') as f:
                            data = json.load(f)
                        
                        print(f"✓ Loaded {len(data)} items from JSON array file (raw mode)")
                        
                        # Debug: print first entry structure
                        if len(data) > 0:
                            print("\n[DEBUG] First entry structure:")
                            first_entry = data[0]
                            for key, value in first_entry.items():
                                if isinstance(value, dict):
                                    print(f"  {key}: <dict with {len(value)} keys>")
                                elif isinstance(value, list):
                                    print(f"  {key}: <list with {len(value)} items>")
                                elif isinstance(value, str) and len(value) > 100:
                                    print(f"  {key}: <string with {len(value)} chars>")
                                else:
                                    print(f"  {key}: {value}")
                            
                            # Check if documents field exists and show its structure
                            if 'extra_info' in first_entry and isinstance(first_entry['extra_info'], dict):
                                if 'documents' in first_entry['extra_info']:
                                    docs = first_entry['extra_info']['documents']
                                    print(f"  └─ extra_info.documents: {len(docs)} documents")
                            print()
                        
                        # Use raw Python list to preserve variable dict structures
                        self._use_raw_data = True
                        dataframe = data
                    else:
                        # Load as JSONL (one JSON object per line)
                        dataframe = datasets.load_dataset(
                            "json", 
                            data_files=data_file, 
                            split="train",
                            cache_dir=self.cache_dir,
                            keep_in_memory=False
                        )
                        print(f"✓ Loaded {len(dataframe)} items from JSONL file")
                        
                        # Debug: print first entry structure
                        if len(dataframe) > 0:
                            print("\n[DEBUG] First entry structure:")
                            first_entry = dataframe[0]
                            for key, value in first_entry.items():
                                if isinstance(value, dict):
                                    print(f"  {key}: <dict with {len(value)} keys: {list(value.keys())}>")
                                elif isinstance(value, list):
                                    print(f"  {key}: <list with {len(value)} items>")
                                elif isinstance(value, str) and len(value) > 100:
                                    print(f"  {key}: <string with {len(value)} chars, truncated: {value[:100]}...>")
                                else:
                                    print(f"  {key}: {value}")
                            print()
                    
                except Exception as e:
                    print(f"✗ Error loading JSON file {data_file}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    raise e
            else:
                # Try parquet first, then fallback to JSON
                try:
                    dataframe = datasets.load_dataset(
                        "parquet", 
                        data_files=data_file, 
                        split="train",
                        cache_dir=self.cache_dir,
                        keep_in_memory=False
                    )
                    print(f"✓ Loaded parquet file: {data_file}")
                except Exception as e:
                    print(f"✗ Error reading parquet file {data_file}: {str(e)}")
                    # Try JSON as fallback
                    json_file = data_file.replace('.parquet', '.json')
                    try:
                        print(f"Trying to load JSON version from {json_file}")
                        
                        dataframe = datasets.load_dataset(
                            "json", 
                            data_files=json_file, 
                            split="train",
                            cache_dir=self.cache_dir,
                            keep_in_memory=False
                        )
                        print(f"✓ Successfully loaded JSON version: {len(dataframe)} items")
                    except Exception as json_e:
                        print(f"✗ Also failed to read JSON file {json_file}: {str(json_e)}")
                        import traceback
                        traceback.print_exc()
                        raise e
            
            dataframes.append(dataframe)
        
        # Handle raw data vs HF Dataset
        if self._use_raw_data or any(isinstance(df, list) for df in dataframes):
            # Convert all to raw Python lists for consistency
            self.dataframe = []
            for df in dataframes:
                if isinstance(df, list):
                    self.dataframe.extend(df)
                else:
                    # Convert HF Dataset to list
                    self.dataframe.extend([dict(item) for item in df])
            self._use_raw_data = True
            print(f"Total dataset len: {len(self.dataframe)} (raw mode)")
        else:
            # Standard HF Dataset concatenation
            self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)
            print(f"Total dataset len: {len(self.dataframe)}")

        self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe)

    def maybe_filter_out_long_prompts(self, dataframe):
        # filter out too long prompts
        if self.filter_overlong_prompts:
            tokenizer = self.tokenizer
            processor = self.processor
            prompt_key = self.prompt_key
            image_key = self.image_key
            video_key = self.video_key

            if processor is not None:
                from verl.utils.dataset.vision_utils import process_image, process_video

                def doc2len(doc) -> int:
                    messages = self._build_messages(doc.copy())  # Use copy to avoid modifying original
                    raw_prompt = self.processor.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=False
                    )
                    images = [process_image(image) for image in doc[image_key]] if image_key in doc else None
                    videos = [process_video(video) for video in doc[video_key]] if video_key in doc else None

                    return len(processor(text=[raw_prompt], images=images, videos=videos)["input_ids"][0])

            else:

                def doc2len(doc) -> int:
                    return len(tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True))

            if self._use_raw_data:
                # Filter raw Python list
                print(f"Filtering prompts longer than {self.max_prompt_length} tokens (raw mode)...")
                filtered = []
                for i, doc in enumerate(dataframe):
                    try:
                        if doc2len(doc) <= self.max_prompt_length:
                            filtered.append(doc)
                    except Exception as e:
                        logger.warning(f"Error filtering document {i}: {e}")
                dataframe = filtered
            else:
                # Filter HF Dataset
                dataframe = dataframe.filter(
                    lambda doc: doc2len(doc) <= self.max_prompt_length,
                    num_proc=self.num_workers,
                    desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
                )

            print(f"filter dataset len: {len(dataframe)}")
        return dataframe

    def resume_dataset_state(self):
        self.serialize_dataset = not hasattr(self, "original_data_files")
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)  # download and resume from original parquet files
            self._read_files_and_tokenize()
        else:
            print(r"old dataloader ckpt file is used, please train from scratch for better ckpt performance")

    def __len__(self):
        return len(self.dataframe)

    def _build_messages(self, example: dict):
        messages: list = example.pop(self.prompt_key)

        if self.image_key in example or self.video_key in example:
            for message in messages:
                content = message["content"]
                content_list = []
                segments = re.split("(<image>|<video>)", content)
                segments = [item for item in segments if item != ""]
                for segment in segments:
                    if segment == "<image>":
                        content_list.append({"type": "image"})
                    elif segment == "<video>":
                        content_list.append({"type": "video"})
                    else:
                        content_list.append({"type": "text", "text": segment})

                message["content"] = content_list

        return messages

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        if self._use_raw_data:
            # Make a copy to avoid modifying the original data
            row_dict = copy.deepcopy(self.dataframe[item])
        else:
            row_dict: dict = self.dataframe[item]
        messages = self._build_messages(row_dict)
        model_inputs = {}

        if self.processor is not None:
            from verl.utils.dataset.vision_utils import process_image, process_video

            raw_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            multi_modal_data = {}

            images = None
            if self.image_key in row_dict and row_dict.get(self.image_key, None) is not None:
                images = [process_image(image) for image in row_dict.pop(self.image_key)]

                # due to the image key is "image" instead of "images" in vllm, we need to use "image" here
                # link: https://github.com/vllm-project/vllm/blob/3c545c0c3b98ee642373a308197d750d0e449403/vllm/multimodal/parse.py#L205
                multi_modal_data["image"] = images

            videos = None
            if self.video_key in row_dict and row_dict.get(self.video_key, None) is not None:
                videos = [process_video(video) for video in row_dict.pop(self.video_key)]

                # due to the video key is "video" instead of "videos" in vllm, we need to use "video" here
                # link: https://github.com/vllm-project/vllm/blob/3c545c0c3b98ee642373a308197d750d0e449403/vllm/multimodal/parse.py#L205
                multi_modal_data["video"] = [video.numpy() for video in videos]

            model_inputs = self.processor(text=[raw_prompt], images=images, videos=videos, return_tensors="pt")

            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

            if "second_per_grid_ts" in model_inputs:
                model_inputs.pop("second_per_grid_ts")

            # There's a trap here, multi_modal_inputs has to be a dict, not BatchFeature
            row_dict["multi_modal_data"] = multi_modal_data

            # We will do batch.union() in the trainer,
            # so we cannot have "multi_modal_inputs" in row_dict if rollout generates new multi_modal_inputs
            if self.return_multi_modal_inputs:
                row_dict["multi_modal_inputs"] = dict(model_inputs)

                # second_per_grid_ts isn't used for training, just for mrope
                row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)

        else:
            raw_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            from verl.models.transformers.qwen2_vl import get_rope_index

            position_ids = [
                get_rope_index(
                    self.processor,
                    input_ids=input_ids[0],
                    image_grid_thw=model_inputs.get("image_grid_thw"),
                    video_grid_thw=model_inputs.get("video_grid_thw"),
                    second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                    attention_mask=attention_mask[0],
                )
            ]  # (1, 3, seq_len)

        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages

        # get prompts with chat template
        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt  # array of strings

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        interaction_kwargs = row_dict.get("extra_info", {}).get("interaction_kwargs", {})
        need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)
        
        # Ensure required fields exist with default values
        if "data_source" not in row_dict:
            row_dict["data_source"] = "unknown"
        
        if need_tools_kwargs and not tools_kwargs:
            logger.warning("tools_kwargs is empty for index {}, data source: {}", index, row_dict["data_source"])
        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs
        row_dict["interaction_kwargs"] = interaction_kwargs
        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if "dataframe" in state:
                del state["dataframe"]
            return state

        return self.__dict__.copy()

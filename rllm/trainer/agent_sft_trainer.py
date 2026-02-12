import logging

import torch
from torch.distributed.device_mesh import init_device_mesh

from rllm.agents.agent import Trajectory
from rllm.parser.chat_template_parser import ChatTemplateParser
from verl.trainer.fsdp_sft_trainer import FSDPSFTTrainer
from verl.utils import hf_tokenizer
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
from verl.utils.device import get_device_name
from verl.utils.distributed import destroy_global_process_group, initialize_global_process_group
from verl.utils.fs import copy_to_local

logger = logging.getLogger(__name__)


class RLLMSFTDataset(MultiTurnSFTDataset):
    def __init__(self, parquet_files: str | list[str], tokenizer, config=None):
        super().__init__(parquet_files, tokenizer, config)

        self.tokenize_and_mask_method = config.rllm.tokenize_and_mask_method
        logger.info(f"Using {self.tokenize_and_mask_method} tokenization and masking method")

        self.parser = ChatTemplateParser.get_parser(tokenizer)

    def _tokenize_and_mask(self, messages):
        if self.tokenize_and_mask_method == "cumulative":
            return self._tokenize_and_mask_cumulative(messages)
        elif self.tokenize_and_mask_method == "stepwise":
            return self._tokenize_and_mask_stepwise(messages)
        else:
            raise ValueError(f"Unknown tokenize_and_mask_method {self.tokenize_and_mask_method}")

    def _tokenize_and_mask_cumulative(self, messages):
        tokens = []
        loss_mask = []

        for i in range(len(messages)):
            parsed_msg = self.parser.parse([messages[i]], is_first_msg=(i == 0), add_generation_prompt=False)
            ids = self.tokenizer.encode(parsed_msg, add_special_tokens=False)
            if messages[i]["role"] == "assistant":
                loss_mask.extend([1] * len(ids))
            else:
                loss_mask.extend([0] * len(ids))
            tokens.extend(ids)

        return tokens, loss_mask

    def _tokenize_and_mask_stepwise(self, messages):
        tokens = []
        loss_mask = []

        # Find the index of the last assistant message
        last_assistant_idx = -1
        for i in range(len(messages)):
            if messages[i]["role"] == "assistant":
                last_assistant_idx = i
        assert last_assistant_idx != -1, "No assistant message found in chat_completions"

        for i in range(len(messages)):
            parsed_msg = self.parser.parse([messages[i]], is_first_msg=(i == 0), add_generation_prompt=False)
            ids = self.tokenizer.encode(parsed_msg, add_special_tokens=False)
            if i == last_assistant_idx and messages[i]["role"] == "assistant":
                loss_mask.extend([1] * len(ids))
            else:
                loss_mask.extend([0] * len(ids))
            tokens.extend(ids)

        return tokens, loss_mask

    def __getitem__(self, item):
        messages = self.messages[item]

        tokens, loss_mask = self._tokenize_and_mask(messages)

        input_ids = torch.tensor(tokens, dtype=torch.long)
        loss_mask = torch.tensor(loss_mask, dtype=torch.long)
        attention_mask = torch.tensor([1] * len(tokens), dtype=torch.long)

        # Handle sequence length
        sequence_length = input_ids.shape[0]
        if sequence_length < self.max_length:
            # Pad sequences
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            padded_input_ids = torch.full((self.max_length - sequence_length,), pad_token_id, dtype=input_ids.dtype)
            padded_attention_mask = torch.zeros((self.max_length - sequence_length,), dtype=attention_mask.dtype)
            padded_loss_mask = torch.zeros((self.max_length - sequence_length,), dtype=loss_mask.dtype)

            input_ids = torch.cat((input_ids, padded_input_ids))
            attention_mask = torch.cat((attention_mask, padded_attention_mask))
            loss_mask = torch.cat((loss_mask, padded_loss_mask))

        elif sequence_length > self.max_length:
            if self.truncation == "left":
                input_ids = input_ids[-self.max_length :]
                attention_mask = attention_mask[-self.max_length :]
                loss_mask = loss_mask[-self.max_length :]
            elif self.truncation == "right":
                input_ids = input_ids[: self.max_length]
                attention_mask = attention_mask[: self.max_length]
                loss_mask = loss_mask[: self.max_length]
            elif self.truncation == "error":
                raise ValueError(f"{sequence_length=} is larger than {self.max_length=}")
            else:
                raise ValueError(f"Unknown truncation method {self.truncation}")

        # Create position IDs
        position_ids = torch.arange(len(input_ids), dtype=torch.long)
        # Zero out position IDs for padding
        position_ids = position_ids * attention_mask

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }


class AgentSFTTrainer:
    def __init__(self, config):
        self.config = config

    def run_sft(self):
        config = self.config
        device_name = get_device_name()
        local_rank, rank, world_size = initialize_global_process_group()

        device_mesh = init_device_mesh(device_type=device_name, mesh_shape=(world_size,), mesh_dim_names=("fsdp",))
        dp_size = world_size // config.ulysses_sequence_parallel_size
        ulysses_device_mesh = init_device_mesh(
            device_type=device_name,
            mesh_shape=(dp_size, config.ulysses_sequence_parallel_size),
            mesh_dim_names=("dp", "sp"),
        )
        # build tokenizer and datasets first
        local_model_path = copy_to_local(src=config.model.partial_pretrain, verbose=True)
        tokenizer = hf_tokenizer(local_model_path, trust_remote_code=config.model.trust_remote_code)

        train_dataset = RLLMSFTDataset(config.data.train_files, tokenizer, config.data)
        val_dataset = RLLMSFTDataset(config.data.val_files, tokenizer, config.data)

        trainer = FSDPSFTTrainer(
            config=config,
            device_mesh=device_mesh,
            ulysses_device_mesh=ulysses_device_mesh,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
        )

        trainer.fit()

        destroy_global_process_group()

    @staticmethod
    def process_trajectories(trajectories: list[Trajectory], reward_threshold: float):
        """Process trajectories into SFT format."""
        sft_data = []

        for traj in trajectories:
            if not traj:
                continue

            reward = traj.reward

            if reward < reward_threshold:
                continue

            # Get chat_completions from the last step of the trajectory
            messages = None
            if traj.steps and hasattr(traj.steps[-1], "chat_completions"):
                messages = traj.steps[-1].chat_completions

            if not messages:
                continue

            clean_messages = [{"role": msg["role"], "content": str(msg["content"]).strip()} for msg in messages if isinstance(msg, dict) and msg.get("role") and str(msg.get("content", "")).strip()]

            if len(clean_messages) >= 2:
                sft_data.append({"messages": clean_messages})

        print(f"Processed {len(trajectories)} trajectories -> {len(sft_data)} valid examples")
        return sft_data

    def train(self):
        """Start training."""
        self.run_sft()

#!/usr/bin/env python3
"""
Convert llm-in-sandbox-rl datasets to rllm format.

Usage:
    python examples/llm_in_sandbox/convert_llm_sandbox_dataset.py --config math_mini --output ./data/llm_sandbox_math_mini
    python examples/llm_in_sandbox/convert_llm_sandbox_dataset.py --all --output-dir ./data
"""

import argparse
import json
import os

from datasets import load_dataset


# Map config to split
CONFIG_TO_SPLIT = {
    "math_mini": "test",
    "chem_mini": "test",
    "physics_mini": "test",
    "biomed_mini": "test",
    "long_context_mini": "test",
    "instruct_follow_mini": "test",
    "instruct_pretrain": "train",
}


def convert_row(row: dict, config: str) -> dict:
    """Convert a single row from llm-in-sandbox format to rllm format."""
    extra_info = {
        "id": row["id"],
        "domain": row.get("domain", config),
        "problem_statement": row["problem_statement"],
        "ground_truth": row["ground_truth"],
        # "data_source": f"llm_in_sandbox_{config}",
    }
    
    # Optional fields
    if "input_files" in row and row["input_files"]:
        extra_info["input_files"] = row["input_files"]
    if "qa_type" in row and row["qa_type"]:
        extra_info["qa_type"] = row["qa_type"]
    
    return {
        "prompt": [{"content": "placeholder", "role": "user"}],
        "reward_model": {
            "ground_truth": row["ground_truth"],
            "style": "llm_in_sandbox",
        },
        "data_source": row.get("domain"),
        "extra_info": extra_info,
    }


def convert_dataset(config: str, output_path: str):
    """Convert a dataset config to rllm format and save as single JSON file."""
    print(f"\n{'='*60}")
    print(f"Converting: {config}")
    print(f"Output: {output_path}")
    
    # Load dataset from HuggingFace
    split = CONFIG_TO_SPLIT.get(config, "test")
    print(f"Loading daixuancheng/llm-in-sandbox-rl:{config} split={split}")
    ds = load_dataset("daixuancheng/llm-in-sandbox-rl", config, split=split)
    print(f"Loaded {len(ds)} rows")
    
    # Convert all rows
    converted = [convert_row(dict(row), config) for row in ds]
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Save as single JSON file
    output_file = os.path.join(output_path, f"{split}_verl.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)
    print(f"✓ Saved {len(converted)} rows to {output_file}")
    
    return len(converted)


def main():
    parser = argparse.ArgumentParser(description="Convert llm-in-sandbox datasets to rllm format")
    parser.add_argument("--config", type=str, help="Dataset config name (e.g., math_mini, instruct_pretrain)")
    parser.add_argument("--output", type=str, help="Output directory path")
    parser.add_argument("--all", action="store_true", help="Convert all available configs")
    parser.add_argument("--output-dir", type=str, default="./data", 
                        help="Base output directory (used with --all)")
    args = parser.parse_args()
    
    if args.all:
        configs = list(CONFIG_TO_SPLIT.keys())
        total = 0
        for config in configs:
            output_path = os.path.join(args.output_dir, f"llm_sandbox_{config}")
            total += convert_dataset(config, output_path)
        print(f"\n{'='*60}")
        print(f"✓ All done! Converted {total} total rows from {len(configs)} configs")
    elif args.config and args.output:
        convert_dataset(args.config, args.output)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python convert_llm_sandbox_dataset.py --config math_mini --output ./data/llm_sandbox_math_mini")
        print("  python convert_llm_sandbox_dataset.py --all --output-dir ./data")


if __name__ == "__main__":
    main()

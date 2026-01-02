#!/usr/bin/env python
"""Convert demo files from cleaned format to required format.

This script converts JSONL files from the format:
    {"input_text": "...", "llm_output_text": "...", ...}

To the required format:
    {"input": "...", "output": "..."}

And saves them with the correct naming convention for ICL retrieval.
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from icl_retrieval.utils.io_utils import load_jsonl, dump_jsonl


def convert_demo_file(input_path: str, output_path: str, prompt_type: str) -> int:
    """Convert a single demo file to the required format.
    
    Args:
        input_path: Path to input JSONL file
        output_path: Path to save converted JSONL file
        prompt_type: Type of prompt (for logging)
        
    Returns:
        Number of samples converted
    """
    print(f"\nConverting {prompt_type} demos...")
    print(f"  Input: {input_path}")
    print(f"  Output: {output_path}")
    
    # Load data
    data = load_jsonl(input_path)
    
    # Convert format
    converted = []
    for item in data:
        converted_item = {
            "input": item.get("input_text", ""),
            "output": item.get("llm_output_text", "")
        }
        converted.append(converted_item)
    
    # Save converted data
    dump_jsonl(converted, output_path)
    
    print(f"  Converted {len(converted)} samples")
    return len(converted)


def main():
    parser = argparse.ArgumentParser(
        description="Convert demo files to required format for ICL retrieval"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/Users/sihua/personal/GPTSwarm/data/llm_experiences/20251206_152148",
        help="Directory containing cleaned demo files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/Users/sihua/personal/GPTSwarm/data/final_demo_dec_13",
        help="Directory to save converted demo files"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define file mappings
    file_mappings = [
        ("cleaned_if_correct_prompt_demos.jsonl", "list_demos_if_correct.json", "if_correct"),
        ("cleaned_suggest_prompt_demos.jsonl", "list_demos_suggest.json", "suggest"),
        ("cleaned_value_prompt_demos.jsonl", "list_demos_value.json", "value"),
        ("cleaned_propose_prompt_demos.jsonl", "list_demos_propose.json", "propose"),
    ]
    
    total_converted = 0
    
    # Convert each file
    for input_file, output_file, prompt_type in file_mappings:
        input_path = os.path.join(args.input_dir, input_file)
        output_path = os.path.join(args.output_dir, output_file)
        
        if os.path.exists(input_path):
            count = convert_demo_file(input_path, output_path, prompt_type)
            total_converted += count
        else:
            print(f"\nWarning: {input_path} not found, skipping...")
    
    print(f"\n{'='*60}")
    print(f"Conversion complete!")
    print(f"Total samples converted: {total_converted}")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

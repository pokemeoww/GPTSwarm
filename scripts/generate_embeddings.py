#!/usr/bin/env python
"""Generate embeddings for demo files.

This script creates .npy embedding files for all demo JSON files in a directory.
"""

import os
import sys
import argparse
from pathlib import Path
import glob

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from icl_retrieval.utils.encode_data import encode_corpus


def main():
    parser = argparse.ArgumentParser(
        description="Generate embeddings for demo files"
    )
    parser.add_argument(
        "--demo_dir",
        type=str,
        required=True,
        help="Directory containing demo JSON files"
    )
    parser.add_argument(
        "--embed_model",
        type=str,
        default="BAAI/bge-base-en-v1.5",
        help="Sentence transformer model name or path"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for encoding"
    )
    
    args = parser.parse_args()
    
    # Find all demo JSON files
    demo_files = glob.glob(os.path.join(args.demo_dir, "list_demos_*.json"))
    
    if not demo_files:
        print(f"No demo files found in {args.demo_dir}")
        print("Looking for files matching pattern: list_demos_*.json")
        return
    
    print(f"Found {len(demo_files)} demo files")
    print(f"Using embedding model: {args.embed_model}")
    print("=" * 60)
    
    # Generate embeddings for each file
    for demo_file in demo_files:
        # Determine output path
        output_file = demo_file.replace(".json", ".npy")
        
        # Skip if embeddings already exist
        if os.path.exists(output_file):
            print(f"\n⏭️  Skipping {os.path.basename(demo_file)} (embeddings already exist)")
            continue
        
        print(f"\n📝 Processing: {os.path.basename(demo_file)}")
        
        # Generate embeddings
        try:
            encode_corpus(
                corpus_path=demo_file,
                model_name_or_path=args.embed_model,
                to_path=output_file,
                batch_size=args.batch_size
            )
            print(f"✅ Created: {os.path.basename(output_file)}")
        except Exception as e:
            print(f"❌ Error processing {demo_file}: {e}")
    
    print("\n" + "=" * 60)
    print("✨ Embedding generation complete!")
    print(f"Output directory: {args.demo_dir}")


if __name__ == "__main__":
    main()

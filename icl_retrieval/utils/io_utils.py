"""I/O utilities for loading and saving JSON/JSONL files."""

import json
from typing import List, Dict, Any


def load_json(filepath: str) -> Dict[str, Any]:
    """Load a JSON file.
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        Dictionary containing the JSON data
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def dump_json(data: Dict[str, Any], filepath: str) -> None:
    """Save data to a JSON file.
    
    Args:
        data: Dictionary to save
        filepath: Path to save the JSON file
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """Load a JSONL file (one JSON object per line).
    
    Args:
        filepath: Path to the JSONL file
        
    Returns:
        List of dictionaries, one per line
    """
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def dump_jsonl(data: List[Dict[str, Any]], filepath: str) -> None:
    """Save data to a JSONL file (one JSON object per line).
    
    Args:
        data: List of dictionaries to save
        filepath: Path to save the JSONL file
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

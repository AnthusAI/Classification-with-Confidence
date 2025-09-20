"""
Dataset loader for the new flat file format (dataset_v2).
"""

import os
from typing import List, Dict, Tuple
from pathlib import Path


class DatasetLoader:
    """Loads examples from the flat file dataset format."""
    
    def __init__(self, dataset_dir: str = "dataset"):
        """
        Initialize the loader.
        
        Args:
            dataset_dir: Directory containing the flat text files
        """
        self.dataset_dir = Path(dataset_dir)
        
        # Mapping from filename to sentiment and strength
        self.file_mapping = {
            "strong_positive.txt": ("positive", "strong"),
            "strong_negative.txt": ("negative", "strong"),
            "medium_positive.txt": ("positive", "medium"),
            "medium_negative.txt": ("negative", "medium"),
            "weak_positive.txt": ("positive", "weak"),
            "weak_negative.txt": ("negative", "weak"),
        }
    
    def load_file(self, filename: str) -> List[str]:
        """
        Load examples from a single file.
        
        Args:
            filename: Name of the file to load
            
        Returns:
            List of example texts (comments and empty lines filtered out)
        """
        filepath = self.dataset_dir / filename
        
        if not filepath.exists():
            return []
        
        examples = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if line and not line.startswith('#'):
                    examples.append(line)
        
        return examples
    
    def load_all(self) -> List[Dict[str, str]]:
        """
        Load all examples from all files.
        
        Returns:
            List of example dictionaries with text, expected, category, and strength
        """
        all_examples = []
        
        for filename, (sentiment, strength) in self.file_mapping.items():
            examples = self.load_file(filename)
            
            for text in examples:
                all_examples.append({
                    'text': text,
                    'expected': sentiment,
                    'category': f"{strength}_{sentiment}",
                    'strength': strength,
                    'source_file': filename
                })
        
        return all_examples
    
    def get_stats(self) -> Dict[str, int]:
        """
        Get statistics about the dataset.
        
        Returns:
            Dictionary with counts by category
        """
        stats = {}
        total = 0
        
        for filename, (sentiment, strength) in self.file_mapping.items():
            examples = self.load_file(filename)
            count = len(examples)
            category = f"{strength}_{sentiment}"
            stats[category] = count
            total += count
        
        stats['total'] = total
        return stats
    
    def print_summary(self):
        """Print a summary of the dataset."""
        stats = self.get_stats()
        
        print("=== Dataset Summary ===")
        print(f"Total examples: {stats['total']}")
        print()
        
        for category, count in sorted(stats.items()):
            if category != 'total':
                print(f"{category:20s}: {count:4d} examples")


def main():
    """Demo the loader."""
    loader = DatasetLoader()
    loader.print_summary()
    
    # Show some examples
    all_examples = loader.load_all()
    if all_examples:
        print(f"\nFirst few examples:")
        for i, ex in enumerate(all_examples[:6], 1):
            print(f"{i}. [{ex['category']}] {ex['text']}")


if __name__ == "__main__":
    main()

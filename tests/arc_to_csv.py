import json
import csv
from pathlib import Path
from typing import List, Dict

def process_json_file(file_path: Path, is_training: bool) -> List[Dict[str, str]]:
    """Process a single JSON file and return list of rows for CSV."""
    rows = []
    
    # Determine generator file name
    folder_prefix = "arc_training" if is_training else "arc_evaluation"
    task_id = file_path.stem  # Get filename without extension
    generator_file = f"{folder_prefix}/task{task_id}.py"
    
    # Read and parse JSON file
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Process training pairs
    for pair in data.get('train', []):
        rows.append({
            'generator_file': generator_file,
            'taskvars': 'dummy_var: 1',
            'input_grid': json.dumps(pair['input']),
            'output_grid': json.dumps(pair['output'])
        })
    
    # Process test pairs
    for pair in data.get('test', []):
        rows.append({
            'generator_file': generator_file,
            'taskvars': 'dummy_var: 1',
            'input_grid': json.dumps(pair['input']),
            'output_grid': json.dumps(pair['output'])
        })
    
    return rows

def convert_arc_to_csv(base_dir: Path, output_file: str):
    """Convert all ARC JSON files to single CSV file."""
    all_rows = []
    
    # Process training folder
    training_dir = base_dir / 'training'
    if training_dir.exists():
        for json_file in training_dir.glob('*.json'):
            rows = process_json_file(json_file, is_training=True)
            all_rows.extend(rows)
    
    # Process evaluation folder
    eval_dir = base_dir / 'evaluation'
    if eval_dir.exists():
        for json_file in eval_dir.glob('*.json'):
            rows = process_json_file(json_file, is_training=False)
            all_rows.extend(rows)
    
    # Write to CSV
    if all_rows:
        fieldnames = ['generator_file', 'taskvars', 'input_grid', 'output_grid']
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        
        print(f"Successfully converted {len(all_rows)} test cases to {output_file}")
        print(f"Found {len(set(row['generator_file'] for row in all_rows))} unique generators")
    else:
        print("No data found to convert!")

def main():
    base_dir = Path('tests/')
    output_file = 'arc_data_tests.csv'
    
    convert_arc_to_csv(base_dir, output_file)

if __name__ == "__main__":
    main()

import argparse
import importlib
import os
from pathlib import Path
import sys
from typing import List, Type
from tqdm import tqdm
from arc_task_generator import ARCTaskGenerator

from execution import get_generator_class

def get_generator_files(folder: str) -> List[str]:
    """Get all Python files in a folder that contain generator classes."""
    python_files = []
    for file in Path(folder).glob('*.py'):
        if not file.name.startswith('__'):
            python_files.append(str(file))
    return python_files

def generate_dataset(generator_folders: List[str], nr_of_tasks: int, output_path: str):
    """Generate datasets for each folder and combine them."""
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_tasks_file = output_dir / 'all.csv'
    if all_tasks_file.exists():
        os.remove(all_tasks_file)
    
    # Count total number of files to process
    total_files = sum(len(get_generator_files(folder)) 
                     for folder in generator_folders 
                     if Path(folder).is_dir())
    
    # Initialize progress bar for files
    file_progress = tqdm(total=total_files, desc="Processing files")
    
    for folder in generator_folders:
        folder_path = Path(folder)
        if not folder_path.is_dir():
            print(f"Warning: {folder} is not a directory, skipping...")
            continue
            
        generator_files = get_generator_files(folder)
        if not generator_files:
            print(f"No Python files found in {folder}, skipping...")
            continue
            
        folder_output = output_dir / f"{folder_path.name}.csv"
        if folder_output.exists():
            os.remove(folder_output)
            
        # print(f"\nProcessing folder: {folder}")
        
        for gen_file in generator_files:
            try:
                # Get generator class and create instance
                generator_class = get_generator_class(gen_file)
                generator = generator_class()
                
                # print(f"\nGenerating {nr_of_tasks} tasks using {gen_file}...")
                for _ in range(nr_of_tasks):  
                    task = generator.create_task()
                    task.append_to_csv(str(folder_output))
                    task.append_to_csv(str(all_tasks_file))
                
                file_progress.update(1)  
                    
            except Exception as e:
                print(f"Error processing {gen_file}: {e}")
                file_progress.update(1)  # Update even on error
                continue
    
    file_progress.close()

def main():
    parser = argparse.ArgumentParser(description='Generate ARC task datasets.')
    parser.add_argument('--generator_folders', '-g', nargs='+',
                      help='Generator folders to use (default: all subfolders except "datasets")')
    parser.add_argument('--nr_of_tasks', '-n', type=int, default=100,
                      help='Number of tasks to create per generator (default: 100)')
    parser.add_argument('--output', '-o', type=str, default='datasets',
                      help='Path to store the outputs (default: "datasets")')
    
    args = parser.parse_args()
    
    if not args.generator_folders:
        args.generator_folders = [d for d in os.listdir('.')
                                if os.path.isdir(d) 
                                and d != 'datasets'
                                and not d.startswith('.')
                                and not d.startswith('_')]
    
    generate_dataset(args.generator_folders, args.nr_of_tasks, args.output)

if __name__ == "__main__":
    main()

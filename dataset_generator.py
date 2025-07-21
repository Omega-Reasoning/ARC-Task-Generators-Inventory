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

def generate_dataset(generator_folders: List[str], generator_files: List[str], nr_of_tasks: int, output_path: str):
    """Generate datasets for each folder and/or individual files and combine them."""
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_tasks_file = output_dir / 'all.csv'
    if all_tasks_file.exists():
        os.remove(all_tasks_file)
    
    # Collect all files to process
    all_files_to_process = []
    
    # Add files from folders
    if generator_folders:
        for folder in generator_folders:
            folder_path = Path(folder)
            if not folder_path.is_dir():
                print(f"Warning: {folder} is not a directory, skipping...")
                continue
            all_files_to_process.extend(get_generator_files(folder))
    
    # Add individual files
    if generator_files:
        for file_path in generator_files:
            if Path(file_path).exists() and file_path.endswith('.py'):
                all_files_to_process.append(file_path)
            else:
                print(f"Warning: {file_path} does not exist or is not a Python file, skipping...")
    
    if not all_files_to_process:
        print("No generator files found to process.")
        return
    
    # Initialize progress bar for files
    file_progress = tqdm(total=len(all_files_to_process), desc="Processing files")
    
    # Process folders first
    if generator_folders:
        for folder in generator_folders:
            folder_path = Path(folder)
            if not folder_path.is_dir():
                continue
                
            generator_files_in_folder = get_generator_files(folder)
            if not generator_files_in_folder:
                print(f"No Python files found in {folder}, skipping...")
                continue
                
            folder_output = output_dir / f"{folder_path.name}.csv"
            if folder_output.exists():
                os.remove(folder_output)
                
            for gen_file in generator_files_in_folder:
                try:
                    # Get generator class and create instance
                    generator_class = get_generator_class(gen_file)
                    generator = generator_class()
                    generator_name = Path(gen_file).stem
                    
                    for _ in range(nr_of_tasks):  
                        task = generator.create_task()
                        task.generator_name = generator_name
                        task.append_to_csv(str(folder_output))
                        task.append_to_csv(str(all_tasks_file))
                    
                    file_progress.update(1)  
                        
                except Exception as e:
                    print(f"Error processing {gen_file}: {e}")
                    file_progress.update(1)  # Update even on error
                    continue
    
    # Process individual files
    if generator_files:
        individual_files_output = output_dir / "individual_files.csv"
        if individual_files_output.exists():
            os.remove(individual_files_output)
            
        for gen_file in generator_files:
            if not (Path(gen_file).exists() and gen_file.endswith('.py')):
                continue
                
            try:
                # Get generator class and create instance
                generator_class = get_generator_class(gen_file)
                generator = generator_class()
                generator_name = Path(gen_file).stem
                
                for _ in range(nr_of_tasks):  
                    task = generator.create_task()
                    task.generator_name = generator_name
                    task.append_to_csv(str(individual_files_output))
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
    parser.add_argument('--generator_files', '-f', nargs='+',
                      help='Individual generator files to use')
    parser.add_argument('--nr_of_tasks', '-n', type=int, default=100,
                      help='Number of tasks to create per generator (default: 100)')
    parser.add_argument('--output', '-o', type=str, default='datasets',
                      help='Path to store the outputs (default: "datasets")')
    
    args = parser.parse_args()
    
    if not args.generator_folders and not args.generator_files:
        args.generator_folders = [d for d in os.listdir('.')
                                if os.path.isdir(d) 
                                and d != 'datasets'
                                and not d.startswith('.')
                                and not d.startswith('_')]
    
    generate_dataset(args.generator_folders or [], args.generator_files or [], args.nr_of_tasks, args.output)

if __name__ == "__main__":
    main()

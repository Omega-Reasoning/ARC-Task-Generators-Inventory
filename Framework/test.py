import argparse
import importlib
import sys
from pathlib import Path
from Framework.arc_task_generator import ARCTaskGenerator

from Framework.execution import get_generator_class

def main():
    parser = argparse.ArgumentParser(description='Generate ARC tasks from a generator class.')
    parser.add_argument('generator', type=str, 
                      help='Path to the Python file containing the generator class')
    parser.add_argument('--visualize', '-v', action='store_true',
                      help='Visualize the generated task data')
    
    args = parser.parse_args()
    
    # Get the generator class and create an instance
    generator_class = get_generator_class(args.generator)
    generator = generator_class()
    
    # Create the task
    task = generator.create_task()
    
    # Print the task
    print(task)

    print(task.data)
    
    # Visualize if requested
    if args.visualize:
        ARCTaskGenerator.visualize_train_test_data(task.data)

if __name__ == "__main__":
    main()

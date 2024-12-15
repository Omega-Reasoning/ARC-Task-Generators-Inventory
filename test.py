import argparse
import importlib
import sys
from pathlib import Path
from arc_task_generator import ARCTaskGenerator

def get_generator_class(module_path: str) -> type:
    """
    Import and return the generator class from a Python file.
    
    Parameters
    ----------
    module_path : str
        Path to the Python file containing the generator class
    
    Returns
    -------
    type
        The generator class
    """
    # Convert file path to module path
    if module_path.endswith('.py'):
        module_path = module_path[:-3]
    module_path = module_path.replace('/', '.').replace('\\', '.')
    
    try:
        # Import the module
        module = importlib.import_module(module_path)
        
        # Find the generator class (subclass of ARCTaskGenerator)
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (isinstance(attr, type) and 
                issubclass(attr, ARCTaskGenerator) and 
                attr != ARCTaskGenerator):
                return attr
                
        raise ValueError(f"No ARCTaskGenerator subclass found in {module_path}")
    
    except ImportError as e:
        print(f"Error importing module {module_path}: {e}")
        sys.exit(1)

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
    
    # Visualize if requested
    if args.visualize:
        ARCTaskGenerator.visualize_train_test_data(task.data)

if __name__ == "__main__":
    main()

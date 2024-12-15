import importlib
import sys
from arc_task_generator import ARCTaskGenerator  # Adjust import path as needed

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
    if module_path.endswith('.py'):
        module_path = module_path[:-3]
    module_path = module_path.replace('/', '.').replace('\\', '.')
    
    try:
        module = importlib.import_module(module_path)
        
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

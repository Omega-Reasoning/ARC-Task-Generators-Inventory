import numpy as np
from collections import defaultdict
from typing import Any, Callable, Dict, List, Tuple
from arc_task_generator import ARCTaskGenerator

import importlib
import sys

import transformation_library as tlib

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


def execute_transform_code(transform_code_str: str, generator_instance: Any) -> Callable:
    """
    Execute a transform code string with all necessary imports and return the callable function.

    Args:
        transform_code_str: String containing the Python function definition
        generator_instance: Instance of the generator class that will be used as 'self'

    Returns:
        Callable: The executable transform function

    Raises:
        Exception: If there's an error in executing the transform code
    """
    # Create namespace with all required imports and functions
    namespace = {
        # Core imports
        'np': np,
        'defaultdict': defaultdict,

        # Type hints
        'Dict': Dict,
        'List': List,
        'Any': Any,
        'Tuple': Tuple,

        # All transformation library functions and classes
        **{name: getattr(tlib, name) for name in dir(tlib) if not name.startswith('_')}
    }

    # Execute the transform code in the namespace
    exec(transform_code_str, namespace)

    # Return the transformed function
    return namespace['transform_input']

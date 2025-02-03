from arc_task_generator import ARCTaskGenerator, GridPair
from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Union, Tuple, List, Optional, TypedDict, NamedTuple

class GridData(TypedDict):
    input: np.ndarray
    output: Optional[Union[np.ndarray, int]]



class ObjectnessTaskGenerator(ABC):
    COLOR_MAP = {
        0: "empty",
        1: "blue",
        2: "red",
        3: "green",
        4: "yellow",
        5: "grey",
        6: "pink",
        7: "orange",
        8: "cyan",
        9: "maroon"
    }
    """Base class for generating objectness-related tasks with optional solution chains."""
    
    def __init__(self,
                 input_reasoning_chain: List[str],
                 solution_chain: List[str]):
        """
        Parameters
        ----------
        input_reasoning_chain : List[str]
            Template strings describing the input reasoning chain.
        solution_chain : List[str]
            Template strings describing the reasoning steps.
        """
        self.input_reasoning_chain = input_reasoning_chain
        self.solution_chain = solution_chain

    
    def _instantiate_templates(self, templates: List[str], vars: Dict[str, Any]) -> List[str]:
        """
        Instantiate template strings using f-string evaluation with given variables.
        color('key') is supported as a shorthand for COLOR_MAP[vars['key']]}
        """
        def color(key): return self.COLOR_MAP[vars[key]]
        
        context = {
            "vars": vars,
            "color": color
        }
        return [eval(f"f'{tmpl}'", context) for tmpl in templates]
    
    @abstractmethod
    def create_input(self, taskvars: Dict[str, Any]) -> np.ndarray:
        """
        Create an input grid according to the input reasoning chain given the task variables.
        """
        pass

    @abstractmethod
    def update_reasoning_chain(self, reasoning_chain: List[str], taskvars: Dict[str, Any]) -> List[str]:
        """
        Update the reasoning chain with the task variables.
        """
        pass

    @abstractmethod
    def answer(self, grid: np.ndarray) -> Union[int, np.ndarray]:
        """
        Compute the expected output for a given input grid.
        """
        pass

    @abstractmethod
    def create_task(self) -> Tuple[Dict[str, Any], GridData]:
        """
        Generate a single input-output pair with task variables.
        
        Returns:
            Tuple containing:
            - taskvars: Dictionary of task-specific variables
            - GridData: NamedTuple with input grid and output
        """
        pass
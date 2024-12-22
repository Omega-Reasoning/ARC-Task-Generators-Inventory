import numpy as np
from typing import Dict, List, Any, Tuple, TypedDict
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

from arc_task import ARCTask

class MatrixPair(TypedDict):
    input: np.ndarray
    output: np.ndarray

class TrainTestData(TypedDict):
    train: List[MatrixPair]
    test: List[MatrixPair]

class ARCTaskGenerator(ABC):

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

    def __init__(self,
                 input_reasoning_chain: List[str],
                 transformation_reasoning_chain: List[str]):
        """
        Parameters
        ----------
        input_reasoning_chain : List[str]
            Template strings describing the input observations.
        transformation_reasoning_chain : List[str]
            Template strings describing the reasoning steps.
        """
        self.input_reasoning_chain = input_reasoning_chain
        self.transformation_reasoning_chain = transformation_reasoning_chain

    @abstractmethod
    def create_matrices(self) -> Tuple[Dict[str, Any], TrainTestData]:
        """
        Initialise task variables used in templates and create train/test data matrices.
        
        Returns
        -------
        Tuple[Dict[str, Any], TrainTestData]
            First element: Dictionary of task variables used in templates
            Second element: Dictionary containing train and test matrices
        """
        pass

    @abstractmethod
    def create_input(self,
                     taskvars: Dict[str, Any],
                     matrixvars: Dict[str, Any]) -> np.ndarray:
        """
        Create an input matrix based according to the input reasoning chain given the task and matrix variables.
        """
        pass

    @abstractmethod
    def transform_input(self,
                        matrix: np.ndarray,
                        taskvars: Dict[str, Any]) -> np.ndarray:
        """
        Transform the input matrix according to the transformation reasoning chain, producing an output matrix.
        """
        pass

    def create_matrices_default(self, nr_train_examples, nr_test_examples, taskvars):
        """
        Creates train and test matrices using default parameters.
        
        Args:
            nr_train_examples (int): Number of training examples to generate
            nr_test_examples (int): Number of test examples to generate
            taskvars (dict): Task-specific variables used for matrix generation
        
        Returns:
            dict: Dictionary containing 'train' and 'test' lists of input/output matrix pairs
        """
        def generate_examples(n):
            return [
                {
                    'input': (input_matrix := self.create_input(taskvars, {})),
                    'output': self.transform_input(input_matrix, taskvars)
                }
                for _ in range(n)
            ]
        
        return {
            'train': generate_examples(nr_train_examples),
            'test': generate_examples(nr_test_examples)
        }

    @staticmethod
    def visualize_train_test_data(train_test_data: TrainTestData):
        colors = [
            (0, 0, 0),          # black for empty
            (30, 147, 255),     # blue
            (249, 60, 49),      # red
            (79, 204, 48),      # green
            (255, 220, 0),      # yellow
            (153, 153, 153),    # grey
            (229, 58, 163),     # pink
            (255, 137, 27),     # orange
            (135, 216, 241),    # cyan
            (146, 18, 49)       # maroon
        ]
        colors = [(r/255, g/255, b/255) for r, g, b in colors]
        cmap = plt.cm.colors.ListedColormap(colors)
        
        num_train = len(train_test_data['train'])
        num_test = len(train_test_data['test'])
        total_examples = num_train + num_test
        
        fig, axes = plt.subplots(total_examples, 2, figsize=(6, 2*total_examples))
        
        if total_examples == 1:
            axes = axes.reshape(1, 2)
        
        def plot_matrix(matrix, ax, title):
            height, width = matrix.shape
            # Plot the matrix
            im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=9)
            
            # Set ticks in the middle of each cell
            ax.set_xticks(np.arange(width))
            ax.set_yticks(np.arange(height))
            
            # Draw grid lines between cells
            ax.set_xticks(np.arange(-0.5, width, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, height, 1), minor=True)
            
            # Configure grid
            ax.grid(which="minor", color="white", linestyle='-', linewidth=1)
            ax.tick_params(which="minor", size=0)
            ax.grid(which="major", visible=False)
            
            # Set title
            ax.set_title(title)
        
        # Plot training examples
        for idx, train_example in enumerate(train_test_data['train']):
            plot_matrix(train_example['input'], axes[idx, 0], f'Train {idx+1} Input')
            plot_matrix(train_example['output'], axes[idx, 1], f'Train {idx+1} Output')
        
        # Plot test examples
        for idx, test_example in enumerate(train_test_data['test']):
            row_idx = idx + num_train
            plot_matrix(test_example['input'], axes[row_idx, 0], f'Test {idx+1} Input')
            plot_matrix(test_example['output'], axes[row_idx, 1], f'Test {idx+1} Output')
        
        plt.tight_layout()
        plt.show()

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

    def _partial_evaluation_code(self, func, taskvars: Dict[str, Any]) -> str:
        """
        Perform partial evaluation on a function by substituting task variables inline.
        """
        import inspect
        source = inspect.getsource(func)

        # We assume that taskvariables keys appear as taskvars['key'] in code and replace them with their chosen values.
        for k, v in taskvars.items():
            # Replace vtaskvars['k'] with str(v)
            source = source.replace(f"taskvars['{k}']", str(v))

        lines = source.split('\n')
        
        # Process each line
        for i, line in enumerate(lines):
            if line.strip().startswith("def"):
                # Extract the function name and return type
                import re
                pattern = r"def\s+(\w+)\s*\([^)]*\)\s*(->\s*[^:]+)?"
                match = re.match(pattern, line.strip())
                if match:
                    func_name = match.group(1)
                    return_type = match.group(2) or ""
                    # Create new function definition with only matrix parameter
                    lines[i] = f"    def {func_name}(self, matrix: np.ndarray){return_type}:"
            else:
                # Reduce indentation by 4 spaces (one level) for non-def lines
                if line.startswith(' ' * 8):  # Check if there's enough indentation to reduce
                    lines[i] = line[4:]  # Remove 4 spaces from the beginning

        # Return joined code
        return "\n".join(lines).strip()

    def create_task(self) -> ARCTask:
        # 1. Instantiate task variables and create train and test matrices
        task_variables, train_test_data = self.create_matrices()

        # 2. Instantiate observation and reasoning chains
        observation_chain = self._instantiate_templates(self.input_reasoning_chain, task_variables)
        reasoning_chain = self._instantiate_templates(self.transformation_reasoning_chain, task_variables)

        # 3. Partial evaluation of transform_input allowing us to remove the dependency on variables and store the function as string
        transform_code = self._partial_evaluation_code(self.transform_input, task_variables)

        # Create and return ARCTask
        return ARCTask(observation_chain, reasoning_chain, task_variables, transform_code, train_test_data)

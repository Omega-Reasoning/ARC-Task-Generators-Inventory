import numpy as np
from typing import Dict, List, Any, Union
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from arc_task import ARCTask

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
                 observation_chain: List[str],
                 reasoning_chain: List[str],
                 taskvars_definitions: Dict[str, Union[range, List[Any]]]):
        """
        Parameters
        ----------
        observation_chain : List[str]
            Template strings describing the input observations.
        reasoning_chain : List[str]
            Template strings describing the reasoning steps.
        taskvars_definitions : Dict[str, Union[range, List[Any]]]
            Dictionary defining the possible values for each task variable.
            For example: {'row_blocks': range(2,5), 'pattern_type': ['block_copy','color_shift']}
        """
        self.observation_chain = observation_chain
        self.reasoning_chain = reasoning_chain
        self.taskvars_definitions = taskvars_definitions

    @abstractmethod
    def create_matrices(self, taskvars: Dict[str, Any]) -> Dict[str, List[Dict[str, np.ndarray]]]:
        """
        Create train and test matrices.
        """
        pass

    @abstractmethod
    def create_input(self,
                     taskvars: Dict[str, Any],
                     matrixvars: Dict[str, Any]) -> np.ndarray:
        """
        Create an input matrix given the task and matrix variables.
        """
        pass

    @abstractmethod
    def transform_input(self,
                        matrix: np.ndarray,
                        taskvars: Dict[str, Any]) -> np.ndarray:
        """
        Transform the input matrix according to the reasoning steps,
        producing an output matrix.
        """
        pass

    @staticmethod
    def visualize_train_test_data(train_test_data: Dict):
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
        
        fig, axes = plt.subplots(total_examples, 2, figsize=(6, 3*total_examples))
        
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

    def _select_task_variables(self) -> Dict[str, Any]:
        """
        Randomly select values for the task variables from the given definitions.
        """
        selected = {}
        for var_name, var_range in self.taskvars_definitions.items():
            if isinstance(var_range, range):
                selected[var_name] = np.random.choice(list(var_range))
            elif isinstance(var_range, list):
                selected[var_name] = np.random.choice(var_range)
            else:
                raise ValueError(f"Unsupported type for variable definition {var_name}")
        return selected

    def _instantiate_templates(self, templates: List[str], variables: Dict[str, Any]) -> List[str]:
        """
        Instantiate template strings using f-string evaluation with given variables.
        """
        # we perform f-string evaluation since this allows us to perform more complex operations (e.g. multiplications) in templates
        # we assume the variables are references as vars['variable'] in the template
        return [eval(f"f'{tmpl}'", {"vars": variables}) for tmpl in templates]

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
        # 1. Instantiate task variables
        task_variables = self._select_task_variables()

        # 2. Create train and test matrices
        train_test_data = self.create_matrices(task_variables)

        # 3. Instantiate observation and reasoning chains
        observation_chain = self._instantiate_templates(self.observation_chain, task_variables)
        reasoning_chain = self._instantiate_templates(self.reasoning_chain, task_variables)

        # 4. Partial evaluation of transform_input allowing us to remove the dependency on variables and store the function as string
        transform_code = self._partial_evaluation_code(self.transform_input, task_variables)

        # Create and return ARCTask
        return ARCTask(observation_chain, reasoning_chain, task_variables, transform_code, train_test_data)

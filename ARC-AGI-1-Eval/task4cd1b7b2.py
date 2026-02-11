from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects
from input_library import create_object, random_cell_coloring, retry
import numpy as np
import random

class Task4cd1b7b2Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are squares of size {vars['rows']} x {vars['columns']}.",
            "Consider every input grid similar to a sudoku game but with colors.",
            "So, here you must have a couple of empty spots to make it look like a real color based sudoku game.",
            "All the input grids must use the same set of colors throughout."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is copied from the input grid.",
            "Just show the solution or entirely filled color based sudoku game from the input grid."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def _generate_valid_sudoku_solution(self, size: int, colors: list) -> np.ndarray:
        """Generate a valid color-based sudoku solution."""
        # For simplicity, we'll create a Latin square pattern
        grid = np.zeros((size, size), dtype=int)
        
        # Create a simple Latin square pattern
        for i in range(size):
            for j in range(size):
                # Use modular arithmetic to create a valid pattern
                color_index = (i + j) % len(colors)
                grid[i, j] = colors[color_index]
        
        # Add some randomization by swapping rows/columns
        if size > 2:
            # Randomly swap some rows
            for _ in range(random.randint(0, size // 2)):
                row1, row2 = random.sample(range(size), 2)
                grid[[row1, row2]] = grid[[row2, row1]]
            
            # Randomly swap some columns
            for _ in range(random.randint(0, size // 2)):
                col1, col2 = random.sample(range(size), 2)
                grid[:, [col1, col2]] = grid[:, [col2, col1]]
        
        return grid

    def _create_puzzle_from_solution(self, solution: np.ndarray, difficulty: float = 0.3) -> np.ndarray:
        """Create a puzzle by removing cells from the solution."""
        puzzle = solution.copy()
        total_cells = solution.size
        cells_to_remove = int(total_cells * difficulty)
        
        # Get all cell positions
        positions = [(i, j) for i in range(solution.shape[0]) for j in range(solution.shape[1])]
        
        # Randomly select cells to remove
        cells_to_clear = random.sample(positions, min(cells_to_remove, len(positions)))
        
        # Clear selected cells (set to 0)
        for row, col in cells_to_clear:
            puzzle[row, col] = 0
        
        return puzzle

    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        """Create input grid with a color-based sudoku puzzle."""
        size = taskvars['rows']
        
        # Get colors from individual task variables
        colors = []
        for i in range(size):
            color_key = f'color_{i}'
            if color_key in taskvars:
                colors.append(taskvars[color_key])
        
        # Generate a complete valid solution
        solution = self._generate_valid_sudoku_solution(size, colors)
        
        # Create puzzle by removing some cells
        difficulty = random.uniform(0.2, 0.5)  # Remove 20-50% of cells
        puzzle = self._create_puzzle_from_solution(solution, difficulty)
        
        # Store the solution for transformation
        gridvars['solution'] = solution
        gridvars['colors'] = colors
        
        return puzzle

    def transform_input(self, grid: np.ndarray) -> np.ndarray:
        """Transform input by showing the complete sudoku solution."""
        size = grid.shape[0]
        
        # Find colors used in the grid by analyzing non-zero cells
        colors = sorted(list(set(grid[grid != 0])))
        
        output_grid = grid.copy()
        
        # Simple solving approach: fill empty cells with valid colors
        empty_cells = np.where(grid == 0)
        
        for i, (row, col) in enumerate(zip(empty_cells[0], empty_cells[1])):
            # Find which colors are already used in this row and column
            used_in_row = set(output_grid[row, :]) - {0}
            used_in_col = set(output_grid[:, col]) - {0}
            used_colors = used_in_row | used_in_col
            
            # Find available colors
            available_colors = [c for c in colors if c not in used_colors]
            
            if available_colors:
                # Choose the first available color
                output_grid[row, col] = available_colors[0]
            else:
                # If no color is available, use a pattern-based approach
                # Use position-based coloring as fallback
                color_index = (row + col) % len(colors)
                output_grid[row, col] = colors[color_index]
        
        return output_grid

    def create_grids(self) -> tuple[dict[str, any], TrainTestData]:
        """Create train and test grids with consistent variables."""
        # Generate consistent colors for all grids
        grid_size = random.randint(3, 4)  # Use 3x3 or 4x4 grids
        
        all_colors = list(range(1, 10))
        random.shuffle(all_colors)
        selected_colors = all_colors[:grid_size]
        
        # Store task variables - separate each color as individual variable
        taskvars = {
            'rows': grid_size,
            'columns': grid_size,
        }
        
        # Add individual color variables
        for i, color in enumerate(selected_colors):
            taskvars[f'color_{i}'] = color
        
        # Generate 3-5 training examples
        num_train_examples = random.randint(3, 5)
        train_examples = []
        
        # Create training examples
        for i in range(num_train_examples):
            gridvars = {'grid_size': grid_size}
            
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid)
            
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Create test example
        test_gridvars = {'grid_size': grid_size}
        test_input = self.create_input(taskvars, test_gridvars)
        test_output = self.transform_input(test_input)
        
        test_examples = [{
            'input': test_input,
            'output': test_output
        }]
        
        return taskvars, {
            'train': train_examples,
            'test': test_examples
        }

# Test code
if __name__ == "__main__":
    generator = ColorSudokuTaskGenerator()
    taskvars, train_test_data = generator.create_grids()
    
    print("Task Variables:", taskvars)
    print(f"Number of train examples: {len(train_test_data['train'])}")
    print(f"Number of test examples: {len(train_test_data['test'])}")
    
    # Print grid information
    for i, example in enumerate(train_test_data['train']):
        input_shape = example['input'].shape
        output_shape = example['output'].shape
        input_empty_cells = np.sum(example['input'] == 0)
        print(f"Train example {i+1}: {input_shape[0]}x{input_shape[1]}, {input_empty_cells} empty cells")
        
        # Print colors used
        colors_used = []
        size = taskvars['rows']
        for j in range(size):
            color_key = f'color_{j}'
            if color_key in taskvars:
                colors_used.append(taskvars[color_key])
        print(f"  Colors used: {sorted(colors_used)}")
    
    test_shape = train_test_data['test'][0]['input'].shape
    test_empty_cells = np.sum(train_test_data['test'][0]['input'] == 0)
    print(f"Test example: {test_shape[0]}x{test_shape[1]}, {test_empty_cells} empty cells")
    
    # Visualize if the visualization method is available
    try:
        ARCTaskGenerator.visualize_train_test_data(train_test_data)
    except:
        print("Visualization not available, but grids created successfully!")
        
        # Print a sample grid for verification
        print("\nSample Input Grid:")
        print(train_test_data['train'][0]['input'])
        print("\nSample Output Grid:")
        print(train_test_data['train'][0]['output'])
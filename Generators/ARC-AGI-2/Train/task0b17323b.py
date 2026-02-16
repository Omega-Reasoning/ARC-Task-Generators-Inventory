from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

class Task0b17323bGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['rows']} x {vars['rows']}.",
            "Each grid contains {vars['rows']//5} single-colored cells of {color('cell_color')} arranged along the main diagonal (from top-left to bottom-right), with all other cells being empty (0).",
            "The {color('cell_color')} cells start from a position on the main diagonal that may or may not be (0,0).",
            "The {color('cell_color')} cells extend from this starting position up to the middle or slightly beyond the middle of the main diagonal.",
            "The {color('cell_color')} cells are placed at fixed intervals along the diagonal, with a consistent gap (number of empty (0) cells) between consecutive colored cells.",
            "If the colored cells do not start at (0,0), then the number of empty cells before the first colored cell on the main diagonal equals the fixed gap between all subsequent colored cells.",
            "The gap between two consecutive {color('cell_color')} cells should be at least 1 and at most {(vars['rows'] //5) - 1} empty cells.",
            "The fixed gap between two consecutive {color('cell_color')} cells within a grid, varies across examples."
        ]
        
        transformation_reasoning_chain = [
            "The output grids are constructed by copying the input grids and identifying the number of empty (0) cells on the main diagonal (top-left to bottom-right) between two consecutive {color('cell_color')} cells.",
            "Once the gap is identified, new {color('new_color')} cells are added on the main diagonal, starting immediately after the last existing {color('cell_color')} cell plus the identified gap, maintaining the same fixed gap between these {color('new_color')}  cells as found between the {color('cell_color')} colored cells.",
            "All other cells in the grid remain unchanged."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self) -> tuple[dict[str, any], TrainTestData]:
        # Initialize task variables
        rows = random.choice([ 15, 20, 25, 30])
        
        # Choose two distinct colors
        available_colors = list(range(1, 10))
        cell_color = random.choice(available_colors)
        available_colors.remove(cell_color)
        new_color = random.choice(available_colors)
        
        taskvars = {
            'rows': rows,
            'cell_color': cell_color,
            'new_color': new_color
        }
        
        # Determine possible gaps and starting positions
        max_gap = max(1, min((rows // 5) - 1, 3))  # Ensure at least 1, at most 3
        max_start = max(0, min((rows // 5) - 2, (rows // 5) - 2))  # Ensure at least 0
        
        # Create all valid combinations of gaps and starting positions
        all_combinations = [(gap, start) for gap in range(1, max_gap + 1) 
                                        for start in range(0, max_start + 1)]
        
        # Shuffle the combinations
        random.shuffle(all_combinations)
        
        # Select 4 combinations (3 for train, 1 for test)
        selected_combinations = all_combinations[:4] if len(all_combinations) >= 4 else all_combinations + [(1, 0)] * (4 - len(all_combinations))
        
        # Generate train examples
        train_examples = []
        for i in range(3):  # 3 train examples
            gap, start_pos = selected_combinations[i]
            gridvars = {'gap': gap, 'start_pos': start_pos}
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            train_examples.append({'input': input_grid, 'output': output_grid})
        
        # Generate test example
        gap, start_pos = selected_combinations[3]  # Use the 4th combination
        test_gridvars = {'gap': gap, 'start_pos': start_pos}
        test_input = self.create_input(taskvars, test_gridvars)
        test_output = self.transform_input(test_input, taskvars)
        
        return taskvars, {
            'train': train_examples,
            'test': [{'input': test_input, 'output': test_output}]
        }
    
    def create_input(self, taskvars: dict[str, any], gridvars: dict[str, any]) -> np.ndarray:
        rows = taskvars['rows']
        cell_color = taskvars['cell_color']
        
        # Get gap and starting position from gridvars
        gap = gridvars.get('gap', 1)
        start_pos = gridvars.get('start_pos', 0)
        
        # Create empty grid
        grid = np.zeros((rows, rows), dtype=int)
        
        # Calculate number of colored cells
        num_cells = rows // 5
        
        # Place colored cells on the diagonal with the specified gap
        for i in range(num_cells):
            pos = start_pos + i * (gap + 1)
            if pos < rows:  # Ensure we're still within grid bounds
                grid[pos, pos] = cell_color
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: dict[str, any]) -> np.ndarray:
        rows = grid.shape[0]
        cell_color = taskvars['cell_color']
        new_color = taskvars['new_color']
        
        # Create a copy of the input grid
        output_grid = grid.copy()
        
        # Find all colored cells on the diagonal
        diagonal_positions = []
        for i in range(rows):
            if grid[i, i] == cell_color:
                diagonal_positions.append(i)
        
        if len(diagonal_positions) >= 2:
            # Calculate the gap between consecutive colored cells
            gap = diagonal_positions[1] - diagonal_positions[0] - 1
            
            # Find the position of the last colored cell
            last_pos = diagonal_positions[-1]
            
            # Continue the pattern with new color
            current_pos = last_pos + gap + 1
            while current_pos < rows:
                output_grid[current_pos, current_pos] = new_color
                current_pos += gap + 1
        
        return output_grid


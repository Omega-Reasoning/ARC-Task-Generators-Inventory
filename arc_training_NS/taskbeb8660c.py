from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects, GridObjects, GridObject
from input_library import random_cell_coloring, retry
import numpy as np
import random
from typing import Dict, Any, Tuple, List

class Taskbeb8660c(ARCTaskGenerator):
    
    def __init__(self):
        input_reasoning_chain = [
            "All input grids are of size n × {vars['m']}, where n varies across grids but is always greater than or equal to {vars['m']}.",
            "In each input grid, {vars['m']} rows are randomly selected.",
            "In every selected row, a contiguous sequence of cells is chosen, with a length less than or equal to the number of columns.",
            "The sequence lengths are all distinct, ensuring that each row contains a sequence of a different size.",
            "Each sequence is uniformly colored with a unique color, guaranteeing that no two rows share the same color."
        ]
        
        transformation_reasoning_chain = [
            "The sequences that exist in the input grid are sorted in ascending order by their length.",
            "The output grid is constructed by stacking these sequences at the bottom of the grid, with the shortest sequence placed above and the longest sequence placed on the bottom row.",
            "Each sequence is right-aligned, so that its rightmost cell coincides with the right boundary of the output grid."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        m = taskvars['m']
        n = gridvars['n']  # Row count varies per grid
        
        # Create empty grid
        grid = np.zeros((n, m), dtype=int)
        
        # Generate m distinct sequence lengths (1 to m)
        sequence_lengths = random.sample(range(1, m + 1), m)
        
        # Select m random rows
        selected_rows = random.sample(range(n), m)
        
        # Available colors (excluding background 0)
        available_colors = list(range(1, 10))
        random.shuffle(available_colors)
        
        # Place sequences in selected rows
        for i, row_idx in enumerate(selected_rows):
            length = sequence_lengths[i]
            color = available_colors[i]
            
            # Choose random starting position for the sequence
            start_col = random.randint(0, m - length)
            
            # Place the sequence
            for j in range(length):
                grid[row_idx, start_col + j] = color
                
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        m = taskvars['m']
        n = grid.shape[0]  # Get actual height from the input grid
        
        # Find all sequences (horizontal contiguous objects)
        sequences = []
        
        # Check each row for contiguous sequences
        for r in range(n):
            current_sequence = []
            current_color = 0
            
            for c in range(m):
                if grid[r, c] != 0:  # Non-background cell
                    if grid[r, c] == current_color:
                        # Continue current sequence
                        current_sequence.append((r, c, grid[r, c]))
                    else:
                        # Start new sequence or end previous one
                        if current_sequence:
                            sequences.append(current_sequence)
                        current_sequence = [(r, c, grid[r, c])]
                        current_color = grid[r, c]
                else:
                    # Background cell - end current sequence
                    if current_sequence:
                        sequences.append(current_sequence)
                        current_sequence = []
                        current_color = 0
            
            # Don't forget the last sequence in the row
            if current_sequence:
                sequences.append(current_sequence)
        
        # Sort sequences by length (ascending)
        sequences.sort(key=len)
        
        # Create output grid with same dimensions as input
        output_grid = np.zeros((n, m), dtype=int)
        
        # Place sequences at the bottom, right-aligned
        for i, sequence in enumerate(sequences):
            target_row = n - len(sequences) + i  # Bottom rows
            sequence_length = len(sequence)
            color = sequence[0][2]  # Get color from first cell
            
            # Right-align: start from rightmost position
            start_col = m - sequence_length
            
            # Place the sequence
            for j in range(sequence_length):
                output_grid[target_row, start_col + j] = color
        
        return output_grid
    
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Generate fixed column count for all grids in this task (between 5 and 30)
        m = random.randint(5, 9)  # Number of columns (fixed for all grids)
        
        taskvars = {
            'm': m
        }
        
        # Generate training examples with varying row counts
        num_train = random.randint(3, 6)
        train_examples = []
        
        for _ in range(num_train):
            # Each grid gets its own random number of rows (between m and 30)
            n = random.randint(m, 30)  # Ensure n >= m and within ARC constraints
            # Create a fresh gridvars dict for this specific grid
            current_gridvars = {'n': n}
            
            input_grid = self.create_input(taskvars, current_gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Generate test example with its own different row count
        n_test = random.randint(m, 30)
        test_gridvars = {'n': n_test}
        test_input = self.create_input(taskvars, test_gridvars)
        test_output = self.transform_input(test_input, taskvars)
        test_examples = [{
            'input': test_input,
            'output': test_output
        }]
        
        train_test_data = {
            'train': train_examples,
            'test': test_examples
        }
        
        return taskvars, train_test_data

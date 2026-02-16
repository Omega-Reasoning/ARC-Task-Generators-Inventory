from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from typing import Dict, Any, Tuple, List
import numpy as np
import random

class Taskbeb8660cGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "The input grids are of size n x m, where n ≥ m.",
            "In each input grid, m rows are randomly selected.",
            "In every selected row, a contiguous sequence of cells is chosen, with a length less than or equal to the number of columns.",
            "The sequence lengths are all distinct, ensuring that each row contains a sequence of a different size.",
            "Each sequence is uniformly colored with a unique color, guaranteeing that no two rows share the same color."
        ]
        
        transformation_reasoning_chain = [
            "Output grids are of the same size of the input grids.",
            "The sequences that exist in the input grid are sorted in descending order by their length.",
            "The output grid is constructed by stacking these sequences at the {vars['i']} of the grid, with the longest sequence placed on the {vars['i']} row.",
            "Each sequence is {vars['j']} aligned, so that its {vars['j']}most cell coincides with the {vars['j']} boundary of the output grid."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        rows = gridvars['rows']
        cols = gridvars['cols']
        num_sequences = cols  # m sequences for n x m grid
        
        grid = np.zeros((rows, cols), dtype=int)
        
        # Generate distinct sequence lengths (1 to cols)
        sequence_lengths = random.sample(range(1, cols + 1), num_sequences)
        
        # Select random rows (m rows out of n)
        selected_rows = random.sample(range(rows), num_sequences)
        
        # Generate unique colors (1-9, excluding 0 which is background)
        available_colors = list(range(1, 10))
        random.shuffle(available_colors)
        colors = available_colors[:num_sequences]
        
        # Place sequences in selected rows
        for row_idx, length, color in zip(selected_rows, sequence_lengths, colors):
            # Random starting position for the sequence
            max_start = cols - length
            start_col = random.randint(0, max_start)
            
            # Place the sequence
            grid[row_idx, start_col:start_col + length] = color
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        rows, cols = grid.shape
        output = np.zeros((rows, cols), dtype=int)
        
        # Extract sequences from input
        sequences = []
        for r in range(rows):
            row = grid[r]
            non_zero = np.where(row != 0)[0]
            if len(non_zero) > 0:
                # Found a sequence
                start = non_zero[0]
                end = non_zero[-1] + 1
                length = end - start
                color = row[start]
                sequences.append({'length': length, 'color': color})
        
        # Sort by length (descending - longest first)
        sequences.sort(key=lambda x: x['length'], reverse=True)
        
        # Stack sequences based on alignment
        alignment = taskvars['i']  # 'top' or 'bottom'
        horizontal = taskvars['j']  # 'left' or 'right'
        
        for idx, seq in enumerate(sequences):
            length = seq['length']
            color = seq['color']
            
            # Determine row position
            if alignment == 'top':
                # Longest at top (row 0), then descending
                row_idx = idx
            else:  # bottom
                # Longest at bottom (last row with sequences), then ascending
                row_idx = rows - 1 - idx
            
            # Determine column position
            if horizontal == 'left':
                output[row_idx, :length] = color
            else:  # right
                output[row_idx, cols - length:] = color
        
        return output
    
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Randomly choose alignment parameters (consistent across all examples)
        taskvars = {
            'i': random.choice(['top', 'bottom']),
            'j': random.choice(['left', 'right'])
        }
        
        # Create 3-6 training examples
        num_train = random.randint(3, 6)
        train_pairs = []
        
        for _ in range(num_train):
            # Random grid dimensions (n >= m, both between 5 and 30)
            # m (cols) can be at most 9 since we only have 9 non-background colors
            cols = random.randint(3, 9)
            rows = random.randint(cols, min(30, cols + 20))
            
            gridvars = {
                'rows': rows,
                'cols': cols
            }
            
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            train_pairs.append({'input': input_grid, 'output': output_grid})
        
        # Create test example
        cols = random.randint(3, 9)
        rows = random.randint(cols, min(30, cols + 20))
        
        gridvars = {
            'rows': rows,
            'cols': cols
        }
        
        test_input = self.create_input(taskvars, gridvars)
        test_output = self.transform_input(test_input, taskvars)
        test_pairs = [{'input': test_input, 'output': test_output}]
        
        return taskvars, {'train': train_pairs, 'test': test_pairs}



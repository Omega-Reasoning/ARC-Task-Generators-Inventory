from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from typing import Dict, List, Any, Tuple


class Task221dfab4Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['rows']} × {vars['cols']}.",
            "Each input grid has a completely filled background of {color('background_color')} and contains several clusters of {color('cluster_color')} color scattered across, resembling clouds.",
            "To construct these {color('cluster_color')} clusters, connect horizontal strips of different sizes vertically, ensuring that their centers are aligned.",
            "Once this is done, place a {color('strip_color')} horizontal strip, 3 or 4 cells wide, anywhere in the last row—excluding the extreme left and right ends."
        ]
        
        transformation_reasoning_chain = [
            "The output grids are constructed by copying the input grids and identifying the {color('strip_color')} strip located in the last row.",
            "From there, add additional strips upward toward the first row, forming a vertical ladder with different colored strips arranged in a fixed pattern.",
            "Begin with the {color('strip_color')} strip in the last row, then add a {color('background_color')} strip, followed by another {color('strip_color')} strip, then again a {color('background_color')} strip, and finally a {color('newstrip_color')} strip followed by {color('background_color')} strip. Repeat this sequence until the first row is reached.",
            "Ensure that all strips are perfectly vertically aligned with the strip in the last row and that they have the exact same length as the {color('strip_color')} in the last row.",
            "For rows containing a {color('newstrip_color')} strip, ensure that any {color('cluster_color')} cells within the same row are also filled with the {color('newstrip_color')}."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Initialize task variables
        taskvars = {
            'rows': random.choice(range(11, 31, 2)) ,
            'cols': random.randint(10, 30),  # odd numbers between 11 and 29
            'background_color': random.randint(1, 9),
            'cluster_color': random.randint(1, 9),
            'strip_color': random.randint(1, 9),
            'newstrip_color': random.randint(1, 9)
        }
        
        # Ensure all colors are different
        colors = [taskvars['background_color'], taskvars['cluster_color'], 
                 taskvars['strip_color'], taskvars['newstrip_color']]
        while len(set(colors)) != 4:
            taskvars['cluster_color'] = random.randint(1, 9)
            taskvars['strip_color'] = random.randint(1, 9) 
            taskvars['newstrip_color'] = random.randint(1, 9)
            colors = [taskvars['background_color'], taskvars['cluster_color'],
                     taskvars['strip_color'], taskvars['newstrip_color']]
        
        # Generate train examples (3-5) with normal strip placement (last row)
        num_train = random.randint(3, 5)
        train_examples = []
        
        for _ in range(num_train):
            gridvars = {'test_mode': False}  # Normal mode - strip in last row
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            train_examples.append({'input': input_grid, 'output': output_grid})
        
        # Generate test example with different grid size and strip in first row
        # Create different grid size for test
        test_rows = random.randint(10, 30)
        test_cols = random.choice(range(11, 31, 2))
        # Ensure test grid is different from training grids
        while test_rows == taskvars['rows'] and test_cols == taskvars['cols']:
            test_rows = random.randint(10, 30)
            test_cols = random.choice(range(11, 31, 2))
        
        gridvars = {'test_mode': True, 'rows': test_rows, 'cols': test_cols}  # Test mode - strip in first row
        test_input = self.create_input(taskvars, gridvars)
        test_output = self.transform_input(test_input, taskvars)
        test_examples = [{'input': test_input, 'output': test_output}]
        
        return taskvars, {'train': train_examples, 'test': test_examples}
    
    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        # Use test-specific dimensions if provided, otherwise use main task variables
        rows = gridvars.get('rows', taskvars['rows'])
        cols = gridvars.get('cols', taskvars['cols'])
        
        background_color = taskvars['background_color']
        cluster_color = taskvars['cluster_color']
        strip_color = taskvars['strip_color']
        test_mode = gridvars.get('test_mode', False)
        
        # Create grid filled with background color
        grid = np.full((rows, cols), background_color, dtype=int)
        
        # Create cloud clusters - horizontal strips of varying sizes connected vertically
        num_clusters = random.randint(5, 8)  # Increased number of clusters
        
        for _ in range(num_clusters):
            # Choose a random center column for the cluster
            center_col = random.randint(2, cols - 3)
            
            # Create 2-5 horizontal strips of varying lengths
            num_strips = random.randint(2, 5)
            start_row = random.randint(1, rows - num_strips - 2)
            
            for i in range(num_strips):
                # Vary strip length but keep centers aligned
                strip_length = random.randint(2, min(8, cols // 2))
                start_col = max(0, center_col - strip_length // 2)
                end_col = min(cols, start_col + strip_length)
                
                # Fill the strip
                grid[start_row + i, start_col:end_col] = cluster_color
        
        # Place horizontal strip - in first row for test, last row for training
        strip_width = random.choice([3, 4])
        start_pos = random.randint(2, cols - strip_width - 2)
        
        if test_mode:
            # Place strip in first row for test
            grid[0, start_pos:start_pos + strip_width] = strip_color
        else:
            # Place strip in last row for training
            grid[rows - 1, start_pos:start_pos + strip_width] = strip_color
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        output_grid = grid.copy()
        rows, cols = grid.shape
        
        background_color = taskvars['background_color']
        cluster_color = taskvars['cluster_color'] 
        strip_color = taskvars['strip_color']
        newstrip_color = taskvars['newstrip_color']
        
        # Pattern: strip_color, background_color, strip_color, background_color, newstrip_color, background_color
        pattern = [strip_color, background_color, strip_color, background_color, newstrip_color, background_color]
        
        # Check if strip is in first row (test mode) or last row (training mode)
        first_row = grid[0, :]
        last_row = grid[rows - 1, :]
        
        strip_in_first = np.any(first_row == strip_color)
        strip_in_last = np.any(last_row == strip_color)
        
        if strip_in_first:
            # Test mode: strip in first row, fill downward
            strip_positions = np.where(first_row == strip_color)[0]
            strip_start = strip_positions[0]
            strip_end = strip_positions[-1] + 1
            
            # Fill strips downward from the first row
            for row in range(rows):
                pattern_index = row % len(pattern)
                color = pattern[pattern_index]
                
                # Fill the strip area
                output_grid[row, strip_start:strip_end] = color
                
                # If this is a newstrip_color row, also convert any cluster_color cells in the same row
                if color == newstrip_color:
                    cluster_positions = np.where(output_grid[row, :] == cluster_color)[0]
                    output_grid[row, cluster_positions] = newstrip_color
        
        elif strip_in_last:
            # Training mode: strip in last row, fill upward
            strip_positions = np.where(last_row == strip_color)[0]
            strip_start = strip_positions[0]
            strip_end = strip_positions[-1] + 1
            
            # Fill strips upward from the last row
            for row in range(rows - 1, -1, -1):
                pattern_index = (rows - 1 - row) % len(pattern)
                color = pattern[pattern_index]
                
                # Fill the strip area
                output_grid[row, strip_start:strip_end] = color
                
                # If this is a newstrip_color row, also convert any cluster_color cells in the same row
                if color == newstrip_color:
                    cluster_positions = np.where(output_grid[row, :] == cluster_color)[0]
                    output_grid[row, cluster_positions] = newstrip_color
        
        return output_grid


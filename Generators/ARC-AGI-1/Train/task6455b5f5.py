from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from Framework.transformation_library import find_connected_objects
from Framework.input_library import retry

class Task6455b5f5Generator(ARCTaskGenerator):

    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of different sizes.",
            "They contain {color('line_color')} vertical and horizontal lines, while the remaining cells are empty (0).",
            "These lines completely divide the grid into multiple subgrids of varying sizes.",
            "Among these subgrids, exactly one is the largest in terms of area."
        ]
        
        transformation_reasoning_chain = [
            "The output grids are constructed by copying the input grids and identifying the largest and smallest subgrids created by the {color('line_color')} lines.",
            "The largest subgrid is filled with {color('fill_color1')}, and the smallest subgrid is filled with {color('fill_color2')}.",
            "If there are more than one subgrids that have the smallest size, all of them are filled with {color('fill_color2')}.",
            "All other cells remain unchanged."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self):
        
        taskvars = self._generate_task_variables()
        
        # Generate train and test examples
        num_train_examples = 4
        train_examples = []
        
        for _ in range(num_train_examples):
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Generate test example
        test_input = self.create_input(taskvars, {})
        test_output = self.transform_input(test_input, taskvars)
        
        return taskvars, {
            'train': train_examples,
            'test': [{
                'input': test_input,
                'output': test_output
            }]
        }
    
    def _generate_task_variables(self):
        # Generate three different colors
        colors = random.sample(range(1, 10), 3)
        return {
            'line_color': colors[0],
            'fill_color1': colors[1],  # For largest subgrid
            'fill_color2': colors[2]   # For smallest subgrid
        }
    
    def create_input(self, taskvars, gridvars):
        # Determine grid size (randomized but reasonable)
        height = random.randint(9, 30)
        width = random.randint(9, 30)
        
        # Initialize empty grid
        grid = np.zeros((height, width), dtype=int)
        
        # Draw horizontal and vertical lines
        line_color = taskvars['line_color']
        
        # Function to check if the generated grid produces a single largest subgrid and at least one smallest subgrid
        def is_valid_subgrid_structure(grid):
            try:
                subgrid_sizes = self._get_subgrid_sizes(grid, taskvars)
                
                # Must have at least 3 subgrids for interesting problems
                if len(subgrid_sizes) < 3:
                    return False
                
                # Count occurrences of largest size
                sizes = [size for _, size in subgrid_sizes]
                max_size = max(sizes)
                max_count = sizes.count(max_size)
                
                # There should be exactly one largest subgrid
                if max_count != 1:
                    return False
                
                # There should be at least one and at most three smallest subgrids
                min_size = min(sizes)
                min_count = sizes.count(min_size)
                if min_count < 1 or min_count > 3:
                    return False
                
                return True
            except:
                return False
        
        # Retry until we get a valid grid with the desired subgrid structure
        def generate_grid():
            grid = np.zeros((height, width), dtype=int)
            
            # Add horizontal lines (at least 1, spaced apart)
            # Calculate maximum number of horizontal lines we can fit with spacing
            max_h_lines = (height - 1) // 2  # At most every other row can be a line
            num_h_lines = random.randint(1, min(4, max_h_lines))
            
            # Select positions ensuring no two lines are adjacent
            h_positions = []
            available_h_positions = list(range(1, height-1))
            
            for _ in range(num_h_lines):
                if not available_h_positions:
                    break
                    
                pos = random.choice(available_h_positions)
                h_positions.append(pos)
                
                # Remove this position and adjacent positions from available positions
                available_h_positions = [p for p in available_h_positions 
                                        if abs(p - pos) > 1]
            
            # Sort the positions for clarity
            h_positions.sort()
            
            # Draw horizontal lines
            for row in h_positions:
                grid[row, :] = line_color
            
            # Add vertical lines (at least 1, spaced apart)
            # Calculate maximum number of vertical lines we can fit with spacing
            max_v_lines = (width - 1) // 2  # At most every other column can be a line
            num_v_lines = random.randint(1, min(4, max_v_lines))
            
            # Select positions ensuring no two lines are adjacent
            v_positions = []
            available_v_positions = list(range(1, width-1))
            
            for _ in range(num_v_lines):
                if not available_v_positions:
                    break
                    
                pos = random.choice(available_v_positions)
                v_positions.append(pos)
                
                # Remove this position and adjacent positions from available positions
                available_v_positions = [p for p in available_v_positions 
                                        if abs(p - pos) > 1]
            
            # Sort the positions for clarity
            v_positions.sort()
            
            # Draw vertical lines
            for col in v_positions:
                grid[:, col] = line_color
                
            return grid
        
        return retry(generate_grid, is_valid_subgrid_structure, max_attempts=100)
    
    def _get_subgrid_sizes(self, grid, taskvars):
        # Copy grid to avoid modifying the original
        grid_copy = grid.copy()
        
        # Find all empty (0) regions bordered by lines
        subgrids = find_connected_objects(grid_copy, diagonal_connectivity=False, background=taskvars['line_color'], monochromatic=True)
        
        # Calculate size of each subgrid
        subgrid_sizes = [(i, len(subgrid)) for i, subgrid in enumerate(subgrids.objects)]
        
        return subgrid_sizes
    
    def transform_input(self, grid, taskvars):
        # Copy input grid
        output = grid.copy()
        line_color = taskvars['line_color']
        fill_color1 = taskvars['fill_color1']
        fill_color2 = taskvars['fill_color2']
        
        # Find all connected regions (subgrids)
        subgrids = find_connected_objects(output, diagonal_connectivity=False, background=line_color, monochromatic=True)
        
        # Calculate size of each subgrid
        subgrid_sizes = [(i, len(subgrid)) for i, subgrid in enumerate(subgrids.objects)]
        
        # Find largest and smallest subgrids
        if not subgrid_sizes:
            return output  # Return unchanged if no subgrids found
        
        max_size = max(size for _, size in subgrid_sizes)
        min_size = min(size for _, size in subgrid_sizes)
        
        # Fill largest subgrid with fill_color1
        largest_indices = [i for i, size in subgrid_sizes if size == max_size]
        for i in largest_indices:
            for r, c, _ in subgrids.objects[i]:
                output[r, c] = fill_color1
        
        # Fill smallest subgrids with fill_color2
        smallest_indices = [i for i, size in subgrid_sizes if size == min_size]
        for i in smallest_indices:
            for r, c, _ in subgrids.objects[i]:
                output[r, c] = fill_color2
        
        return output


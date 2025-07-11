from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from input_library import retry
import numpy as np
import random
from typing import Dict, Any, Tuple, List

class Taskf1cefba8(ARCTaskGenerator):
    
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['rows']} x {vars['columns']}.",
            "Within each input grid, there is a single inner rectangle filled with a random color, surrounded by a border of thickness two in a different random color.",
            "The entire structure (inner rectangle + border) is positioned such that it is fully surrounded by empty cells (0) and is not necessarily centered in the grid.",
            "Additionally, some random border cells directly adjacent to the inner rectangle are colored with the same color as the inner rectangle, creating slight irregularities along the edge.",
            "Both the inner rectangle color and the border color are chosen randomly for each input to ensure diversity.",
            "The size and position of the inner rectangle vary across inputs to ensure diversity."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid and identifying one inner rectangle, the border of size two surrounding it, their respective colors, and the adjacent cells along the edge of the inner rectangle that share its color.",
            "For each of these adjacent cells: If the cell is adjacent to the left or right of the inner rectangle, the entire row containing that cell is colored with the border color of the full structure (inner rectangle + border). On the same row, the horizontal distance between the edges of the full structure and the left and right boundaries of the grid is filled with the color of the inner rectangle.",
            "For each of these adjacent cells: If the cell is adjacent to the top or bottom of the inner rectangle, the entire column containing that cell is colored with the border color of the full structure. On the same column, the vertical distance between the edges of the full structure and the top and bottom boundaries of the grid is filled with the color of the inner rectangle."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
        
        # Store information about the structure for transformation
        self.structure_info = None
    
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Generate random grid dimensions
        rows = random.randint(10, 25)
        columns = random.randint(10, 25)
        
        taskvars = {
            'rows': rows,
            'columns': columns
        }
        
        # Create 3-6 training examples and 1 test example
        num_train = random.randint(3, 6)
        
        train_examples = []
        for _ in range(num_train):
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            train_examples.append({'input': input_grid, 'output': output_grid})
        
        test_examples = []
        input_grid = self.create_input(taskvars, {})
        output_grid = self.transform_input(input_grid, taskvars)
        test_examples.append({'input': input_grid, 'output': output_grid})
        
        return taskvars, {'train': train_examples, 'test': test_examples}
    
    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        rows = taskvars['rows']
        columns = taskvars['columns']
        
        # Create empty grid
        grid = np.zeros((rows, columns), dtype=int)
        
        # Choose colors (ensure they're different from background and each other)
        available_colors = [i for i in range(1, 10)]
        inner_color = random.choice(available_colors)
        border_color = random.choice([c for c in available_colors if c != inner_color])
        
        # Generate inner rectangle dimensions and position
        # Need to ensure the full structure (inner + 2-thick border) fits with padding
        min_inner_width = 2
        min_inner_height = 2
        max_inner_width = columns - 6  # Leave space for border (2) + padding (2) on each side
        max_inner_height = rows - 6
        
        if max_inner_width < min_inner_width or max_inner_height < min_inner_height:
            # Fallback for very small grids
            inner_width = max(1, max_inner_width)
            inner_height = max(1, max_inner_height)
        else:
            inner_width = random.randint(min_inner_width, max_inner_width)
            inner_height = random.randint(min_inner_height, max_inner_height)
        
        # Position the structure (inner + border) with at least 1 empty cell around it
        struct_width = inner_width + 4  # inner + 2 border on each side
        struct_height = inner_height + 4
        
        max_start_col = columns - struct_width - 1
        max_start_row = rows - struct_height - 1
        
        if max_start_col < 1 or max_start_row < 1:
            # Fallback positioning
            start_col = max(1, max_start_col)
            start_row = max(1, max_start_row)
        else:
            start_col = random.randint(1, max_start_col)
            start_row = random.randint(1, max_start_row)
        
        # Draw the structure
        inner_start_row = start_row + 2
        inner_end_row = inner_start_row + inner_height
        inner_start_col = start_col + 2
        inner_end_col = inner_start_col + inner_width
        
        # Draw border (2-thick)
        for r in range(start_row, start_row + struct_height):
            for c in range(start_col, start_col + struct_width):
                if (r < inner_start_row or r >= inner_end_row or 
                    c < inner_start_col or c >= inner_end_col):
                    grid[r, c] = border_color
        
        # Draw inner rectangle
        for r in range(inner_start_row, inner_end_row):
            for c in range(inner_start_col, inner_end_col):
                grid[r, c] = inner_color
        
        # Add random adjacent cells with inner color
        adjacent_positions = []
        
        # Find border cells adjacent to inner rectangle
        for r in range(inner_start_row, inner_end_row):
            # Left border adjacent
            if inner_start_col - 1 >= start_col:
                adjacent_positions.append((r, inner_start_col - 1, 'left'))
            # Right border adjacent  
            if inner_end_col < start_col + struct_width:
                adjacent_positions.append((r, inner_end_col, 'right'))
        
        for c in range(inner_start_col, inner_end_col):
            # Top border adjacent
            if inner_start_row - 1 >= start_row:
                adjacent_positions.append((inner_start_row - 1, c, 'top'))
            # Bottom border adjacent
            if inner_end_row < start_row + struct_height:
                adjacent_positions.append((inner_end_row, c, 'bottom'))
        
        # Randomly select some adjacent positions to color with inner color
        selected_adjacent = []
        if adjacent_positions:
            num_to_color = random.randint(1, min(4, len(adjacent_positions)))
            selected = random.sample(adjacent_positions, num_to_color)
            
            for r, c, direction in selected:
                if 0 <= r < rows and 0 <= c < columns:
                    grid[r, c] = inner_color
                    selected_adjacent.append((r, c, direction))
        
        # Store structure information for transformation
        self.structure_info = {
            'inner_color': inner_color,
            'border_color': border_color,
            'inner_bounds': (inner_start_row, inner_end_row, inner_start_col, inner_end_col),
            'struct_bounds': (start_row, start_row + struct_height, start_col, start_col + struct_width),
            'adjacent_cells': selected_adjacent
        }
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        if self.structure_info is None:
            return grid.copy()
            
        output = grid.copy()
        rows, columns = grid.shape
        
        inner_color = self.structure_info['inner_color']
        border_color = self.structure_info['border_color']
        struct_min_row, struct_max_row, struct_min_col, struct_max_col = self.structure_info['struct_bounds']
        adjacent_cells = self.structure_info['adjacent_cells']
        
        # Process each adjacent cell according to the transformation rules
        for r, c, direction in adjacent_cells:
            if direction in ['left', 'right']:
                # Fill entire row with border color
                output[r, :] = border_color
                
                # Fill horizontal distances outside the structure with inner color
                # Left side: from column 0 to struct_min_col-1
                if struct_min_col > 0:
                    output[r, :struct_min_col] = inner_color
                
                # Right side: from struct_max_col to end
                if struct_max_col < columns:
                    output[r, struct_max_col:] = inner_color
                
            elif direction in ['top', 'bottom']:
                # Fill entire column with border color
                output[:, c] = border_color
                
                # Fill vertical distances outside the structure with inner color
                # Top side: from row 0 to struct_min_row-1
                if struct_min_row > 0:
                    output[:struct_min_row, c] = inner_color
                
                # Bottom side: from struct_max_row to end
                if struct_max_row < rows:
                    output[struct_max_row:, c] = inner_color
        
        return output


from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects, GridObject, GridObjects
from input_library import retry
import numpy as np
import random

class Taskb7249182Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['grid_height']}x{vars['grid_width']}.",
            "Each grid contains exactly two distinct colored cells.",
            "These two colored cells are aligned either horizontally (in the same row) or vertically (in the same column).",
            "The distance between the two colored cells must follow the formula: distance = 2×E + 4 + R, where E is the extension length (≥1) on each side, 4 is the required empty space, and R is the rectangle dimension along the alignment axis."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is an exact copy of the input grid initially.",
            "The two colored cells extend toward each other in a straight line (row or column), stopping when exactly four empty cells remain between them.",
            "The extensions are equal length on both sides to maintain symmetry.",
            "At this point, the path diverges in opposite directions to form a hollow rectangle.",
            "The rectangles size and orientation depend on the alignment of the original cells. If the two cells are aligned vertically, a 5 (width) × 4 (height) rectangle is formed. If they are aligned horizontally, a 4 (width) × 5 (height) rectangle is formed.",
            "The perimeter of the rectangle is drawn such that the first half uses one input color and the second half uses the other input color."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars: dict = None, gridvars: dict = None):
        # Use taskvars if provided, otherwise use defaults
        if taskvars is None:
            taskvars = {}
        if gridvars is None:
            gridvars = {}
            
        # Get grid dimensions from taskvars
        grid_height = taskvars['grid_height']
        grid_width = taskvars['grid_width']
        color1 = taskvars.get('color1', random.randint(1, 9))
        color2 = taskvars.get('color2', random.choice([c for c in range(1, 10) if c != color1]))
        alignment = taskvars.get('alignment', random.choice(['horizontal', 'vertical']))
        
        def generate_valid_grid():
            grid = np.zeros((grid_height, grid_width), dtype=int)
            
            if alignment == 'horizontal':
                # Same row, different columns
                # Need space for 5-high rectangle (±2 from center row)
                row = random.randint(2, grid_height - 3)
                
                # For horizontal: minimum distance = 10 (2×1 + 4 + 4)
                min_distance = 10
                max_distance = grid_width - 2
                
                if max_distance < min_distance:
                    min_distance = 8
                
                # Use distances that work with the formula
                possible_distances = []
                for d in range(min_distance, max_distance + 1):
                    possible_distances.append(d)
                
                if not possible_distances:
                    distance = 10
                else:
                    distance = random.choice(possible_distances)
                
                # Place cells
                col1 = random.randint(1, max(1, grid_width - distance - 1))
                col2 = col1 + distance
                
                if col2 >= grid_width:
                    col2 = grid_width - 1
                    col1 = col2 - distance
                    col1 = max(0, col1)
                
                grid[row, col1] = color1
                grid[row, col2] = color2
                
            else:  # vertical
                # Same column, different rows
                col = random.randint(2, grid_width - 3)
                
                min_distance = 10
                max_distance = grid_height - 2
                
                if max_distance < min_distance:
                    min_distance = 8
                
                possible_distances = []
                for d in range(min_distance, max_distance + 1):
                    possible_distances.append(d)
                
                if not possible_distances:
                    distance = 10
                else:
                    distance = random.choice(possible_distances)
                
                # Place cells
                row1 = random.randint(1, max(1, grid_height - distance - 1))
                row2 = row1 + distance
                
                if row2 >= grid_height:
                    row2 = grid_height - 1
                    row1 = row2 - distance
                    row1 = max(0, row1)
                
                grid[row1, col] = color1
                grid[row2, col] = color2
            
            return grid
        
        def is_valid_grid(grid):
            colored_cells = np.where(grid != 0)
            if len(colored_cells[0]) != 2:
                return False
            
            # Check if we have enough space for the rectangle formation
            r1, r2 = colored_cells[0]
            c1, c2 = colored_cells[1]
            
            if r1 == r2:  # horizontal alignment
                row = r1
                col_min, col_max = min(c1, c2), max(c1, c2)
                distance = col_max - col_min
                
                if distance < 8:
                    return False
                
                # Check rectangle bounds (4×5 rectangle)
                center_col = (col_min + col_max) // 2
                rect_left = center_col - 1
                rect_right = center_col + 2
                rect_top = row - 2
                rect_bottom = row + 2
                
                return (rect_left >= 0 and rect_right < grid.shape[1] and 
                        rect_top >= 0 and rect_bottom < grid.shape[0])
                        
            else:  # vertical alignment
                col = c1
                row_min, row_max = min(r1, r2), max(r1, r2)
                distance = row_max - row_min
                
                if distance < 8:
                    return False
                
                # Check rectangle bounds (5×4 rectangle)
                center_row = (row_min + row_max) // 2
                rect_top = center_row - 1
                rect_bottom = center_row + 2
                rect_left = col - 2
                rect_right = col + 2
                
                return (rect_left >= 0 and rect_right < grid.shape[1] and 
                        rect_top >= 0 and rect_bottom < grid.shape[0])
            
            return False
        
        return retry(generate_valid_grid, is_valid_grid, max_attempts=200)
    
    def transform_input(self, grid):
        # Remove taskvars parameter since we don't use it
        # Work on a copy of the input grid
        output_grid = grid.copy()
        
        # Find the two colored cells
        colored_positions = np.where(output_grid != 0)
        cells = [(colored_positions[0][i], colored_positions[1][i], output_grid[colored_positions[0][i], colored_positions[1][i]]) 
                for i in range(len(colored_positions[0]))]
        
        if len(cells) != 2:
            return output_grid
        
        (r1, c1, color1), (r2, c2, color2) = cells
        
        # Determine alignment and calculate equal extensions
        if r1 == r2:  # horizontal alignment -> 4×5 rectangle
            row = r1
            col_min, col_max = min(c1, c2), max(c1, c2)
            distance = col_max - col_min
            
            # Calculate the exact center where rectangle will be placed
            center_col = (col_min + col_max) // 2
            
            # Rectangle will be 4 cells wide, centered at center_col
            rect_left = center_col - 1   # 4-wide: center-1 to center+2
            rect_right = center_col + 2
            rect_top = row - 2          # 5-high: row-2 to row+2
            rect_bottom = row + 2
            
            # Extensions should connect directly to rectangle edges
            # Calculate where extensions should end to connect with rectangle
            left_connection = rect_left  # Left extension connects to left edge of rectangle
            right_connection = rect_right  # Right extension connects to right edge of rectangle
            
            # Draw the connecting lines FROM cells TO rectangle edges
            # Left extension: from col_min to left_connection (inclusive)
            for c in range(col_min, left_connection + 1):
                if 0 <= c < output_grid.shape[1]:
                    output_grid[row, c] = color1
            
            # Right extension: from right_connection to col_max (inclusive)
            for c in range(right_connection, col_max + 1):
                if 0 <= c < output_grid.shape[1]:
                    output_grid[row, c] = color2
            
        else:  # vertical alignment -> 5×4 rectangle
            col = c1
            row_min, row_max = min(r1, r2), max(r1, r2)
            distance = row_max - row_min
            
            # Calculate the exact center where rectangle will be placed
            center_row = (row_min + row_max) // 2
            
            # Rectangle will be 4 cells high, centered at center_row
            rect_top = center_row - 1    # 4-high: center-1 to center+2
            rect_bottom = center_row + 2
            rect_left = col - 2          # 5-wide: col-2 to col+2
            rect_right = col + 2
            
            # Extensions should connect directly to rectangle edges
            top_connection = rect_top     # Top extension connects to top edge of rectangle
            bottom_connection = rect_bottom  # Bottom extension connects to bottom edge of rectangle
            
            # Draw the connecting lines FROM cells TO rectangle edges
            # Top extension: from row_min to top_connection (inclusive)
            for r in range(row_min, top_connection + 1):
                if 0 <= r < output_grid.shape[0]:
                    output_grid[r, col] = color1
            
            # Bottom extension: from bottom_connection to row_max (inclusive)
            for r in range(bottom_connection, row_max + 1):
                if 0 <= r < output_grid.shape[0]:
                    output_grid[r, col] = color2
        
        # Draw hollow rectangle with C-shaped halves
        if r1 == r2:  # horizontal alignment (4×5 rectangle) - split into left and right C shapes
            # Left edge (full height) - but skip the connection point since it's already colored
            for r in range(rect_top, rect_bottom + 1):
                if 0 <= r < output_grid.shape[0] and 0 <= rect_left < output_grid.shape[1]:
                    if r != row:  # Skip the row where extension connects
                        output_grid[r, rect_left] = color1
            
            # Right edge (full height) - but skip the connection point since it's already colored
            for r in range(rect_top, rect_bottom + 1):
                if 0 <= r < output_grid.shape[0] and 0 <= rect_right < output_grid.shape[1]:
                    if r != row:  # Skip the row where extension connects
                        output_grid[r, rect_right] = color2
            
            # Top edge (split in middle) - 4 cells wide
            mid_col = (rect_left + rect_right) // 2
            # Left half of top edge
            for c in range(rect_left + 1, mid_col + 1):
                if 0 <= rect_top < output_grid.shape[0] and 0 <= c < output_grid.shape[1]:
                    output_grid[rect_top, c] = color1
            # Right half of top edge
            for c in range(mid_col + 1, rect_right):
                if 0 <= rect_top < output_grid.shape[0] and 0 <= c < output_grid.shape[1]:
                    output_grid[rect_top, c] = color2
            
            # Bottom edge (split in middle)
            for c in range(rect_left + 1, mid_col + 1):
                if 0 <= rect_bottom < output_grid.shape[0] and 0 <= c < output_grid.shape[1]:
                    output_grid[rect_bottom, c] = color1
            for c in range(mid_col + 1, rect_right):
                if 0 <= rect_bottom < output_grid.shape[0] and 0 <= c < output_grid.shape[1]:
                    output_grid[rect_bottom, c] = color2
                    
        else:  # vertical alignment (5×4 rectangle) - split into top and bottom C shapes
            # Top edge (full width) - but skip the connection point since it's already colored
            for c in range(rect_left, rect_right + 1):
                if 0 <= rect_top < output_grid.shape[0] and 0 <= c < output_grid.shape[1]:
                    if c != col:  # Skip the column where extension connects
                        output_grid[rect_top, c] = color1
            
            # Bottom edge (full width) - but skip the connection point since it's already colored
            for c in range(rect_left, rect_right + 1):
                if 0 <= rect_bottom < output_grid.shape[0] and 0 <= c < output_grid.shape[1]:
                    if c != col:  # Skip the column where extension connects
                        output_grid[rect_bottom, c] = color2
            
            # Left edge (split in middle) - 4 cells high
            mid_row = (rect_top + rect_bottom) // 2
            for r in range(rect_top + 1, mid_row + 1):
                if 0 <= r < output_grid.shape[0] and 0 <= rect_left < output_grid.shape[1]:
                    output_grid[r, rect_left] = color1
            for r in range(mid_row + 1, rect_bottom):
                if 0 <= r < output_grid.shape[0] and 0 <= rect_left < output_grid.shape[1]:
                    output_grid[r, rect_left] = color2
            
            # Right edge (split in middle)
            for r in range(rect_top + 1, mid_row + 1):
                if 0 <= r < output_grid.shape[0] and 0 <= rect_right < output_grid.shape[1]:
                    output_grid[r, rect_right] = color1
            for r in range(mid_row + 1, rect_bottom):
                if 0 <= r < output_grid.shape[0] and 0 <= rect_right < output_grid.shape[1]:
                    output_grid[r, rect_right] = color2
        
        return output_grid
    
    def create_grids(self):
        """Create train and test grids with consistent variables."""
        # Define explicit grid variables (ensuring ALL dimensions are large enough)
        possible_dimensions = [
            (15, 18), (18, 15), (16, 20), (20, 16), (17, 19), 
            (19, 17), (14, 18), (18, 14), (16, 22), (22, 16),
            (15, 20), (20, 15), (18, 21), (21, 18), (19, 23)
        ]
        
        # Choose random dimensions for this task
        grid_height, grid_width = random.choice(possible_dimensions)
        
        taskvars = {
            'grid_height': grid_height,
            'grid_width': grid_width
        }
        
        # Generate training pairs
        train_examples = []
        num_train = random.randint(3, 5)
        
        for _ in range(num_train):
            # Randomize parameters for diversity while keeping grid size consistent
            color1 = random.randint(1, 9)
            color2 = random.choice([c for c in range(1, 10) if c != color1])
            alignment = random.choice(['horizontal', 'vertical'])
            
            # Create task variables for this specific grid
            current_taskvars = taskvars.copy()
            current_taskvars.update({
                'color1': color1,
                'color2': color2,
                'alignment': alignment
            })
            
            input_grid = self.create_input(current_taskvars, {})
            output_grid = self.transform_input(input_grid)  # Remove taskvars parameter
            
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Generate test pair with same grid dimensions
        test_color1 = random.randint(1, 9)
        test_color2 = random.choice([c for c in range(1, 10) if c != test_color1])
        test_alignment = random.choice(['horizontal', 'vertical'])
        
        test_taskvars = taskvars.copy()
        test_taskvars.update({
            'color1': test_color1,
            'color2': test_color2,
            'alignment': test_alignment
        })
        
        test_input = self.create_input(test_taskvars, {})
        test_output = self.transform_input(test_input)  # Remove taskvars parameter
        
        test_examples = [{
            'input': test_input,
            'output': test_output
        }]
        
        return taskvars, {
            'train': train_examples,
            'test': test_examples
        }

# Test the generator
if __name__ == "__main__":
    generator = Taskb7249182Generator()
    taskvars, train_test_data = generator.create_grids()
    
    # Convert to the expected format for visualization
    train_pairs = [GridPair(example['input'], example['output']) for example in train_test_data['train']]
    test_pairs = [GridPair(example['input'], example['output']) for example in train_test_data['test']]
    formatted_data = TrainTestData(train_pairs, test_pairs)
    
    generator.visualize_train_test_data(formatted_data)
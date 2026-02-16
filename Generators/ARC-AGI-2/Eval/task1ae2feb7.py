from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from typing import Dict, List, Any, Tuple
from input_library import create_object, random_cell_coloring, retry
from transformation_library import find_connected_objects, BorderBehavior

class Task1ae2feb7Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids can have different sizes.",
            "They contain a vertical {color('line_color')} line and several colored (1–9) cells, with all remaining cells being empty (0).",
            "The {color('line_color')} line is created by completely filling one column, either the fourth, fifth, or sixth, with {color('line_color')} color.",
            "The other colored cells are arranged horizontally in rows to the left of the {color('line_color')} line, following one of four possible styles.",
            "Style One: A horizontal strip made of a single color, with two or more cells horizontally connected, positioned anywhere in the specific row, before the {color('line_color')} line.",
            "Style Two: A horizontal strip made of two colors, a and b, starts from the first column with (color a) cells and stops exactly before the {color('line_color')} line with exactly one (color b) cell. The strip connects to the vertical line through the b-colored cell.",
            "Style Three: A single colored cell in the column that is exactly before the {color('line_color')} line resulting in the colored cell being horizontally connected to the {color('line_color')} line.",
            "Style Four: A horizontal strip consisting of two colors, a and b. The pattern starts from the first column with a single (color b) cell, followed by (color a) cells until the cell before the vertical line. The strip connects to the vertical line.",
            "Each grid can display up to four different styles and sometimes can have one or two styles repeating within the same grid but using a different color scheme.",
            "The horizontal strips must be separated by at least one empty (0) row between two consecutive rows having colored cells arranged horizontally."
        ]

        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid and identifying the vertical line and the colored cells which are arranged horizontally in rows, following one of the four possible styles.",
            "Style One: A horizontal strip made of a single color, with two or more cells horizontally connected, positioned anywhere in the specific row, before the vertical line.",
            "Style Two: A horizontal strip made of two colors, a and b, starts from the first column with (color a) cells and stops exactly before the vertical line with exactly one (color b) cell. The strip connects to the vertical line through the (color b) cell.",
            "Style Three: A single colored cell in the column that is exactly before the vertical line resulting in the colored cell being horizontally connected to the vertical line.",
            "Style Four: A horizontal strip consisting of two colors, a and b. The pattern starts from the first column with a single (color b) cell, followed by (color a) cells until the cell before the vertical line. The strip connects to the vertical line.",
            "For Style One, first count the number of cells used for the horizontal strip in a specific row (call it n). Starting from the first cell to the right of the vertical cell in the same row, fill it with the strip color and leave n-1 cells empty (0) to its right. Repeat the pattern until the end of the row.",
            "For Styles Two and Three, fill the row starting from the first empty cell to the right of the vertical cell in the same row. Use the color that appears immediately before the vertical cell in the same row, and continue filling until the end of the row.",
            "For Style Four, create a checkerboard pattern where (color a) and (color b) cells alternate in row, starting from the first empty (0) cell to the right of the vertical cell and continuing until the end of the row.",
            "In case the horizontal strips appear to the right of the vertical strips instead of left, then the identification of the style must be done according to the changed position of the strips, all the rules still apply. However, the results will be displayed on the left side of the vertical line.",
            "In case some styles, like Two and Four, have a shorter length and do not start from the first column, the rules still apply.",
            "The vertical line remains unchanged from input to output grid."
        ]

        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Set up task variables
        taskvars = {
            'line_color': random.randint(1, 9)
        }

        # Create train grids
        train_grids = []
        
        # First train grid - style five, three and two
        styles_grid1 = [(5, self._get_random_color(taskvars['line_color'])), 
                        (3, self._get_random_color(taskvars['line_color'])), 
                        (2, self._get_random_color(taskvars['line_color']))]
        height1, width1 = random.randint(15, 30), random.randint(15, 30)
        line_col1 = random.choice([4, 5, 6])
        gridvars1 = {'styles': styles_grid1, 'line_col': line_col1, 'strips_side': 'left', 'height': height1, 'width': width1}
        input_grid1 = self.create_input(taskvars, gridvars1)
        output_grid1 = self.transform_input(input_grid1, taskvars)
        train_grids.append({'input': input_grid1, 'output': output_grid1})
        
        # Second train grid - style one, three, two and four
        styles_grid2 = [(1, self._get_random_color(taskvars['line_color'])), 
                        (3, self._get_random_color(taskvars['line_color'])), 
                        (2, self._get_random_color(taskvars['line_color'])),
                        (4, (self._get_random_color(taskvars['line_color']), self._get_random_color(taskvars['line_color'])))]
        height2, width2 = random.randint(15, 30), random.randint(15, 30)
        line_col2 = random.choice([4, 5, 6])
        gridvars2 = {'styles': styles_grid2, 'line_col': line_col2, 'strips_side': 'left', 'height': height2, 'width': width2}
        input_grid2 = self.create_input(taskvars, gridvars2)
        output_grid2 = self.transform_input(input_grid2, taskvars)
        train_grids.append({'input': input_grid2, 'output': output_grid2})
        
        # Third train grid - style one, six, two
        styles_grid3 = [(1, self._get_random_color(taskvars['line_color'])), 
                        (6, (self._get_random_color(taskvars['line_color']), self._get_random_color(taskvars['line_color']))), 
                        (2, self._get_random_color(taskvars['line_color']))]
        height3, width3 = random.randint(15, 30), random.randint(15, 30)
        line_col3 = random.choice([4, 5, 6])
        gridvars3 = {'styles': styles_grid3, 'line_col': line_col3, 'strips_side': 'left', 'height': height3, 'width': width3}
        input_grid3 = self.create_input(taskvars, gridvars3)
        output_grid3 = self.transform_input(input_grid3, taskvars)
        train_grids.append({'input': input_grid3, 'output': output_grid3})
        
        # Test grids - change the line color
        test_line_color = self._get_random_color(taskvars['line_color'])
        
        # First test grid - vertical line in the fourth or fifth last column, with styles on the right
        test_styles1 = [(random.choice([1, 2, 3, 4, 5, 6]), 
                        self._get_random_color(test_line_color) if random.choice([1, 2, 3, 5]) else 
                        (self._get_random_color(test_line_color), self._get_random_color(test_line_color))) 
                        for _ in range(random.randint(2, 4))]
        height_test1, width_test1 = random.randint(15, 30), random.randint(15, 30)
        # Ensure vertical line is at the fourth or fifth-last column
        test_line_col1 = width_test1 - random.choice([4, 5])
        test_gridvars1 = {'styles': test_styles1, 'line_col': test_line_col1, 'strips_side': 'right', 
                         'line_color': test_line_color, 'height': height_test1, 'width': width_test1, 'is_test': True}
        test_input_grid1 = self.create_input(taskvars, test_gridvars1)
        test_output_grid1 = self.transform_input(test_input_grid1, taskvars)
        
        # Second test grid - vertical line in 4th or 5th column, styles on the left
        test_styles2 = []
        # Ensure style 2 with [a,a,b] pattern is included
        style2_colors = (self._get_random_color(test_line_color), self._get_random_color(test_line_color))
        test_styles2.append((2, style2_colors))  # Add style 2 with two colors [a,a,b]
        
        # Add other random styles
        for _ in range(random.randint(1, 3)):
            style = random.choice([1, 3, 4, 5, 6])  # Avoid adding style 2 again
            color = self._get_random_color(test_line_color) if style in [1, 3, 5] else (
                self._get_random_color(test_line_color), self._get_random_color(test_line_color))
            test_styles2.append((style, color))
            
        height_test2, width_test2 = random.randint(15, 30), random.randint(15, 30)
        # Place vertical line in the fourth or fifth column
        test_line_col2 = random.choice([3, 4])  # 0-indexed, so 3,4 = 4th,5th columns
        test_gridvars2 = {'styles': test_styles2, 'line_col': test_line_col2, 'strips_side': 'left', 
                         'line_color': test_line_color, 'height': height_test2, 'width': width_test2, 'is_test': True}
        test_input_grid2 = self.create_input(taskvars, test_gridvars2)
        test_output_grid2 = self.transform_input(test_input_grid2, taskvars)
        
        # Additional test grid for variety
        test_styles3 = [(random.choice([1, 2, 3, 4, 5, 6]), 
                        self._get_random_color(test_line_color) if random.choice([1, 2, 3, 5]) else 
                        (self._get_random_color(test_line_color), self._get_random_color(test_line_color))) 
                        for _ in range(random.randint(3, 5))]
        height_test3, width_test3 = random.randint(15, 30), random.randint(15, 30)
        
        # Randomly choose between the patterns from test1 and test2
        if random.choice([True, False]):
            # Like test1: vertical line in 4th/5th last column, strips on right
            test_line_col3 = width_test3 - random.choice([4, 5])
            strips_side3 = 'right'
        else:
            # Like test2: vertical line in 4th/5th column, strips on left
            test_line_col3 = random.choice([3, 4])  # 0-indexed, so 3,4 = 4th,5th columns
            strips_side3 = 'left'
            
            # If strips are on left, ensure we include style 2 with [a,a,b] pattern
            if strips_side3 == 'left':
                # Reset styles list
                test_styles3 = []
                # Add style 2 with [a,a,b] pattern
                style2_colors = (self._get_random_color(test_line_color), self._get_random_color(test_line_color))
                test_styles3.append((2, style2_colors))
                # Add other random styles
                for _ in range(random.randint(2, 4)):
                    style = random.choice([1, 3, 4, 5, 6])
                    color = self._get_random_color(test_line_color) if style in [1, 3, 5] else (
                        self._get_random_color(test_line_color), self._get_random_color(test_line_color))
                    test_styles3.append((style, color))
            
        test_gridvars3 = {'styles': test_styles3, 'line_col': test_line_col3, 
                        'strips_side': strips_side3, 
                        'line_color': test_line_color, 'height': height_test3, 'width': width_test3, 'is_test': True}
        test_input_grid3 = self.create_input(taskvars, test_gridvars3)
        test_output_grid3 = self.transform_input(test_input_grid3, taskvars)
        
        return taskvars, {
            'train': train_grids,
            'test': [
                {'input': test_input_grid1, 'output': test_output_grid1},
                {'input': test_input_grid2, 'output': test_output_grid2},
                {'input': test_input_grid3, 'output': test_output_grid3}
            ]
        }
    
    def _get_random_color(self, avoid_color=None):
        """Get a random color between 1 and 9, avoiding a specific color if provided."""
        colors = list(range(1, 10))
        if avoid_color:
            if avoid_color in colors:
                colors.remove(avoid_color)
        return random.choice(colors)

    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        """Create an input grid with a vertical line and horizontal strips."""
        # Grid dimensions
        height = gridvars.get('height', random.randint(15, 30))
        width = gridvars.get('width', random.randint(15, 30))
        
        # Initialize the grid with zeros
        grid = np.zeros((height, width), dtype=int)
        
        # Extract variables
        line_color = gridvars.get('line_color', taskvars['line_color'])
        line_col = min(gridvars['line_col'], width - 2)  # Ensure line_col is within grid bounds
        strips_side = gridvars['strips_side']
        styles = gridvars['styles']
        is_test = gridvars.get('is_test', False)
        
        # Create vertical line
        if is_test:
            # For test examples: vertical line from first row to second-last row
            for row in range(height - 1):
                grid[row, line_col] = line_color
        else:
            # For train examples: vertical line from first row to last row
            for row in range(height):
                grid[row, line_col] = line_color
        
        # Create horizontal strips according to styles
        available_rows = list(range(1, height - 2))  # Avoid first and last rows
        random.shuffle(available_rows)
        
        for i, (style, color) in enumerate(styles):
            if i >= len(available_rows):
                break  # Ensure we don't run out of rows
                
            row = available_rows[i]
            
            # Determine strip position based on side
            if strips_side == 'left':
                self._create_left_strip(grid, row, line_col, style, color)
            else:  # strips_side == 'right'
                self._create_right_strip(grid, row, line_col, style, color)
            
            # Ensure space between strips
            if row + 1 in available_rows:
                available_rows.remove(row + 1)
            if row - 1 in available_rows:
                available_rows.remove(row - 1)
        
        return grid
    
    def _create_left_strip(self, grid, row, line_col, style, color):
        """Create a horizontal strip to the left of the vertical line."""
        if style == 1:  # Single color strip with gap
            strip_length = random.randint(1, max(1, line_col - 2))
            for col in range(0, min(strip_length, line_col - 1)):
                grid[row, col] = color
        
        elif style == 2:  # Two colors strip [a,a,...,b] ending with b connected to vertical line
            if isinstance(color, tuple) and len(color) == 2:
                color_a, color_b = color
                # Create pattern [a,a,...,b] where b is just before the vertical line
                for col in range(1, line_col - 1):
                    grid[row, col] = color_a
                if line_col > 1:
                    grid[row, line_col - 1] = color_b  # b color right before vertical line
            else:
                # Fallback if color is not a tuple
                for col in range(1, line_col):
                    grid[row, col] = color
        
        elif style == 3:  # Single color strip from 1st column, connected
            for col in range(0, line_col):
                grid[row, col] = color
        
        elif style == 4:  # Two colors strip [a,a,...,b] - starts from first column, ends with b
            if isinstance(color, tuple) and len(color) == 2:
                color_a, color_b = color
                # Start from first column with color_a cells and end with a single color_b cell
                for col in range(0, line_col - 1):
                    grid[row, col] = color_a
                if line_col > 0:
                    grid[row, line_col - 1] = color_b  # Exactly one color_b cell at the end
            else:
                # Fallback to style 1 if color is not a tuple of length 2
                strip_length = random.randint(1, max(1, line_col - 2))
                for col in range(0, min(strip_length, line_col - 1)):
                    grid[row, col] = color
        
        elif style == 5:  # Single cell exactly to the left of vertical line
            if line_col > 0:
                # Place the single cell exactly before the vertical line
                grid[row, line_col - 1] = color
        
        elif style == 6:  # Two colors strip [b,a,...,a] - starts from first column with b, then a cells
            if isinstance(color, tuple) and len(color) == 2:
                color_a, color_b = color
                # First cell is color_b, rest are color_a
                grid[row, 0] = color_b  # First cell is color_b
                for col in range(1, line_col):
                    grid[row, col] = color_a  # Rest are color_a
            else:
                # Fallback to style 1 if color is not a tuple of length 2
                strip_length = random.randint(1, max(1, line_col - 2))
                for col in range(0, min(strip_length, line_col - 1)):
                    grid[row, col] = color
    
    def _create_right_strip(self, grid, row, line_col, style, color):
        """Create a horizontal strip to the right of the vertical line."""
        width = grid.shape[1]
        
        if style == 1:  # Single color strip with gap
            strip_length = random.randint(1, max(1, width - line_col - 2))
            for col in range(line_col + 2, min(width, line_col + 2 + strip_length)):
                grid[row, col] = color
        
        elif style == 2:  # Single color strip from next column, connected
            for col in range(line_col + 1, min(width, line_col + 1 + random.randint(1, max(1, width - line_col - 1)))):
                grid[row, col] = color
        
        elif style == 3:  # Single color strip from next column to end, connected
            for col in range(line_col + 1, width):
                grid[row, col] = color
        
        elif style == 4:  # Two colors strip [b,a,a,...] - starts with b next to vertical line
            if isinstance(color, tuple) and len(color) == 2:
                color_a, color_b = color
                if line_col + 1 < width:
                    grid[row, line_col + 1] = color_b  # First cell is color_b (connected to vertical line)
                    for col in range(line_col + 2, width):
                        grid[row, col] = color_a  # Rest are color_a
            else:
                # Fallback to style 1 if color is not a tuple of length 2
                strip_length = random.randint(1, max(1, width - line_col - 2))
                for col in range(line_col + 2, min(width, line_col + 2 + strip_length)):
                    grid[row, col] = color
        
        elif style == 5:  # Single cell exactly to the right of vertical line
            if line_col + 1 < width:
                # Place the single cell exactly to the right of the vertical line
                grid[row, line_col + 1] = color
        
        elif style == 6:  # Two colors strip [b,a,...,a] - starts with b next to vertical line
            if isinstance(color, tuple) and len(color) == 2:
                color_a, color_b = color
                if line_col + 1 < width:
                    grid[row, line_col + 1] = color_b  # First cell is color_b
                    for col in range(line_col + 2, width):
                        grid[row, col] = color_a  # Rest are color_a
            else:
                # Fallback to style 1 if color is not a tuple of length 2
                strip_length = random.randint(1, max(1, width - line_col - 2))
                for col in range(line_col + 2, min(width, line_col + 2 + strip_length)):
                    grid[row, col] = color

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        """Transform the input grid according to the specified rules."""
        # Create a copy of the input grid
        output = grid.copy()
        height, width = grid.shape
        
        # Find the vertical line
        line_color = taskvars['line_color']
        line_col = None
        
        # If we cannot find the line with line_color, find it with any color in the first column
        line_found = False
        for col in range(width):
            if np.sum(grid[:, col] > 0) > height / 3:  # If more than 1/3 of the cells in a column have color
                line_col = col
                line_found = True
                break
        
        if not line_found:
            return output  # Return unchanged if no vertical line found
        
        # Process each row to identify styles and apply transformations
        for row in range(height):
            if row >= grid.shape[0] or line_col >= grid.shape[1] or grid[row, line_col] == 0:
                continue  # Skip if no vertical line in this row or if out of bounds
            
            # Check if there are colored cells to the left of the line
            left_strip = grid[row, :line_col]
            left_has_color = np.any(left_strip > 0)
            
            # Check if there are colored cells to the right of the line
            right_strip = grid[row, line_col+1:] if line_col + 1 < width else np.array([])
            right_has_color = len(right_strip) > 0 and np.any(right_strip > 0)
            
            # Apply transformations based on the side with strips
            if left_has_color:
                self._transform_left_strip(grid, output, row, line_col, width)
            elif right_has_color:
                self._transform_right_strip(grid, output, row, line_col)
        
        return output
    
    def _transform_left_strip(self, grid, output, row, line_col, width):
        """Apply transformation for strips on the left of the vertical line."""
        left_strip = grid[row, :line_col]
        
        # Detect style
        style = self._detect_style_left(grid, row, line_col)
        
        if style in [1, 2, 3]:
            # Count cells in the horizontal strip
            non_zero_cells = left_strip[left_strip > 0]
            if len(non_zero_cells) > 0:
                strip_color = non_zero_cells[0]  # Use the first color in the strip
                strip_length = np.sum(left_strip == strip_color)
                
                # Apply pattern to the right
                for col in range(line_col + 1, width):
                    if (col - line_col - 1) % max(1, strip_length) == 0:
                        output[row, col] = strip_color
                    else:
                        output[row, col] = 0
        
        elif style in [4, 5]:
            # Get the color immediately before the vertical line
            if line_col > 0 and grid[row, line_col - 1] > 0:
                last_color = grid[row, line_col - 1]
                
                # Fill all cells to the right with this color
                for col in range(line_col + 1, width):
                    output[row, col] = last_color
        
        elif style == 6:
            # Find the two colors in the pattern
            unique_colors = np.unique(left_strip[left_strip > 0])
            if len(unique_colors) >= 2:
                color_a, color_b = unique_colors[1], unique_colors[0]  # color_a is the majority color, color_b is the first cell
                
                # Create checkerboard pattern to the right
                col_idx = 0
                for col in range(line_col + 1, width):
                    output[row, col] = color_a if col_idx % 2 == 0 else color_b
                    col_idx += 1
    
    def _transform_right_strip(self, grid, output, row, line_col):
        """Apply transformation for strips on the right of the vertical line."""
        width = grid.shape[1]
        right_strip = grid[row, line_col+1:] if line_col + 1 < width else np.array([])
        
        # Detect style
        style = self._detect_style_right(grid, row, line_col)
        
        if style in [1, 2, 3]:
            # Count cells in the horizontal strip
            non_zero_cells = right_strip[right_strip > 0]
            if len(non_zero_cells) > 0:
                strip_color = non_zero_cells[0]  # Use the first color in the strip
                strip_length = np.sum(right_strip == strip_color)
                
                # Apply pattern to the left
                for col in range(line_col - 1, -1, -1):
                    if (line_col - col - 1) % max(1, strip_length) == 0:
                        output[row, col] = strip_color
                    else:
                        output[row, col] = 0
        
        elif style in [4, 5]:
            # Get the color immediately after the vertical line
            if line_col + 1 < width and grid[row, line_col + 1] > 0:
                first_color = grid[row, line_col + 1]
                
                # Fill all cells to the left with this color
                for col in range(0, line_col):
                    output[row, col] = first_color
        
        elif style == 6:
            # Find the two colors in the pattern
            unique_colors = np.unique(right_strip[right_strip > 0])
            if len(unique_colors) >= 2:
                color_a, color_b = unique_colors[1], unique_colors[0]  # color_a is the majority color, color_b is the first cell
                
                # Create checkerboard pattern to the left
                col_idx = 0
                for col in range(0, line_col):
                    output[row, col] = color_a if col_idx % 2 == 0 else color_b
                    col_idx += 1
    
    def _detect_style_left(self, grid, row, line_col):
        """Detect the style of a horizontal strip on the left of the vertical line."""
        if line_col <= 0:
            return 1  # Default to style 1 if there's no space to the left
            
        left_strip = grid[row, :line_col]
        
        # Style 5: Single colored cell exactly to the left of vertical line
        if np.sum(left_strip > 0) == 1 and left_strip[-1] > 0:
            return 5
        
        # Style 1: Single color strip with gap
        if np.sum(left_strip > 0) > 0 and (left_strip.size == 0 or left_strip[-1] == 0):
            return 1
        
        # Style 2: Single color strip from 2nd column, connected
        if left_strip.size > 1 and left_strip[0] == 0 and np.all(left_strip[1:] > 0) and np.unique(left_strip[left_strip > 0]).size == 1:
            return 2
        
        # Style 3: Single color strip from 1st column, connected
        if left_strip.size > 0 and left_strip[0] > 0 and np.all(left_strip > 0) and np.unique(left_strip).size == 1:
            return 3
        
        # Style 4: Two colors strip [a,a,...,b]
        unique_colors = np.unique(left_strip[left_strip > 0])
        if len(unique_colors) == 2 and left_strip.size > 0 and left_strip[-1] != 0 and left_strip[-1] != left_strip[0]:
            # Check if all but the last are the same color
            first_color = left_strip[np.where(left_strip > 0)[0][0]]
            if np.all(left_strip[:-1][left_strip[:-1] > 0] == first_color) and left_strip[-1] != first_color:
                return 4
        
        # Style 6: Two colors strip [b,a,...,a] starting from first column
        if len(unique_colors) == 2 and left_strip[0] > 0:
            first_color = left_strip[0]
            # Check if all cells after the first are the same different color
            other_color = None
            for col in range(1, len(left_strip)):
                if left_strip[col] > 0 and left_strip[col] != first_color:
                    other_color = left_strip[col]
                    break
            
            if other_color and np.all(left_strip[1:][left_strip[1:] > 0] == other_color):
                return 6
        
        # Default to style 1 if can't determine
        return 1
    
    def _detect_style_right(self, grid, row, line_col):
        """Detect the style of a horizontal strip on the right of the vertical line."""
        width = grid.shape[1]
        if line_col + 1 >= width:
            return 1  # Default to style 1 if there's no space to the right
            
        right_strip = grid[row, line_col+1:]
        
        # Style 5: Single colored cell exactly to the right of vertical line
        if np.sum(right_strip > 0) == 1 and right_strip[0] > 0:
            return 5
        
        # Style 1: Single color strip with gap
        if np.sum(right_strip > 0) > 0 and right_strip[0] == 0:
            return 1
        
        # Style 2: Single color strip starts immediately after line but doesn't reach the end
        if right_strip[0] > 0 and (right_strip.size == 0 or right_strip[-1] == 0) and np.unique(right_strip[right_strip > 0]).size == 1:
            return 2
        
        # Style 3: Single color strip fills the whole right side
        if np.all(right_strip > 0) and np.unique(right_strip).size == 1:
            return 3
        
        # Style 4: Two colors strip [b,a,a,...] - first cell b, rest a
        unique_colors = np.unique(right_strip[right_strip > 0])
        if len(unique_colors) == 2 and right_strip.size > 0 and right_strip[0] > 0:
            # Check if first color is different from others
            first_color = right_strip[0]
            if np.any(right_strip[1:] > 0):
                other_color = right_strip[np.where(right_strip[1:] > 0)[0][0] + 1]
                if np.all(right_strip[1:][right_strip[1:] > 0] == other_color) and first_color != other_color:
                    return 4
        
        # Style 6: Two colors strip [b,a,...,a] starting immediately after vertical line
        if len(unique_colors) == 2 and right_strip[0] > 0:
            first_color = right_strip[0]
            if np.any(right_strip[1:] > 0):
                other_color = right_strip[np.where(right_strip[1:] > 0)[0][0] + 1]
                if np.all(right_strip[1:][right_strip[1:] > 0] == other_color) and first_color != other_color:
                    return 6
        
        # Default to style 1 if can't determine
        return 1
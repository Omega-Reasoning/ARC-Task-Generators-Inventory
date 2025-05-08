# from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
# import numpy as np
# import random
# from input_library import create_object, Contiguity, random_cell_coloring
# from transformation_library import find_connected_objects

# class LessOccurringColorsTask(ARCTaskGenerator):
#     def __init__(self):
#         input_reasoning_chain = [
#             "Input grids can have different sizes.",
#             "They contain three colored cells (between 1-9){{color(\"object_color1\")}}color, {{color(\"object_color2\")}}color and {{color(\"object_color3\")}}color.",
#             "The{{color(\"object_color1\")}}color cells are always in excess.",
#             "The {{color(\"object_color2\")}}color cells and {{color(\"object_color3\")}}color cells are in smaller quantities compared to {{color(\"object_color1\")}}color cells."
#         ]
        
#         transformation_reasoning_chain = [
#             "The output grid is different and is determined by the less occurring color cells.",
#             "So, the colors which were less occurring on the input grid, forms a new grid of size comprising only the less occurring two colors which are {{color(\"object_color2\")}}color and {{color(\"object_color3\")}}color.",
#             "The output grid will be a smaller grid that contains only the cells of the two less occurring colors.",
#             "The least occurring cells can form a pattern to stay intact in the output grid, so the pattern would look concise and not scattered around."
#         ]
        
#         taskvars_definitions = {
#             "object_color1": {"type": "integer", "range": [1, 9]},
#             "object_color2": {"type": "integer", "range": [1, 9]},
#             "object_color3": {"type": "integer", "range": [1, 9]}
#         }
#         super().__init__(input_reasoning_chain, transformation_reasoning_chain)
#         self.taskvars_definitions = taskvars_definitions
#         self.taskvars = {}  # Initialize empty taskvars dictionary
    
#     def create_input(self):
#         # Random grid size between 5x5 and 20x20
#         rows = random.randint(7, 20)
#         cols = random.randint(7, 20)
        
#         # Select three different colors between 1 and 9
#         available_colors = list(range(1, 10))
#         colors = random.sample(available_colors, 3)
        
#         excess_color = colors[0]  # This will be the dominant color
#         minor_color1 = colors[1]  # Less frequent color 1
#         minor_color2 = colors[2]  # Less frequent color 2
        
#         # Create the grid filled with the excess color
#         grid = np.full((rows, cols), excess_color, dtype=int)
        
#         # Create concise, minimalistic patterns for minor colors
#         # We'll use simple geometric shapes instead of random scattered points
        
#         # Choose pattern type for both minor colors
#         pattern_type = random.choice(["line", "rectangle", "corner", "diagonal", "circle"])
        
#         # Pattern size - keeping it small for minimalism
#         pattern_size = random.randint(2, min(5, rows-2, cols-2))
        
#         # Random position to place pattern (with padding to ensure it fits)
#         row_pos = random.randint(1, rows - pattern_size - 1)
#         col_pos = random.randint(1, cols - pattern_size - 1)
        
#         # Create minor color 1 pattern
#         if pattern_type == "line":
#             # Horizontal or vertical line
#             is_horizontal = random.choice([True, False])
#             if is_horizontal:
#                 for c in range(col_pos, col_pos + pattern_size):
#                     grid[row_pos, c] = minor_color1
#             else:
#                 for r in range(row_pos, row_pos + pattern_size):
#                     grid[r, col_pos] = minor_color1
                    
#         elif pattern_type == "rectangle":
#             # Simple rectangle or square
#             rect_height = random.randint(2, pattern_size)
#             rect_width = random.randint(2, pattern_size)
            
#             # Either outline only or filled
#             filled = random.choice([True, False])
            
#             for r in range(row_pos, row_pos + rect_height):
#                 for c in range(col_pos, col_pos + rect_width):
#                     if filled or r == row_pos or r == row_pos + rect_height - 1 or c == col_pos or c == col_pos + rect_width - 1:
#                         grid[r, c] = minor_color1
                        
#         elif pattern_type == "corner":
#             # L-shape corner
#             for r in range(row_pos, row_pos + pattern_size):
#                 grid[r, col_pos] = minor_color1
#             for c in range(col_pos, col_pos + pattern_size):
#                 grid[row_pos, c] = minor_color1
                
#         elif pattern_type == "diagonal":
#             # Diagonal line
#             for i in range(pattern_size):
#                 grid[row_pos + i, col_pos + i] = minor_color1
                
#         elif pattern_type == "circle":
#             # Approximate circle
#             radius = pattern_size // 2
#             center_r = row_pos + radius
#             center_c = col_pos + radius
            
#             for r in range(center_r - radius, center_r + radius + 1):
#                 for c in range(center_c - radius, center_c + radius + 1):
#                     # Avoid index out of bounds
#                     if 0 <= r < rows and 0 <= c < cols:
#                         # Simple circle approximation
#                         if ((r - center_r)**2 + (c - center_c)**2) <= radius**2:
#                             grid[r, c] = minor_color1
        
#         # Create minor color 2 pattern
#         # Choose a different position for the second pattern
#         row_pos2 = random.randint(1, rows - pattern_size - 1)
#         col_pos2 = random.randint(1, cols - pattern_size - 1)
        
#         # Make sure patterns don't overlap too much
#         while abs(row_pos - row_pos2) < 2 and abs(col_pos - col_pos2) < 2:
#             row_pos2 = random.randint(1, rows - pattern_size - 1)
#             col_pos2 = random.randint(1, cols - pattern_size - 1)
        
#         # Choose a different pattern type for minor color 2
#         pattern_type2 = random.choice(["line", "rectangle", "corner", "diagonal", "circle"])
#         while pattern_type2 == pattern_type:
#             pattern_type2 = random.choice(["line", "rectangle", "corner", "diagonal", "circle"])
        
#         # Apply second pattern
#         if pattern_type2 == "line":
#             is_horizontal = random.choice([True, False])
#             if is_horizontal:
#                 for c in range(col_pos2, col_pos2 + pattern_size):
#                     grid[row_pos2, c] = minor_color2
#             else:
#                 for r in range(row_pos2, row_pos2 + pattern_size):
#                     grid[r, col_pos2] = minor_color2
                    
#         elif pattern_type2 == "rectangle":
#             rect_height = random.randint(2, pattern_size)
#             rect_width = random.randint(2, pattern_size)
#             filled = random.choice([True, False])
            
#             for r in range(row_pos2, row_pos2 + rect_height):
#                 for c in range(col_pos2, col_pos2 + rect_width):
#                     if filled or r == row_pos2 or r == row_pos2 + rect_height - 1 or c == col_pos2 or c == col_pos2 + rect_width - 1:
#                         grid[r, c] = minor_color2
                        
#         elif pattern_type2 == "corner":
#             for r in range(row_pos2, row_pos2 + pattern_size):
#                 grid[r, col_pos2] = minor_color2
#             for c in range(col_pos2, col_pos2 + pattern_size):
#                 grid[row_pos2, c] = minor_color2
                
#         elif pattern_type2 == "diagonal":
#             for i in range(pattern_size):
#                 grid[row_pos2 + i, col_pos2 + i] = minor_color2
                
#         elif pattern_type2 == "circle":
#             radius = pattern_size // 2
#             center_r = row_pos2 + radius
#             center_c = col_pos2 + radius
            
#             for r in range(center_r - radius, center_r + radius + 1):
#                 for c in range(center_c - radius, center_c + radius + 1):
#                     if 0 <= r < rows and 0 <= c < cols:
#                         if ((r - center_r)**2 + (c - center_c)**2) <= radius**2:
#                             grid[r, c] = minor_color2
                
#         return grid
    
#     def transform_input(self, input_grid):
#         unique_colors, counts = np.unique(input_grid, return_counts=True)
        
#         # Find the dominant color (most frequent)
#         excess_color = unique_colors[np.argmax(counts)]
        
#         # The other two colors are the less frequent ones
#         minor_colors = [color for color in unique_colors if color != excess_color]
        
#         # Create a mask for the positions of the minor colors
#         mask1 = input_grid == minor_colors[0]
#         mask2 = input_grid == minor_colors[1]
        
#         # Find the bounding box containing both minor colors
#         rows_with_minor_colors = np.where(mask1 | mask2)[0]
#         cols_with_minor_colors = np.where(mask1 | mask2)[1]
        
#         if len(rows_with_minor_colors) == 0 or len(cols_with_minor_colors) == 0:
#             # Edge case: if no minor colors found, return empty grid
#             return np.zeros((1, 1), dtype=int)
        
#         min_row = rows_with_minor_colors.min()
#         max_row = rows_with_minor_colors.max()
#         min_col = cols_with_minor_colors.min()
#         max_col = cols_with_minor_colors.max()
        
#         # Create a new grid with the dimensions of the bounding box
#         new_rows = max_row - min_row + 1
#         new_cols = max_col - min_col + 1
#         output_grid = np.zeros((new_rows, new_cols), dtype=int)
        
#         # Copy only the minor colors to the output grid
#         for i, r in enumerate(range(min_row, max_row + 1)):
#             for j, c in enumerate(range(min_col, max_col + 1)):
#                 if input_grid[r, c] in minor_colors:
#                     output_grid[i, j] = input_grid[r, c]
        
#         return output_grid
    
#     def create_grids(self):
#         # Generate random number of training pairs
#         num_train_pairs = random.randint(3, 5)
        
#         train_pairs = []
#         for _ in range(num_train_pairs):
#             input_grid = self.create_input()
#             output_grid = self.transform_input(input_grid)
#             train_pairs.append(GridPair(input=input_grid, output=output_grid))
        
#         # Generate test pair
#         test_input = self.create_input()
#         test_output = self.transform_input(test_input)
#         test_pairs = [GridPair(input=test_input, output=test_output)]
        
#         return self.taskvars_definitions, TrainTestData(train=train_pairs, test=test_pairs)  # Return taskvars_definitions instead of taskvars
    

from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from input_library import create_object, Contiguity, random_cell_coloring
from transformation_library import find_connected_objects

class Taska740d043Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids can have different sizes.",
            "They contain three colored cells (between 1-9){{color(\"object_color1\")}}color, {{color(\"object_color2\")}}color and {{color(\"object_color3\")}}color.",
            "The{{color(\"object_color1\")}}color cells are always in excess.",
            "The {{color(\"object_color2\")}}color cells and {{color(\"object_color3\")}}color cells are in smaller quantities compared to {{color(\"object_color1\")}}color cells."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is different and is determined by the less occurring color cells.",
            "So, the colors which were less occurring on the input grid, forms a new grid of size comprising only the less occurring two colors which are {{color(\"object_color2\")}}color and {{color(\"object_color3\")}}color.",
            "The output grid will be a smaller grid that contains only the cells of the two less occurring colors.",
            "The least occurring cells can form a pattern to stay intact in the output grid, so the pattern would look concise and not scattered around."
        ]
        
        taskvars_definitions = {
            "object_color1": {"type": "integer", "range": [1, 9]},
            "object_color2": {"type": "integer", "range": [1, 9]},
            "object_color3": {"type": "integer", "range": [1, 9]}
        }
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
        self.taskvars_definitions = taskvars_definitions
        self.taskvars = {}  # Initialize empty taskvars dictionary
    
    def create_input(self):
        # Random grid size between 5x5 and 20x20
        rows = random.randint(7, 10)  # Keeping sizes more constrained like in the example
        cols = random.randint(7, 10)
        
        # Select three different colors between 1 and 9
        available_colors = list(range(1, 10))
        colors = random.sample(available_colors, 3)
        
        excess_color = colors[0]  # This will be the dominant color
        minor_color1 = colors[1]  # Less frequent color 1
        minor_color2 = colors[2]  # Less frequent color 2
        
        # Create the grid filled with the excess color
        grid = np.full((rows, cols), excess_color, dtype=int)
        
        # Based on the image example, we should create extremely minimal patterns
        # Only 1-3 cells for each minor color, often in very simple arrangements
        
        # First, decide centers for both patterns, keeping them apart
        center1_row = random.randint(1, rows-2)
        center1_col = random.randint(1, cols-2)
        
        # Make sure the second center is separated from the first
        while True:
            center2_row = random.randint(1, rows-2)
            center2_col = random.randint(1, cols-2)
            if abs(center1_row - center2_row) + abs(center1_col - center2_col) > 2:
                break
        
        # Now create ultraminimal patterns for minor color 1
        # Looking at the example, often just 1-2 adjacent cells
        pattern_type = random.choice(["single", "two_adjacent", "small_L"])
        
        if pattern_type == "single":
            # Just a single cell
            grid[center1_row, center1_col] = minor_color1
            
        elif pattern_type == "two_adjacent":
            # Two adjacent cells in a line
            grid[center1_row, center1_col] = minor_color1
            
            # Choose a direction (horizontal or vertical)
            direction = random.choice([(0, 1), (1, 0)])
            r2 = center1_row + direction[0]
            c2 = center1_col + direction[1]
            
            if 0 <= r2 < rows and 0 <= c2 < cols:
                grid[r2, c2] = minor_color1
                
        elif pattern_type == "small_L":
            # Small L-shape (exactly like in one example)
            grid[center1_row, center1_col] = minor_color1
            
            # Add two more cells to form an L (if possible)
            if center1_row + 1 < rows:
                grid[center1_row + 1, center1_col] = minor_color1
            if center1_col + 1 < cols:
                grid[center1_row, center1_col + 1] = minor_color1
        
        # Now create pattern for minor color 2
        # Use a different pattern type for variety
        pattern_type2 = random.choice(["single", "two_adjacent", "small_L"])
        
        if pattern_type2 == "single":
            grid[center2_row, center2_col] = minor_color2
            
        elif pattern_type2 == "two_adjacent":
            grid[center2_row, center2_col] = minor_color2
            
            direction = random.choice([(0, 1), (1, 0)])
            r2 = center2_row + direction[0]
            c2 = center2_col + direction[1]
            
            if 0 <= r2 < rows and 0 <= c2 < cols:
                grid[r2, c2] = minor_color2
                
        elif pattern_type2 == "small_L":
            grid[center2_row, center2_col] = minor_color2
            
            if center2_row + 1 < rows:
                grid[center2_row + 1, center2_col] = minor_color2
            if center2_col + 1 < cols:
                grid[center2_row, center2_col + 1] = minor_color2
                
        return grid
    
    def transform_input(self, input_grid):
        unique_colors, counts = np.unique(input_grid, return_counts=True)
        
        # Find the dominant color (most frequent)
        excess_color = unique_colors[np.argmax(counts)]
        
        # The other two colors are the less frequent ones
        minor_colors = [color for color in unique_colors if color != excess_color]
        
        # Create a mask for the positions of the minor colors
        mask1 = input_grid == minor_colors[0]
        mask2 = input_grid == minor_colors[1]
        
        # Find the bounding box containing both minor colors
        rows_with_minor_colors = np.where(mask1 | mask2)[0]
        cols_with_minor_colors = np.where(mask1 | mask2)[1]
        
        if len(rows_with_minor_colors) == 0 or len(cols_with_minor_colors) == 0:
            # Edge case: if no minor colors found, return empty grid
            return np.zeros((1, 1), dtype=int)
        
        min_row = rows_with_minor_colors.min()
        max_row = rows_with_minor_colors.max()
        min_col = cols_with_minor_colors.min()
        max_col = cols_with_minor_colors.max()
        
        # Create a new grid with the dimensions of the bounding box
        new_rows = max_row - min_row + 1
        new_cols = max_col - min_col + 1
        output_grid = np.zeros((new_rows, new_cols), dtype=int)
        
        # Copy only the minor colors to the output grid
        for i, r in enumerate(range(min_row, max_row + 1)):
            for j, c in enumerate(range(min_col, max_col + 1)):
                if input_grid[r, c] in minor_colors:
                    output_grid[i, j] = input_grid[r, c]
        
        return output_grid
    
    def create_grids(self):
        # Generate random number of training pairs
        num_train_pairs = random.randint(3, 5)
        
        train_pairs = []
        for _ in range(num_train_pairs):
            input_grid = self.create_input()
            output_grid = self.transform_input(input_grid)
            train_pairs.append(GridPair(input=input_grid, output=output_grid))
        
        # Generate test pair
        test_input = self.create_input()
        test_output = self.transform_input(test_input)
        test_pairs = [GridPair(input=test_input, output=test_output)]
        
        return self.taskvars_definitions, TrainTestData(train=train_pairs, test=test_pairs)  # Return taskvars_definitions instead of taskvars
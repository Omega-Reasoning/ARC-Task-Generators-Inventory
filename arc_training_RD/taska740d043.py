from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from input_library import create_object, Contiguity, random_cell_coloring
from transformation_library import find_connected_objects

class Taska740d043Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids can have different sizes.",
            "They contain three colored cells (between 1-9): {color('object_color1')} color, {color('object_color2')} color and {color('object_color3')} color.",
            "The {color('object_color1')} cells are always in excess.",
            "The {color('object_color2')} cells and {color('object_color3')} cells are in smaller quantities compared to {color('object_color1')} cells.",
            "The cells of each color can form connected objects in the grid.",
            "Objects of different colors can be adjacent to each other, forming combined structures, but are still considered distinct objects based on their colors."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is different and is determined by the less occurring color cells.",
            "So, the colors which were less occurring on the input grid, forms a new grid of size comprising only the less occurring two colors which are {color('object_color2')} and {color('object_color3')}.",
            "The output grid will be a smaller grid that contains only the cells of the two less occurring colors.",
            "The relative positions and connections between the less occurring color objects are maintained in the output grid."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def color_name(self, color: int) -> str:
        color_map = {
            0: "black",
            1: "blue",
            2: "red",
            3: "green",
            4: "yellow",
            5: "gray",
            6: "magenta",
            7: "orange",
            8: "cyan",
            9: "brown"
        }
        return color_map.get(color, f"color_{color}")
    
    def create_input(self, taskvars):
        # Get colors from taskvars
        excess_color = taskvars["object_color1"]
        minor_color1 = taskvars["object_color2"]
        minor_color2 = taskvars["object_color3"]
        
        # Random grid size between 7x7 and 10x10
        rows = random.randint(7, 10)
        cols = random.randint(7, 10)
        
        # Create the grid filled with the excess color
        grid = np.full((rows, cols), excess_color, dtype=int)
        
        # Decide whether to create separate objects or combined objects
        object_type = random.choice(["separate", "separate", "combined"])
        
        if object_type == "separate":
            # Create separate objects for each minor color
            
            # First minor color object
            center1_row = random.randint(1, rows-3)
            center1_col = random.randint(1, cols-3)
            
            # Create a connected object for the first minor color
            num_cells1 = random.randint(3, 5)
            cells_added = 1
            grid[center1_row, center1_col] = minor_color1
            
            # Start with the center and add adjacent cells
            potential_cells = [(center1_row+1, center1_col), (center1_row-1, center1_col), 
                              (center1_row, center1_col+1), (center1_row, center1_col-1)]
            random.shuffle(potential_cells)
            
            while cells_added < num_cells1 and potential_cells:
                r, c = potential_cells.pop(0)
                if 0 <= r < rows and 0 <= c < cols and grid[r, c] == excess_color:
                    grid[r, c] = minor_color1
                    cells_added += 1
                    
                    # Add new adjacent cells to consider
                    for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                        new_r, new_c = r + dr, c + dc
                        if (0 <= new_r < rows and 0 <= new_c < cols and 
                            grid[new_r, new_c] == excess_color and
                            (new_r, new_c) not in potential_cells):
                            potential_cells.append((new_r, new_c))
            
            # Find a distant location for the second minor color
            minor1_positions = [(r, c) for r in range(rows) for c in range(cols) if grid[r, c] == minor_color1]
            
            while True:
                center2_row = random.randint(1, rows-3)
                center2_col = random.randint(1, cols-3)
                
                # Check if the center is far enough from any cell of the first minor color
                if all(abs(center2_row - r) + abs(center2_col - c) > 3 for r, c in minor1_positions):
                    break
            
            # Create a connected object for the second minor color
            num_cells2 = random.randint(3, 5)
            cells_added = 1
            grid[center2_row, center2_col] = minor_color2
            
            # Start with the center and add adjacent cells
            potential_cells = [(center2_row+1, center2_col), (center2_row-1, center2_col), 
                              (center2_row, center2_col+1), (center2_row, center2_col-1)]
            random.shuffle(potential_cells)
            
            while cells_added < num_cells2 and potential_cells:
                r, c = potential_cells.pop(0)
                if (0 <= r < rows and 0 <= c < cols and 
                    grid[r, c] == excess_color):
                    grid[r, c] = minor_color2
                    cells_added += 1
                    
                    # Add new adjacent cells to consider
                    for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                        new_r, new_c = r + dr, c + dc
                        if (0 <= new_r < rows and 0 <= new_c < cols and 
                            grid[new_r, new_c] == excess_color and
                            (new_r, new_c) not in potential_cells):
                            potential_cells.append((new_r, new_c))
                    
        else:  # combined objects
            # Create combined objects where the two minor colors are adjacent
            
            # Choose a center for the combined structure
            center_row = random.randint(2, rows-3)
            center_col = random.randint(2, cols-3)
            
            # First, create a small object with the first minor color
            num_cells1 = random.randint(3, 5)
            cells_added = 1
            grid[center_row, center_col] = minor_color1
            
            color1_cells = [(center_row, center_col)]
            
            # Start with the center and add adjacent cells
            potential_cells = [(center_row+1, center_col), (center_row-1, center_col), 
                              (center_row, center_col+1), (center_row, center_col-1)]
            random.shuffle(potential_cells)
            
            while cells_added < num_cells1 and potential_cells:
                r, c = potential_cells.pop(0)
                if 0 <= r < rows and 0 <= c < cols and grid[r, c] == excess_color:
                    grid[r, c] = minor_color1
                    cells_added += 1
                    color1_cells.append((r, c))
                    
                    # Add new adjacent cells to consider
                    for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                        new_r, new_c = r + dr, c + dc
                        if (0 <= new_r < rows and 0 <= new_c < cols and 
                            grid[new_r, new_c] == excess_color and
                            (new_r, new_c) not in potential_cells):
                            potential_cells.append((new_r, new_c))
            
            # Now, create an object with the second minor color adjacent to the first
            # Collect all possible starting points (adjacent to color1 cells)
            potential_starting_points = []
            for r, c in color1_cells:
                for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    new_r, new_c = r + dr, c + dc
                    if (0 <= new_r < rows and 0 <= new_c < cols and 
                        grid[new_r, new_c] == excess_color):
                        potential_starting_points.append((new_r, new_c))
            
            if potential_starting_points:
                # Choose a random starting point adjacent to color1
                start_r, start_c = random.choice(potential_starting_points)
                grid[start_r, start_c] = minor_color2
                
                # Grow the second object
                num_cells2 = random.randint(3, 5)
                cells_added = 1
                
                potential_cells = []
                for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    new_r, new_c = start_r + dr, start_c + dc
                    if (0 <= new_r < rows and 0 <= new_c < cols and 
                        grid[new_r, new_c] == excess_color):
                        potential_cells.append((new_r, new_c))
                
                while cells_added < num_cells2 and potential_cells:
                    r, c = potential_cells.pop(0)
                    if 0 <= r < rows and 0 <= c < cols and grid[r, c] == excess_color:
                        grid[r, c] = minor_color2
                        cells_added += 1
                        
                        # Add new adjacent cells to consider
                        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                            new_r, new_c = r + dr, c + dc
                            if (0 <= new_r < rows and 0 <= new_c < cols and 
                                grid[new_r, new_c] == excess_color and
                                (new_r, new_c) not in potential_cells):
                                potential_cells.append((new_r, new_c))
            
            else:
                # If we couldn't create a combined object, create a separate one instead
                # Find a distant location for the second minor color
                while True:
                    center2_row = random.randint(1, rows-3)
                    center2_col = random.randint(1, cols-3)
                    
                    # Check if the position is empty and far from first color
                    if grid[center2_row, center2_col] == excess_color:
                        break
                
                # Create a connected object for the second minor color
                num_cells2 = random.randint(3, 5)
                cells_added = 1
                grid[center2_row, center2_col] = minor_color2
                
                # Start with the center and add adjacent cells
                potential_cells = [(center2_row+1, center2_col), (center2_row-1, center2_col), 
                                  (center2_row, center2_col+1), (center2_row, center2_col-1)]
                random.shuffle(potential_cells)
                
                while cells_added < num_cells2 and potential_cells:
                    r, c = potential_cells.pop(0)
                    if (0 <= r < rows and 0 <= c < cols and 
                        grid[r, c] == excess_color):
                        grid[r, c] = minor_color2
                        cells_added += 1
                        
                        # Add new adjacent cells to consider
                        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                            new_r, new_c = r + dr, c + dc
                            if (0 <= new_r < rows and 0 <= new_c < cols and 
                                grid[new_r, new_c] == excess_color and
                                (new_r, new_c) not in potential_cells):
                                potential_cells.append((new_r, new_c))
                
        return grid
    
    def transform_input(self, input_grid):
        unique_colors, counts = np.unique(input_grid, return_counts=True)
        
        # Find the dominant color (most frequent)
        excess_color = unique_colors[np.argmax(counts)]
        
        # The other two colors are the less frequent ones
        minor_colors = [color for color in unique_colors if color != excess_color]
        
        # Handle edge case where there might be fewer than 3 colors
        if len(minor_colors) < 2:
            return np.zeros((1, 1), dtype=int)
        
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
        # Select three different colors between 1 and 9
        available_colors = list(range(1, 10))
        colors = random.sample(available_colors, 3)
        
        # Store colors in taskvars
        taskvars = {
            "object_color1": colors[0],  # Excess color
            "object_color2": colors[1],  # Minor color 1
            "object_color3": colors[2]   # Minor color 2
        }
        
        # Store taskvars as instance variable for access in transform_input
        self.taskvars = taskvars
        
        # Helper for reasoning chain formatting
        def color_fmt(key):
            color_id = taskvars[key]
            return f"{self.color_name(color_id)} ({color_id})"

        # Replace {color('object_color1')} etc. in reasoning chains
        self.input_reasoning_chain = [
            chain.replace("{color('object_color1')}", color_fmt('object_color1'))
                 .replace("{color('object_color2')}", color_fmt('object_color2'))
                 .replace("{color('object_color3')}", color_fmt('object_color3'))
            for chain in self.input_reasoning_chain
        ]
        self.transformation_reasoning_chain = [
            chain.replace("{color('object_color1')}", color_fmt('object_color1'))
                 .replace("{color('object_color2')}", color_fmt('object_color2'))
                 .replace("{color('object_color3')}", color_fmt('object_color3'))
            for chain in self.transformation_reasoning_chain
        ]
        
        # Generate random number of training pairs
        num_train_pairs = random.randint(3, 5)
        
        train_pairs = []
        for _ in range(num_train_pairs):
            input_grid = self.create_input(taskvars)
            output_grid = self.transform_input(input_grid)
            train_pairs.append(GridPair(input=input_grid, output=output_grid))
        
        # Generate test pair
        test_input = self.create_input(taskvars)
        test_output = self.transform_input(test_input)
        test_pairs = [GridPair(input=test_input, output=test_output)]
        
        return taskvars, TrainTestData(train=train_pairs, test=test_pairs)
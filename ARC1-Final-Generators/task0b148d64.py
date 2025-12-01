from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from input_library import create_object, Contiguity
from transformation_library import find_connected_objects

class Task0b148d64Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids have size {vars['rows']}x{vars['columns']}",
            "The input grid has four 8-way connected objects which form a subgrid of size {vars['sub_rows']}x{vars['sub_cols']}",
            "Three objects have color color_1(between 1-9) and one object has color color_2(between 1-9)",
            "The four objects always take up the four corners of the input grid."
        ]
        transformation_reasoning_chain = [
            "The output grid size is different from the input grid size.",
            "First identify all the 8-way connected objects in the input grid.",
            "The sub grid which has an object that has color color_2 is the output grid."
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars, gridvars):
        rows = taskvars['rows']
        cols = taskvars['columns']
        sub_rows = taskvars['sub_rows']
        sub_cols = taskvars['sub_cols']
        color_1 = gridvars['color_1']
        color_2 = gridvars['color_2']

        grid = np.zeros((rows, cols), dtype=int)

        corners = [
            (0, 0),  # Top-left
            (0, cols - sub_cols),  # Top-right
            (rows - sub_rows, 0),  # Bottom-left
            (rows - sub_rows, cols - sub_cols)  # Bottom-right
        ]

        colors = [color_1, color_1, color_1, color_2]
        random.shuffle(colors)

        for corner, color in zip(corners, colors):
            # Create object starting from the corner
            obj = np.zeros((sub_rows, sub_cols), dtype=int)
            corner_in_subgrid = (0 if corner[0] == 0 else sub_rows-1, 
                                0 if corner[1] == 0 else sub_cols-1)
            
            # Start with the corner pixel
            obj[corner_in_subgrid[0], corner_in_subgrid[1]] = color
            
            # Randomly grow the object from the corner using 8-way connectivity
            pixels_to_fill = random.randint(sub_rows * sub_cols // 4, sub_rows * sub_cols // 2)
            filled = 1
            
            while filled < pixels_to_fill:
                # Find all empty pixels that are 8-way adjacent to colored pixels
                colored = np.argwhere(obj == color)
                candidates = set()
                
                for px, py in colored:
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            nx, ny = px + dx, py + dy
                            if (0 <= nx < sub_rows and 0 <= ny < sub_cols and 
                                obj[nx, ny] == 0):
                                candidates.add((nx, ny))
                
                if not candidates:
                    break
                    
                # Randomly select and fill one candidate
                next_pixel = random.choice(list(candidates))
                obj[next_pixel[0], next_pixel[1]] = color
                filled += 1

            grid[corner[0]:corner[0] + sub_rows, corner[1]:corner[1] + sub_cols] = obj

        return grid

    def transform_input(self, grid, taskvars):
        # Find connected objects
        objects = find_connected_objects(grid, diagonal_connectivity=True)
        # Count how many objects contain each color
        color_counts = {}
        for obj in objects:
            for color in obj.colors:
                color_counts[color] = color_counts.get(color, 0) + 1

        # Prefer the color that appears exactly once. If none, pick the least
        # frequent color as a fallback (this avoids IndexError when none appear
        # exactly once due to unexpected grid compositions).
        candidates = [color for color, count in color_counts.items() if count == 1]
        if candidates:
            color_2 = candidates[0]
        else:
            if not color_counts:
                raise ValueError("No non-background objects found in the grid.")
            # Choose the color with minimal object count (tie-breaker arbitrary)
            color_2 = min(color_counts.items(), key=lambda kv: kv[1])[0]

        # Locate the object with color_2 and return its cropped array
        for obj in objects:
            if color_2 in obj.colors:
                # Use GridObject.to_array() which returns the minimal bounding array
                return obj.to_array()

        raise ValueError("Object with specified color not found in the grid.")

    def create_grids(self):
        # Base task variables
        taskvars = {
            'rows': random.randint(15, 26),
            'columns': random.randint(15, 26),
            'sub_rows': random.randint(5, 10),
            'sub_cols': random.randint(5, 10),
        }

        # Determine number of examples needed
        nr_train = random.randint(3, 4)
        nr_test = 1
        total_examples = nr_train + nr_test

        # Generate unique color pairs for each example
        all_colors = []
        available_colors = list(range(1, 10))
        
        for _ in range(total_examples):
            # Pick two different random colors
            color_1 = random.choice(available_colors)
            available_colors.remove(color_1)
            
            remaining_colors = available_colors.copy()  # Create a copy for color_2 selection
            color_2 = random.choice(remaining_colors)
            
            all_colors.append((color_1, color_2))
            
            # Put color_1 back for next iterations
            available_colors.append(color_1)
            random.shuffle(available_colors)

        # Create examples with different color pairs
        train_examples = []
        for i in range(nr_train):
            gridvars = taskvars.copy()
            gridvars['color_1'], gridvars['color_2'] = all_colors[i]
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            train_examples.append(GridPair({"input": input_grid, "output": output_grid}))

        # Create test example with the last color pair
        gridvars['color_1'], gridvars['color_2'] = all_colors[-1]
        test_input = self.create_input(taskvars, gridvars)
        test_output = self.transform_input(test_input, taskvars)
        test_examples = [GridPair({"input": test_input, "output": test_output})]

        return taskvars, TrainTestData({"train": train_examples, "test": test_examples})


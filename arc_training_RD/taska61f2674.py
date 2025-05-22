from input_library import create_object, enforce_object_height, retry
from transformation_library import GridObject, find_connected_objects
import numpy as np
import random
from typing import Dict, List, Tuple
from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData

class Taska61f2674Generator(ARCTaskGenerator):
    def __init__(self):
        # Use placeholders in the reasoning chains
        input_reasoning_chain = [
            "Input grids are squares and can have different sizes.",
            "They only contain {color('object_color')} color objects and empty (0) cells.",
            "The {color('object_color')} color cells form tower-like vertical structures of varying sizes and are placed randomly.",
            "Every alternative column must be left empty beside the tower.",
            "No two towers can have the same height."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid and identifying the tallest and the shortest tower only.",
            "Once identified, fill the longest tower with {color('fill_color1')} color and the shortest tower with {color('fill_color2')} color."
        ]
        
        # Initialize with default reasoning chains
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
        object_color = taskvars['object_color']
        fill_color1 = taskvars['fill_color1']   
        fill_color2 = taskvars['fill_color2']

        # Calculate minimum size needed for 4 towers (need at least 7 columns)
        min_size = 9  # 4 towers + 5 empty columns
        
        # Randomly determine grid size (between min_size and 20)
        size = random.randint(min_size, 20)
        
        # Create empty grid
        grid = np.zeros((size, size), dtype=int)
        
        # Use the object color from taskvars
        object_color = taskvars['object_color']
        
        # Determine number of towers (4-6, but limited by available space)
        max_possible_towers = size // 2  # Every other column
        min_towers = 4  # Ensure at least 4 towers
        max_towers = min(6, max_possible_towers)
        
        # Ensure we can fit the minimum number of towers
        if max_possible_towers < min_towers:
            min_towers = max_possible_towers
        
        num_towers = random.randint(min_towers, max_towers)
        
        # Generate unique tower heights (between 2 and size-1)
        possible_heights = list(range(2, size))
        random.shuffle(possible_heights)
        tower_heights = possible_heights[:num_towers]
        
        # Select column positions (every other column)
        available_columns = list(range(0, size, 2))
        random.shuffle(available_columns)
        tower_columns = available_columns[:num_towers]
        
        # Create towers fixed to the bottom row
        for height, col in zip(tower_heights, tower_columns):
            start_row = size - height  # Ensure the tower is fixed to the bottom
            for row in range(start_row, size):
                grid[row, col] = object_color
        
        return grid

    def transform_input(self, input_grid: np.ndarray) -> np.ndarray:
        # Make a copy of the input grid
        output_grid = np.zeros_like(input_grid)  # Start with an empty grid
        
        # Find all connected objects (towers)
        objects = find_connected_objects(input_grid, diagonal_connectivity=False)
        
        if len(objects) < 2:
            # If there's only one tower, just return it as is
            return input_grid
        
        # Sort objects by height (tallest first)
        sorted_objects = sorted(objects, key=lambda obj: obj.height, reverse=True)
        
        tallest = sorted_objects[0]
        shortest = sorted_objects[-1]
        
        # Get colors from taskvars
        fill_color1 = self.taskvars['fill_color1']
        fill_color2 = self.taskvars['fill_color2']
        
        # Color only the tallest and shortest towers
        for r, c, _ in tallest.cells:
            output_grid[r, c] = fill_color1
        for r, c, _ in shortest.cells:
            output_grid[r, c] = fill_color2
        
        return output_grid

    def create_grids(self) -> Tuple[Dict, TrainTestData]:
        # Generate random colors that are all different
        object_color = random.randint(1, 9)
        fill_color1 = random.choice([c for c in range(1, 10) if c != object_color])
        fill_color2 = random.choice([c for c in range(1, 10) if c not in [object_color, fill_color1]])

        # Store color ids for logic
        taskvars = {
            'object_color': object_color,
            'fill_color1': fill_color1,
            'fill_color2': fill_color2,
        }

        # Helper for reasoning chain formatting
        def color_fmt(key):
            color_id = taskvars[key]
            return f"{self.color_name(color_id)} ({color_id})"

        # Make taskvars available to transform_input
        self.taskvars = taskvars

        # Replace {color('object_color')} etc. in reasoning chains
        self.input_reasoning_chain = [
            chain.replace("{color('object_color')}", color_fmt('object_color'))
            for chain in self.input_reasoning_chain
        ]
        self.transformation_reasoning_chain = [
            chain.replace("{color('fill_color1')}", color_fmt('fill_color1'))
                 .replace("{color('fill_color2')}", color_fmt('fill_color2'))
            for chain in self.transformation_reasoning_chain
        ]

        # Create 3-5 train pairs and 1 test pair
        num_train = random.randint(3, 5)
        train_pairs = []
        for _ in range(num_train):
            input_grid = self.create_input(taskvars)
            output_grid = self.transform_input(input_grid)
            train_pairs.append(GridPair(input=input_grid, output=output_grid))

        # Create test pair
        test_input = self.create_input(taskvars)
        test_output = self.transform_input(test_input)
        test_pairs = [GridPair(input=test_input, output=test_output)]

        return taskvars, TrainTestData(train=train_pairs, test=test_pairs)
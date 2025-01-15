from typing import Tuple
from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from input_library import create_object, random_cell_coloring

class ARCTask0ca9ddb6Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are square matrices with size {vars['grid_size']}x{vars['grid_size']}.",
            "Some cells have color value (1-9) and all other cells are empty (0)."
        ]

        transformation_reasoning_chain = [
            "The output grid size is the same as the input grid size.",
            "If the color of the cell in the input grid is red (2), then four cells with color yellow (4) are filled diagonally around it, i.e. for a red cell at position (i, j), yellow cells are filled at positions (i-1, j-1), (i+1, j-1), (i-1, j+1), (i+1, j+1).",
            "If a cell in the input grid is blue (1), four orange cells (7) are placed around it. Specifically, for a blue cell at position (i, j), orange cells are added at the positions directly above (i−1, j), below (i+1, j), to the left (i, j−1), and to the right (i, j+1)."
        ]

        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        grid_size = taskvars['grid_size']
        grid = np.zeros((grid_size, grid_size), dtype=int)

        # Create a list of all possible positions
        all_positions = [(i, j) for i in range(grid_size) for j in range(grid_size)]
        
        # Function to get positions that are too close to existing colored cells
        def get_nearby_positions(pos, min_distance=2):
            nearby = []
            for i in range(-min_distance, min_distance + 1):
                for j in range(-min_distance, min_distance + 1):
                    ni, nj = pos[0] + i, pos[1] + j
                    if 0 <= ni < grid_size and 0 <= nj < grid_size:
                        nearby.append((ni, nj))
            return nearby

        # First, place one red (2) and one blue (1) cell with spacing
        available_positions = all_positions.copy()
        red_pos = random.choice(available_positions)
        grid[red_pos] = 2
        
        # Remove nearby positions for spacing
        for pos in get_nearby_positions(red_pos):
            if pos in available_positions:
                available_positions.remove(pos)
        
        if available_positions:  # Only place blue if we have space
            blue_pos = random.choice(available_positions)
            grid[blue_pos] = 1
            
            # Remove nearby positions again
            for pos in get_nearby_positions(blue_pos):
                if pos in available_positions:
                    available_positions.remove(pos)
        
        # Add additional colors (up to 8 more cells for a total of 10)
        num_additional = random.randint(0, min(8, len(available_positions)))
        if num_additional > 0 and available_positions:
            positions = random.sample(available_positions, k=num_additional)
            for pos in positions:
                grid[pos] = random.randint(1, 9)

        return grid

    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        grid_size = taskvars['grid_size']
        output_grid = grid.copy()

        directions_red = [(-1, -1), (1, -1), (-1, 1), (1, 1)]
        directions_blue = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for i in range(grid_size):
            for j in range(grid_size):
                if grid[i, j] == 2:  # Red cell
                    for di, dj in directions_red:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < grid_size and 0 <= nj < grid_size:
                            output_grid[ni, nj] = 4  # Yellow cells
                elif grid[i, j] == 1:  # Blue cell
                    for di, dj in directions_blue:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < grid_size and 0 <= nj < grid_size:
                            output_grid[ni, nj] = 7  # Orange cells

        return output_grid

    def create_grids(self) -> Tuple[dict, TrainTestData]:
        taskvars = {
            'grid_size': random.randint(9, 25)
        }

        num_train_examples = random.randint(3, 4)
        num_test_examples = 1

        train_test_data = self.create_grids_default(num_train_examples, num_test_examples, taskvars)

        return taskvars, train_test_data


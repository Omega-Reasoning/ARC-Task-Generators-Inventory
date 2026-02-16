from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

class TaskcrudzR6hQ9ZHsRbv4a5BHZ_1Generator(ARCTaskGenerator):
    def __init__(self):
        # 1) Input reasoning chain
        input_reasoning_chain = [
            "Input grids are squares and can have different sizes.",
            "They contain {color('cell_color1')}, {color('cell_color2')} and empty (0) cells.",
            "The {color('cell_color1')} cells are positioned at the center of each grid border, extending diagonally inward until they connect with another {color('cell_color1')} cell, forming a diamond-shaped frame.",
            "The interior of the {color('cell_color1')} frame contains {color('cell_color2')} diamond-shaped frames.",
            "The first {color('cell_color2')} frame appears after three layers of diamond-shaped frames made of empty (0) cells, and each subsequent {color('cell_color2')} frame follows after another three layers of empty (0) diamond-shaped frames."
        ]
        
        # 2) Transformation reasoning chain
        transformation_reasoning_chain = [
            "The output grid is created by copying the input grid and changing the color of four {color('cell_color1')} cells that are positioned at the center of each grid border to {color('cell_color3')}."
        ]
        
        # 3) Call parent constructor
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        """
        Creates a square grid containing a diamond frame pattern.
        - Blue (1) for outermost diamond frame.
        - Red (2) for alternating inner frames.
        """
        size = gridvars["grid_size"]
        c1 = taskvars["cell_color1"]  # Blue (1)
        c2 = taskvars["cell_color2"]  # Red (2)

        # Initialize grid
        grid = np.zeros((size, size), dtype=int)
        
        # Define center and radius
        center = size // 2
        radius = center  # Maximum Manhattan distance

        for k in range((radius // 2) + 1):
            i = radius - 2 * k
            if i < 0:
                break
            
            if k == 0:
                ring_color = c1  # Outermost frame (blue)
            else:
                ring_color = c2 if k % 2 == 0 else 0  # Alternate red and empty

            # Paint cells with Manhattan distance == i
            for row in range(size):
                for col in range(size):
                    if abs(row - center) + abs(col - center) == i:
                        grid[row, col] = ring_color
        
        return grid

    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        """
        Transforms the input grid:
        - Changes the four `cell_color1` cells at the center of each border to `cell_color3`.
        """
        out_grid = grid.copy()
        c1 = taskvars["cell_color1"]  # Blue (1)
        c3 = taskvars["cell_color3"]  # Another distinct color
        size = grid.shape[0]
        center = size // 2

        # Change the center cells of each border
        if out_grid[0, center] == c1:
            out_grid[0, center] = c3
        if out_grid[size - 1, center] == c1:
            out_grid[size - 1, center] = c3
        if out_grid[center, 0] == c1:
            out_grid[center, 0] = c3
        if out_grid[center, size - 1] == c1:
            out_grid[center, size - 1] = c3
        
        return out_grid

    def create_grids(self) -> tuple:
        """
        Creates:
         1) A dictionary of task variables ensuring:
            - Blue (1) for cell_color1
            - Red (2) for cell_color2
            - A distinct third color for cell_color3.
         2) Train and test data, ensuring distinct odd grid sizes.
        """
        # Force blue (1) and red (2) for correct visualization
        c1, c2 = random.choices(range(1, 10), k=2)
        c3 = random.choice([x for x in range(1, 10) if x not in [c1, c2]])  # Ensure c3 is different

        # Pick distinct odd sizes (7 to 29)
        possible_odd_sizes = list(range(11, 31, 2))  # 7, 9, 11, ..., 29
        random.shuffle(possible_odd_sizes)
        
        # Decide number of training examples (3 or 4)
        n_train = random.randint(3, 4)
        
        # Choose sizes
        train_sizes = possible_odd_sizes[:n_train]
        test_size = possible_odd_sizes[n_train]

        # Prepare task variables
        taskvars = {
            "cell_color1": c1,
            "cell_color2": c2,
            "cell_color3": c3
        }

        # Generate training pairs
        train_pairs = []
        for sz in train_sizes:
            gridvars = {"grid_size": sz}
            inp = self.create_input(taskvars, gridvars)
            out = self.transform_input(inp, taskvars)
            train_pairs.append(GridPair(input=inp, output=out))

        # Generate test grid
        test_pairs = []
        gridvars = {"grid_size": test_size}
        inp = self.create_input(taskvars, gridvars)
        out = self.transform_input(inp, taskvars)
        test_pairs.append(GridPair(input=inp, output=out))

        # Return task variables and train/test data
        traindata = TrainTestData(train=train_pairs, test=test_pairs)
        return taskvars, traindata




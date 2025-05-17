from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

class ConcentricLayersRotationGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are square grids of varying sizes.",
            "The grid consists of three zones :",
            "- Outer border (layer 1)- This is a thick edge area of the grid of {color(\"edge_color\")} color.",
            "- Middle Ring (Layer 2) - A ring area inside the outer border, usually empty(0) cells or {color(\"middle_color\")} color.",
            "- Inner block or Center (Layer 3) - A small square shape in the center, filled with {color(\"inner_color\")} color."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is copied from the input grid.",
            "The colors have shifted inward like a rotation, to be more precise:",
            "- The color from the outer border now fills the middle ring.",
            "- The color from the middle ring moved to the inner or center block.",
            "- The color from the center block becomes the outer border.",
            "So, colors rotate one layer inward and the center color goes to the outside layer."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self):
        # Choose a random grid size (odd numbers between 7 and 15 for better symmetry)
        size = random.choice([7, 9, 11, 13, 15])
        
        # Choose three distinct colors between 1 and 9
        colors = random.sample(range(1, 10), 3)
        self.edge_color = colors[0]
        self.middle_color = colors[1]
        self.inner_color = colors[2]
        
        # Create the grid
        grid = np.zeros((size, size), dtype=int)
        
        # Fill the outer border (Layer 1) - always 1 cell thick
        grid[0, :] = self.edge_color  # Top edge
        grid[-1, :] = self.edge_color  # Bottom edge
        grid[:, 0] = self.edge_color  # Left edge
        grid[:, -1] = self.edge_color  # Right edge
        
        # Calculate sizes for middle rings and center block
        center = size // 2
        center_size = 3  # Make center block 3x3
        center_start = center - center_size // 2
        center_end = center + center_size // 2 + 1
        
        # Calculate number of possible middle rings (excluding outer border and center)
        available_space = (center_start - 1)  # Space between border and center
        num_middle_rings = available_space // 2  # Each ring needs 1 cell + 1 space
        
        if num_middle_rings > 0:
            # Choose 1 or 2 rings to fill
            rings_to_fill = random.randint(1, min(2, num_middle_rings))
            
            # Choose ring positions with spacing
            ring_positions = sorted(random.sample(range(num_middle_rings), rings_to_fill))
            
            # Fill selected rings with middle_color
            for ring_idx in ring_positions:
                ring_pos = ring_idx * 2 + 1  # Skip border and ensure spacing
                # Fill complete ring - all four sides
                grid[ring_pos, ring_pos:-ring_pos] = self.middle_color  # Top
                grid[-ring_pos-1, ring_pos:-ring_pos] = self.middle_color  # Bottom
                grid[ring_pos:-ring_pos, ring_pos] = self.middle_color  # Left
                grid[ring_pos:-ring_pos, -ring_pos-1] = self.middle_color  # Right
        
        # Fill the center block (3x3)
        grid[center_start:center_end, center_start:center_end] = self.inner_color
        
        return grid

    def transform_input(self, grid):
        size = grid.shape[0]
        out_grid = np.zeros_like(grid)
        
        # Find center block size from input grid
        center = size // 2
        # Find center block by checking neighborhood
        center_size = 3
        center_start = center - center_size // 2
        center_end = center + center_size // 2 + 1
        
        # 1. Outer border becomes inner_color (previous center color)
        out_grid[0, :] = self.inner_color  # Top
        out_grid[-1, :] = self.inner_color  # Bottom
        out_grid[:, 0] = self.inner_color  # Left
        out_grid[:, -1] = self.inner_color  # Right
        
        # 2. Middle rings become edge_color (previous border color)
        for i in range(1, (size-1)//2, 2):  # Step by 2 to maintain spacing
            if np.any(grid[i, i:-i] == self.middle_color):
                # Fill complete ring with edge_color
                out_grid[i, i:-i] = self.edge_color  # Top
                out_grid[-i-1, i:-i] = self.edge_color  # Bottom
                out_grid[i:-i, i] = self.edge_color  # Left
                out_grid[i:-i, -i-1] = self.edge_color  # Right
        
        # 3. Center becomes middle_color (previous ring color)
        out_grid[center_start:center_end, center_start:center_end] = self.middle_color
        
        return out_grid

    def create_grids(self):
        # We'll create 3-5 training pairs
        num_train_pairs = random.randint(3, 5)
        train_pairs = []
        
        # First create an initial input to set the colors
        initial_grid = self.create_input()
        
        # Now we can define task variables using the set colors
        task_variables = {
            "edge_color": self.edge_color,
            "middle_color": self.middle_color,
            "inner_color": self.inner_color
        }
        
        # Create training pairs
        for _ in range(num_train_pairs):
            input_grid = self.create_input()
            output_grid = self.transform_input(input_grid)
            train_pairs.append(GridPair(input=input_grid, output=output_grid))
        
        # Generate a test pair
        test_input = self.create_input()
        test_output = self.transform_input(test_input)
        test_pairs = [GridPair(input=test_input, output=test_output)]
        
        return task_variables, TrainTestData(train=train_pairs, test=test_pairs)

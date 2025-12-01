from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from input_library import create_object, Contiguity
from transformation_library import find_connected_objects

class Task445eab21Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['grid_size']}x{vars['grid_size']}.",
            "They consist of exactly two colored (1-9) one-cell wide rectangular frames, each fully separated from the other.",
            "Both rectangular frames have different colors within one grid, with colors varying across examples.",
            "The size of both rectangular frames must be different from each other, with both having a length and width greater than two.",
            "All rectangular frames have a completely empty (0) interior."
        ]
        
        transformation_reasoning_chain = [
            "The output grids are of size 2x2.",
            "They are constructed by identifying the two rectangular frames in the input grid and finding the one that covers more area.",
            "Once found, fill the entire output grid with the color of the frame that covers more area."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self) -> tuple[dict[str, any], TrainTestData]:
        # Initialize task variables
        taskvars = {
            'grid_size': random.randint(10, 30)
        }
        
        # Generate train examples (3-4)
        num_train_examples = random.randint(3, 4)
        train_examples = []
        
        for _ in range(num_train_examples):
            # Generate a unique pair of colors for each example
            colors = random.sample(range(1, 10), 2)
            gridvars = {'colors': colors}
            
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Generate test example
        test_colors = random.sample(range(1, 10), 2)
        test_gridvars = {'colors': test_colors}
        
        test_input = self.create_input(taskvars, test_gridvars)
        test_output = self.transform_input(test_input, taskvars)
        
        test_examples = [{
            'input': test_input,
            'output': test_output
        }]
        
        return taskvars, {
            'train': train_examples,
            'test': test_examples
        }
    
    def create_input(self, taskvars: dict[str, any], gridvars: dict[str, any]) -> np.ndarray:
        grid_size = taskvars['grid_size']
        colors = gridvars['colors']
        
        # Initialize empty grid
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Create the first rectangular frame
        frame1_height = random.randint(3, grid_size // 2)
        frame1_width = random.randint(3, grid_size // 2)
        
        # Create the second rectangular frame with different dimensions
        frame2_height = random.randint(3, grid_size // 2)
        frame2_width = random.randint(3, grid_size // 2)
        
        # Ensure the frames have different areas
        while frame1_height * frame1_width == frame2_height * frame2_width:
            frame2_height = random.randint(3, grid_size // 2)
            frame2_width = random.randint(3, grid_size // 2)
        
        # Place the first frame
        start_row1 = random.randint(0, grid_size - frame1_height)
        start_col1 = random.randint(0, grid_size - frame1_width)
        
        # Draw the first frame
        for r in range(start_row1, start_row1 + frame1_height):
            for c in range(start_col1, start_col1 + frame1_width):
                # Only draw the border
                if (r == start_row1 or r == start_row1 + frame1_height - 1 or 
                    c == start_col1 or c == start_col1 + frame1_width - 1):
                    grid[r, c] = colors[0]
        
        # Determine valid placement for the second frame
        valid_placement = False
        attempts = 0
        max_attempts = 100
        
        while not valid_placement and attempts < max_attempts:
            attempts += 1
            
            # Try a random position for the second frame
            start_row2 = random.randint(0, grid_size - frame2_height)
            start_col2 = random.randint(0, grid_size - frame2_width)
            
            # Check if the second frame overlaps with the first frame
            overlap = False
            for r in range(start_row2, start_row2 + frame2_height):
                for c in range(start_col2, start_col2 + frame2_width):
                    if (r == start_row2 or r == start_row2 + frame2_height - 1 or 
                        c == start_col2 or c == start_col2 + frame2_width - 1):
                        # If the border pixel of frame2 overlaps with any pixel of frame1
                        if grid[r, c] != 0:
                            overlap = True
                            break
                if overlap:
                    break
            
            if not overlap:
                valid_placement = True
        
        if valid_placement:
            # Draw the second frame
            for r in range(start_row2, start_row2 + frame2_height):
                for c in range(start_col2, start_col2 + frame2_width):
                    # Only draw the border
                    if (r == start_row2 or r == start_row2 + frame2_height - 1 or 
                        c == start_col2 or c == start_col2 + frame2_width - 1):
                        grid[r, c] = colors[1]
        else:
            # If we couldn't place the second frame, start over with a smaller grid
            return self.create_input(taskvars, gridvars)
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: dict[str, any]) -> np.ndarray:
        # Find all objects (connected components) in the grid
        objects = find_connected_objects(grid, diagonal_connectivity=True, background=0, monochromatic=True)
        
        # Find the two rectangular frames
        frames = []
        for obj in objects:
            if len(obj.colors) == 1:  # Should be monochromatic
                frames.append(obj)
        
        if len(frames) != 2:
            raise ValueError(f"Expected 2 frames, found {len(frames)}")
        
        # Calculate the area (perimeter) of each frame
        areas = [len(frame) for frame in frames]
        
        # Find the color of the frame with larger area
        dominant_color = frames[0].colors.pop() if areas[0] > areas[1] else frames[1].colors.pop()
        
        # Create the 2x2 output grid with the dominant color
        output = np.full((2, 2), dominant_color, dtype=int)
        
        return output


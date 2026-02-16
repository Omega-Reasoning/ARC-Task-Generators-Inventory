from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from Framework.input_library import create_object, Contiguity
from Framework.transformation_library import find_connected_objects, GridObject, GridObjects

class Task0962bcddGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['rows']} x {vars['cols']}..",
            "Each grid contains several flower-shaped objects, with the remaining cells being empty (0).",
            "A flower shape is defined as [[0, c, 0], [c, t, c], [0, c, 0]], where c and t are two different colors.",
            "The colors used within a grid are fixed but vary across different grids.",
            "Each flower shape is placed at the center of an empty 5x5 subgrid, ensuring a one-cell wide empty (0) frame around it.",
            "The positions of the flower shapes must vary across examples."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid and identifying the flower shapes, where each flower shape is defined as [[0, c, 0], [c, t, c], [0, c, 0]], with c and t being two different colors.",
            "Once identified, extend each flower shape by adding four c-colored cells—two above and two below, and two to the left and two to the right—forming vertical and horizontal lines. This results in the intermediate pattern:[[0, 0, c, 0, 0], [0, 0, c, 0, 0], [c, c, t, c, c], [0, 0, c, 0, 0], [0, 0, c, 0, 0]].",
            "Next, add eight t-colored cells to complete the shape, producing the final structure:[[t, 0, c, 0, t], [0, t, c, t, 0], [c, c, t, c, c], [0, t, c, t, 0], [t, 0, c, 0, t]]."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_flower(self, center_r, center_c, petal_color, center_color, grid):
        """Create a flower at the specified location"""
        # Center cell
        grid[center_r, center_c] = center_color
        
        # Petals (cross pattern)
        grid[center_r-1, center_c] = petal_color
        grid[center_r+1, center_c] = petal_color
        grid[center_r, center_c-1] = petal_color
        grid[center_r, center_c+1] = petal_color
        
        return grid
    
    def create_input(self, taskvars, gridvars):
        rows = taskvars['rows']
        cols = taskvars['cols']
        
        # Extract the flower colors from gridvars
        petal_color = gridvars['petal_color']
        center_color = gridvars['center_color']
        
        # Create an empty grid
        grid = np.zeros((rows, cols), dtype=int)
        
        # Define valid positions for flower centers
        # We need enough empty space around each flower to allow transformation
        valid_positions = []
        for r in range(3, rows - 3):
            for c in range(3, cols - 3):
                valid_positions.append((r, c))
        
        # Decide how many flowers to place
        max_flowers = rows // 5
        num_flowers = random.randint(1, max_flowers)
        
        # Randomize candidate positions
        random.shuffle(valid_positions)
        
        flower_positions = []
        for pos in valid_positions:
            # Ensure flowers are far enough apart so their 5x5 expansions do not overlap
            if all(abs(pos[0] - r) > 5 or abs(pos[1] - c) > 5 for r, c in flower_positions):
                flower_positions.append(pos)
                if len(flower_positions) == num_flowers:
                    break
        
        # Place flowers at the selected positions
        for center_r, center_c in flower_positions:
            self.create_flower(center_r, center_c, petal_color, center_color, grid)
        
        return grid

    
    def transform_input(self, grid, taskvars):
        output_grid = grid.copy()
        
        # Find the flower shapes
        objects = find_connected_objects(grid, diagonal_connectivity=False, background=0, monochromatic=False)
        
        # Process each flower shape
        for obj in objects:
            # Convert cells to array for easier processing
            cells_arr = np.array(list(obj.coords))
            
            # Get flower center position - it's the only cell surrounded by 4 petals
            for r, c in obj.coords:
                neighbors = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
                if all((nr, nc) in obj.coords for nr, nc in neighbors):
                    center_r, center_c = r, c
                    center_color = grid[r, c]
                    petal_color = grid[r-1, c]  # Get color from any petal
                    break
            
            # Extend the flower with additional petals
            # Add vertical and horizontal lines
            output_grid[center_r-2, center_c] = petal_color
            output_grid[center_r-1, center_c] = petal_color
            output_grid[center_r+1, center_c] = petal_color
            output_grid[center_r+2, center_c] = petal_color
            
            output_grid[center_r, center_c-2] = petal_color
            output_grid[center_r, center_c-1] = petal_color
            output_grid[center_r, center_c+1] = petal_color
            output_grid[center_r, center_c+2] = petal_color
            
            # Add the t-colored diagonal cells
            output_grid[center_r-2, center_c-2] = center_color
            output_grid[center_r-2, center_c+2] = center_color
            output_grid[center_r-1, center_c-1] = center_color
            output_grid[center_r-1, center_c+1] = center_color
            output_grid[center_r+1, center_c-1] = center_color
            output_grid[center_r+1, center_c+1] = center_color
            output_grid[center_r+2, center_c-2] = center_color
            output_grid[center_r+2, center_c+2] = center_color
            
        return output_grid
    
    def create_grids(self):
        # Choose random grid size between 10 and 30
        rows = random.randint(10, 30)
        cols = random.randint(10, 30)
        
        # Generate between 3 and 6 training examples
        num_train_examples = random.randint(3, 6)
        
        # Create task variables
        taskvars = {'rows': rows, 'cols': cols}
        
        train_examples = []
        used_color_pairs = set()
        
        for _ in range(num_train_examples):
            # Choose random colors for the flower
            while True:
                petal_color = random.randint(1, 9)
                center_color = random.randint(1, 9)
                if petal_color != center_color and (petal_color, center_color) not in used_color_pairs:
                    used_color_pairs.add((petal_color, center_color))
                    break
            
            # Create gridvars for this example
            gridvars = {'petal_color': petal_color, 'center_color': center_color}
            
            # Generate input and output grids
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Create test example with new colors
        while True:
            petal_color = random.randint(1, 9)
            center_color = random.randint(1, 9)
            if petal_color != center_color and (petal_color, center_color) not in used_color_pairs:
                break
        
        gridvars = {'petal_color': petal_color, 'center_color': center_color}
        test_input = self.create_input(taskvars, gridvars)
        test_output = self.transform_input(test_input, taskvars)
        
        test_examples = [{
            'input': test_input,
            'output': test_output
        }]
        
        return taskvars, {
            'train': train_examples,
            'test': test_examples
        }


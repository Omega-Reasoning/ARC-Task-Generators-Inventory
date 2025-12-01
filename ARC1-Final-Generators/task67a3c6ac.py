from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from input_library import create_object, Contiguity, retry
from transformation_library import find_connected_objects, GridObject

class Task67a3c6acGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids can have different sizes.",
            "They contain a completely filled grid with different colored objects in the colors {color('object_color1')}, {color('object_color2')}, {color('object_color3')}, and {color('object_color4')}.",
            "Each object is made of 4-way connected cells of the same color.",
            "The position and shapes of the objects can vary across examples.",
            "There are no empty (0) cells in the grid."
        ]
        
        transformation_reasoning_chain = [
            "The output grid has the same size as the input.",
            "They are constructed by reflecting the input grids horizontally to the right and pasting the mirrored version into the output grids.",
            "All colors, shapes, and positions of the objects are preserved relative to the mirrored layout."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self) -> tuple[dict[str, any], TrainTestData]:
        # Choose four different colors for the objects
        colors = random.sample(range(1, 10), 4)
        
        # Initialize task variables
        taskvars = {
            'object_color1': colors[0],
            'object_color2': colors[1],
            'object_color3': colors[2],
            'object_color4': colors[3],
        }
        
        # Generate 3-4 training examples and 1 test example
        num_train_examples = random.randint(3, 4)
        
        # Create train/test data
        train_test_data = self.create_grids_default(num_train_examples, 1, taskvars)
        
        return taskvars, train_test_data
    
    def create_input(self, taskvars: dict[str, any], gridvars: dict[str, any]) -> np.ndarray:
       # Randomly determine square grid size between 5x5 and 30x30
        size = random.randint(5, 30)
        height = width = size

        
        # Get the four colors for the objects
        colors = [
            taskvars['object_color1'],
            taskvars['object_color2'],
            taskvars['object_color3'],
            taskvars['object_color4']
        ]
        
        # Create a grid filled with random colors
        grid = np.zeros((height, width), dtype=int)
        
        # Start with a grid of random colors from our palette
        for r in range(height):
            for c in range(width):
                grid[r, c] = random.choice(colors)
        
        # Now we need to ensure 4-way connectivity for all objects
        # First, we'll find currently connected components
        components = find_connected_objects(grid, diagonal_connectivity=False, monochromatic=True)
        
        # If we already have valid objects (4-way connected by color), we can return
        if all(len(obj.cells) > 0 for obj in components):
            return grid
        
        # If not, we'll create a new grid where we ensure 4-way connectivity
        # This is done by growing regions from random seeds
        
        # Start with an empty grid
        new_grid = np.zeros((height, width), dtype=int)
        
        # Keep track of unfilled cells
        unfilled = set((r, c) for r in range(height) for c in range(width))
        
        # Continue while there are unfilled cells
        while unfilled:
            # Pick a random unfilled cell to start a new region
            if not unfilled:
                break
                
            start_r, start_c = random.choice(list(unfilled))
            color = random.choice(colors)
            
            # Grow region from this cell using BFS
            queue = [(start_r, start_c)]
            region = set()
            
            while queue and unfilled:
                r, c = queue.pop(0)
                
                if (r, c) not in unfilled:
                    continue
                
                # Add to region and mark as filled
                region.add((r, c))
                unfilled.remove((r, c))
                new_grid[r, c] = color
                
                # Add adjacent unfilled cells with some probability to create irregular shapes
                neighbors = []
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width and (nr, nc) in unfilled:
                        neighbors.append((nr, nc))
                
                # Add all or some neighbors to queue
                if neighbors:
                    # Randomly decide how many neighbors to add
                    if random.random() < 0.3 and len(neighbors) > 1:
                        # Add some subset of neighbors
                        queue.extend(random.sample(neighbors, random.randint(1, len(neighbors))))
                    else:
                        # Add all neighbors
                        queue.extend(neighbors)
        
        return new_grid
    
    def transform_input(self, grid: np.ndarray, taskvars: dict[str, any]) -> np.ndarray:
        # Horizontal reflection: simply flip the grid left-to-right
        return np.fliplr(grid)


from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects, BorderBehavior
from input_library import create_object, retry, Contiguity
import numpy as np
import random

class Taskd0f5fe59(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids can have different sizes.",
            "Each input grid contains n (n>1) objects.",
            "The objects are of {color('object_color')}.",
            "Every object is enclosed by a layer of empty cells.",
            "Each object consists of more than two adjacent cells that form one connected shape without gaps.",
            "The objects have different sizes."
        ]
        
        transformation_reasoning_chain = [
            "Output grid size is n * n.",
            "The entire main diagonal of the output grid is filled with the { color('object_color')}."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self) -> tuple[dict[str, any], TrainTestData]:
        # Define task variables
        taskvars = {
            'object_color': random.randint(1, 9)  # Random color between 1-9
        }
        
        # Create train examples (3-6 examples)
        num_train_examples = random.randint(3, 6)
        train_examples = []
        
        # Create train examples with varying grid sizes and number of objects
        for _ in range(num_train_examples):
            # Ensure grid size is large enough to accommodate multiple objects
            grid_size = random.randint(8, 15)
            # Adjust number of objects based on grid size to ensure they can fit
            max_objects = min(5, grid_size // 3)
            num_objects = random.randint(2, max(2, max_objects))
            
            gridvars = {
                'grid_size': grid_size,
                'num_objects': num_objects
            }
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            train_examples.append({'input': input_grid, 'output': output_grid})
        
        # Create test example with different properties
        test_grid_size = random.randint(10, 20)
        test_max_objects = min(6, test_grid_size // 3)
        test_num_objects = random.randint(3, max(3, test_max_objects))
        
        test_gridvars = {
            'grid_size': test_grid_size,
            'num_objects': test_num_objects
        }
        test_input = self.create_input(taskvars, test_gridvars)
        test_output = self.transform_input(test_input, taskvars)
        
        return taskvars, {
            'train': train_examples,
            'test': [{'input': test_input, 'output': test_output}]
        }
    
    def create_input(self, taskvars: dict[str, any], gridvars: dict[str, any]) -> np.ndarray:
        """Create an input grid with multiple objects of the specified color."""
        grid_size = gridvars['grid_size']
        num_objects = gridvars['num_objects']
        object_color = taskvars['object_color']
        
        # Initialize grid
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Determine maximum object size based on grid size and number of objects
        max_obj_size = min(5, (grid_size - 2) // 2)  # Allow space for borders
        
        # Create objects of different sizes
        object_sizes = self._generate_different_sizes(num_objects, min_size=3, max_size=max_obj_size)
        
        # Keep track of used positions to ensure objects are separated
        used_positions = set()
        
        for size in object_sizes:
            # Limit object dimensions to ensure it fits on the grid
            h, w = random.randint(2, min(size, max_obj_size)), random.randint(2, min(size, max_obj_size))
            
            def generate_object():
                # Generate a random object
                return create_object(
                    height=h,
                    width=w,
                    color_palette=object_color,
                    contiguity=Contiguity.FOUR,
                    background=0
                )
            
            # Ensure the object has at least 3 cells
            object_array = retry(
                generate_object,
                lambda arr: np.sum(arr > 0) >= 3 and np.sum(arr > 0) <= size
            )
            
            # Find a position where we can place the object with a border of empty cells
            max_attempts = 100
            placed = False
            
            for _ in range(max_attempts):
                if grid_size <= h + 2 or grid_size <= w + 2:
                    break  # Skip if object is too large for the grid
                
                # Random position for object, ensuring space for border
                r = random.randint(1, grid_size - h - 1)
                c = random.randint(1, grid_size - w - 1)
                
                # Check if the placement would overlap with existing objects or their borders
                overlap = False
                for dr in range(-1, h + 1):
                    for dc in range(-1, w + 1):
                        if 0 <= r + dr < grid_size and 0 <= c + dc < grid_size:
                            pos = (r + dr, c + dc)
                            if pos in used_positions:
                                overlap = True
                                break
                    if overlap:
                        break
                
                if not overlap:
                    # Place the object
                    for row in range(h):
                        for col in range(w):
                            if object_array[row, col] > 0:
                                grid[r + row, c + col] = object_color
                                used_positions.add((r + row, c + col))
                    
                    # Mark border cells as used
                    for dr in range(-1, h + 1):
                        for dc in range(-1, w + 1):
                            if 0 <= r + dr < grid_size and 0 <= c + dc < grid_size:
                                used_positions.add((r + dr, c + dc))
                    
                    placed = True
                    break
            
            # If we couldn't place this object after max_attempts, continue to the next one
            if not placed:
                continue
        
        # Check if we have at least 2 objects in the grid
        objects = find_connected_objects(grid, diagonal_connectivity=False, background=0)
        
        # If not enough objects were placed, regenerate the grid
        if len(objects) < 2:
            # Adjust parameters to make it easier to place objects
            gridvars['num_objects'] = 2  # Try with minimum required objects
            return self.create_input(taskvars, gridvars)
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: dict[str, any]) -> np.ndarray:
        """Transform the input grid according to the transformation reasoning chain."""
        # Count the number of objects
        objects = find_connected_objects(grid, diagonal_connectivity=False, background=0)
        n = len(objects)
        
        # Create output grid of size n x n
        output_grid = np.zeros((n, n), dtype=int)
        
        # Fill the main diagonal with the object color (changed from anti-diagonal)
        object_color = taskvars['object_color']
        for i in range(n):
            output_grid[i, i] = object_color
        
        return output_grid
    
    def _generate_different_sizes(self, count: int, min_size: int, max_size: int) -> list[int]:
        """Generate 'count' different size values between min_size and max_size."""
        if max_size - min_size + 1 < count:
            # Not enough unique sizes available, adjust the range
            max_size = min_size + count - 1
        
        # Generate unique sizes if possible
        if max_size >= min_size:
            sizes = random.sample(range(min_size, max_size + 1), min(count, max_size - min_size + 1))
            # If we need more sizes, duplicate some
            while len(sizes) < count:
                sizes.append(random.choice(range(min_size, max_size + 1)))
        else:
            # Fallback if range is invalid
            sizes = [min_size] * count
            
        return sizes
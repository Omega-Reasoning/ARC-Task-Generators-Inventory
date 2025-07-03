from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects, GridObjects
from input_library import create_object, retry, Contiguity
import numpy as np
import random
from typing import Dict, Any, Tuple, List

class Taskf8ff0b80(ARCTaskGenerator):
    
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['n']} x {vars['n']}.",
            "Every grid contains exactly three objects, each differing in size.",
            "Each object is uniformly filled with a randomly selected color.",
            "The objects are completely isolated—none of them touch or overlap with any other object.",
            "The sizes and colors of the objects vary across different grids."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is of size 3 x 1.",
            "The output grid is constructed by first identifying the three objects in the input grid and determining the number of cells occupied by each.",
            "The cells of the output grid are then colored based on object size: the first cell is filled with the color of the largest object, the second with that of the medium-sized object, and the third with that of the smallest."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_single_object(self, grid: np.ndarray, color: int, target_size: int, max_attempts: int = 100) -> bool:
        """Create a single connected object of approximately target_size on the grid."""
        n = grid.shape[0]
        
        for attempt in range(max_attempts):
            # Start with a random seed position
            start_r = random.randint(0, n-1)
            start_c = random.randint(0, n-1)
            
            # Skip if position is already occupied
            if grid[start_r, start_c] != 0:
                continue
            
            # Build object by growing from seed
            object_cells = {(start_r, start_c)}
            grid[start_r, start_c] = color
            
            # Grow the object by adding neighboring cells
            for _ in range(target_size - 1):
                # Get all possible expansion positions
                candidates = []
                for r, c in object_cells:
                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nr, nc = r + dr, c + dc
                        if (0 <= nr < n and 0 <= nc < n and 
                            grid[nr, nc] == 0 and (nr, nc) not in object_cells):
                            candidates.append((nr, nc))
                
                if not candidates:
                    break  # Can't grow further
                
                # Choose a random candidate to add
                new_r, new_c = random.choice(candidates)
                object_cells.add((new_r, new_c))
                grid[new_r, new_c] = color
            
            # Check if object is isolated (no touching other objects)
            is_isolated = True
            for r, c in object_cells:
                for dr, dc in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < n and 0 <= nc < n and 
                        grid[nr, nc] != 0 and grid[nr, nc] != color):
                        is_isolated = False
                        break
                if not is_isolated:
                    break
            
            if is_isolated and len(object_cells) >= max(1, target_size // 2):
                return True
            else:
                # Remove the object and try again
                for r, c in object_cells:
                    grid[r, c] = 0
        
        return False
    
    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        n = taskvars['n']
        grid = np.zeros((n, n), dtype=int)
        
        # Generate three objects with different target sizes
        colors = random.sample(range(1, 10), 3)
        
        # Define target sizes based on grid size
        total_cells = n * n
        max_obj_size = min(total_cells // 6, 25)  # Leave room for all objects + spacing
        
        # Create three different size ranges
        small_size = random.randint(1, max(2, max_obj_size // 4))
        medium_size = random.randint(max(3, max_obj_size // 3), max(4, max_obj_size // 2))
        large_size = random.randint(max(5, max_obj_size // 2), max_obj_size)
        
        # Ensure sizes are different
        sizes = [small_size, medium_size, large_size]
        while len(set(sizes)) != 3:
            small_size = random.randint(1, max(2, max_obj_size // 4))
            medium_size = random.randint(max(3, max_obj_size // 3), max(4, max_obj_size // 2))
            large_size = random.randint(max(5, max_obj_size // 2), max_obj_size)
            sizes = [small_size, medium_size, large_size]
        
        # Shuffle the order of creation
        creation_order = list(zip(colors, sizes))
        random.shuffle(creation_order)
        
        # Create objects one by one
        for color, target_size in creation_order:
            success = self.create_single_object(grid, color, target_size)
            if not success:
                raise ValueError(f"Could not create object with color {color} and size {target_size}")
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        # Find all objects in the grid
        objects = find_connected_objects(grid, diagonal_connectivity=False, background=0, monochromatic=True)
        
        if len(objects) != 3:
            raise ValueError(f"Expected 3 objects, found {len(objects)}")
        
        # Get size and color for each object
        object_info = []
        for obj in objects:
            size = len(obj)
            color = next(iter(obj.colors))  # Get the single color
            object_info.append((size, color))
        
        # Sort by size (largest first)
        object_info.sort(key=lambda x: x[0], reverse=True)
        
        # Create 3x1 output grid
        output = np.zeros((3, 1), dtype=int)
        output[0, 0] = object_info[0][1]  # Largest object color
        output[1, 0] = object_info[1][1]  # Medium object color  
        output[2, 0] = object_info[2][1]  # Smallest object color
        
        return output
    
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Random grid size between 8 and 30 
        n = random.randint(8, 30)
        taskvars = {'n': n}
        
        # Generate 3-6 training examples and 1 test example
        num_train = random.randint(3, 6)
        
        def generate_examples(count):
            examples = []
            for _ in range(count):
                # Use retry to ensure we can generate valid grids
                input_grid = retry(
                    lambda: self.create_input(taskvars, {}),
                    lambda g: len(find_connected_objects(g, diagonal_connectivity=False, background=0, monochromatic=True)) == 3,
                    max_attempts=20
                )
                output_grid = self.transform_input(input_grid, taskvars)
                examples.append({'input': input_grid, 'output': output_grid})
            return examples
        
        train_examples = generate_examples(num_train)
        test_examples = generate_examples(1)
        
        return taskvars, {
            'train': train_examples,
            'test': test_examples
        }


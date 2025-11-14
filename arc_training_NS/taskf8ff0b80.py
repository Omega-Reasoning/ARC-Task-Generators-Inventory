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
            "Every grid contains exactly {vars['objects_num']} objects, each differing in size.",
            "Each object is uniformly filled with a randomly selected color.",
            "The objects are completely isolated—none of them touch or overlap with any other object.",
            "The sizes and colors of the objects vary across different grids."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is of size {vars['objects_num']} x 1.",
            "The output grid is constructed by first identifying all the objects in the input grid and determining the number of cells occupied by each.",
            "The objects are then sorted from largest to smallest. Each row of the output grid is filled with the color of the corresponding object in this sorted order: the top cell represents the largest object, and the bottom cell represents the smallest."
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
            for _ in range(max(0, target_size - 1)):
                # Get all possible expansion positions
                candidates = []
                for r, c in list(object_cells):
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
        objects_num = taskvars.get('objects_num', 3)
        grid = np.zeros((n, n), dtype=int)
        
        # Generate distinct colors for each object
        colors = random.sample(range(1, 10), objects_num)
        
        # Define target sizes based on grid size and number of objects
        total_cells = n * n
        max_obj_size = max(1, min(total_cells // (objects_num * 2), 25))
        
        # Generate unique sizes (sample without replacement when possible)
        if max_obj_size >= objects_num:
            sizes = random.sample(range(1, max_obj_size + 1), objects_num)
        else:
            # fallback: generate increasing sizes and then shuffle to avoid predictability
            sizes = list(range(1, objects_num + 1))
        
        # Shuffle the order of creation so sizes/colors are mixed
        creation_order = list(zip(colors, sizes))
        random.shuffle(creation_order)
        
        # Create objects one by one
        for color, target_size in creation_order:
            success = self.create_single_object(grid, color, target_size)
            if not success:
                # On failure, try a few more times with a reduced size
                reduced = max(1, target_size // 2)
                success = self.create_single_object(grid, color, reduced)
            if not success:
                raise ValueError(f"Could not create object with color {color} and size {target_size}")
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        objects_num = taskvars.get('objects_num', 3)
        # Find all objects in the grid
        objects = find_connected_objects(grid, diagonal_connectivity=False, background=0, monochromatic=True)
        
        if len(objects) != objects_num:
            raise ValueError(f"Expected {objects_num} objects, found {len(objects)}")
        
        # Get size and color for each object
        object_info = []
        for obj in objects:
            size = len(obj)
            color = next(iter(obj.colors))  # Get the single color
            object_info.append((size, color))
        
        # Sort by size (largest first)
        object_info.sort(key=lambda x: x[0], reverse=True)
        
        # Create output grid of shape objects_num x 1
        output = np.zeros((objects_num, 1), dtype=int)
        for idx, (_, color) in enumerate(object_info):
            output[idx, 0] = color
        
        return output
    
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Random grid size between 8 and 30 
        n = random.randint(8, 30)
        # Number of objects per input grid (choose a feasible range)
        objects_num = random.randint(2, 6)
        taskvars = {'n': n, 'objects_num': objects_num}
        
        # Generate 3-6 training examples and 1 test example
        num_train = random.randint(3, 6)
        
        def generate_examples(count):
            examples = []
            for _ in range(count):
                # Use retry to ensure we can generate valid grids
                input_grid = retry(
                    lambda: self.create_input(taskvars, {}),
                    lambda g: len(find_connected_objects(g, diagonal_connectivity=False, background=0, monochromatic=True)) == objects_num,
                    max_attempts=30
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
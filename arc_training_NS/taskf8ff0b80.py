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
    
    def create_single_object(self, grid: np.ndarray, color: int, target_size: int, max_attempts: int = 150) -> bool:
        """Create a single connected object of approximately target_size on the grid."""
        n = grid.shape[0]
        
        for attempt in range(max_attempts):
            # Find all empty cells first
            empty_cells = [(r, c) for r in range(n) for c in range(n) if grid[r, c] == 0]
            if len(empty_cells) < max(1, target_size // 3):
                return False  # Not enough space
            
            # Start with a random seed position from empty cells
            start_r, start_c = random.choice(empty_cells)
            
            # Build object by growing from seed
            object_cells = {(start_r, start_c)}
            grid[start_r, start_c] = color
            
            # Grow the object by adding neighboring cells
            growth_attempts = 0
            max_growth_attempts = target_size * 5
            
            while len(object_cells) < target_size and growth_attempts < max_growth_attempts:
                growth_attempts += 1
                
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
            
            # Accept if isolated and at least 33% of target size
            min_acceptable_size = max(1, target_size // 3)
            if is_isolated and len(object_cells) >= min_acceptable_size:
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
        
        # Adaptive sizing based on available space
        total_cells = n * n
        # Reserve space for isolation (assume ~40% efficiency with padding)
        usable_cells = int(total_cells * 0.4)
        avg_size = max(1, usable_cells // objects_num)
        
        # Create size distribution with guaranteed variety
        if objects_num <= 2:
            sizes = [avg_size, max(1, avg_size // 2)]
        else:
            # Geometric sequence for clear size differences
            max_size = min(avg_size * 2, 20)
            min_size = max(1, max_size // (objects_num + 1))
            
            sizes = []
            ratio = (max_size / min_size) ** (1 / (objects_num - 1))
            for i in range(objects_num):
                size = int(max_size / (ratio ** i))
                sizes.append(max(1, size))
        
        # Ensure distinct sizes
        sizes = list(dict.fromkeys(sizes))  # Remove duplicates preserving order
        while len(sizes) < objects_num:
            sizes.append(random.randint(max(1, min_size if 'min_size' in locals() else 1), avg_size))
            sizes = list(dict.fromkeys(sizes))
        sizes = sizes[:objects_num]
        
        # Place largest first (easier to place when grid is empty)
        creation_order = list(zip(colors, sorted(sizes, reverse=True)))
        
        # Create objects one by one
        for color, target_size in creation_order:
            # Adaptive retry with graceful degradation
            for attempt_size in [target_size, max(1, target_size // 2), 1]:
                success = self.create_single_object(grid, color, attempt_size)
                if success:
                    break
            if not success:
                raise ValueError(f"Grid too constrained: {n}x{n} with {objects_num} objects")
        
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
        # Number of objects per input grid
        objects_num = random.randint(2, 5)
        
        # Calculate minimum viable grid size for this many objects
        # Each object needs ~4 cells minimum (1 for object + 3 for isolation padding)
        # Add safety factor of 2x
        min_size = int(np.sqrt(objects_num * 8)) + 4
        n = random.randint(min_size, min(30, min_size + 15))
        
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
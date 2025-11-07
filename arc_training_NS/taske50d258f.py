from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects, GridObject, GridObjects
from input_library import random_cell_coloring, retry
import numpy as np
import random
from typing import Dict, Any, Tuple, List

class Taske50d258f(ARCTaskGenerator):
    
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['n']} x {vars['n']}.",
            "Each input grid contains several rectangular shapes, separated from each other by empty cells (0).",
            "The rectangular shapes in each grid are of different sizes.",
            "Each rectangle contains a random number of cells colored with {color('main_color')}, {color('target_color')}, and {color('other_color')}.",
            "In most rectangles, the majority of cells are colored with {color('main_color')}.",
            "The number of {color('target_color')} cells is different across rectangles within the same grid.",
            "In each input grid, the rectangle that has the highest number of cells colored with {color('target_color')} compared to the other rectangles, does not have the highest number of cells of color {color('other_color')} compared to other rectangles."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by identifying the rectangular shapes in the input grid.",
            "The rectangle that contains the highest number of cells colored with {color('target_color')} is selected.",
            "The output grid consists of only that rectangle, copied exactly as it appears in the input grid."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Initialize task variables
        taskvars = {
            'n': random.randint(10, 30),  # Grid size
            'main_color': random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9]),
            'target_color': random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9]),
            'other_color': random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9])
        }
        
        # Ensure all colors are different
        while taskvars['target_color'] == taskvars['main_color']:
            taskvars['target_color'] = random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9])
        
        while taskvars['other_color'] in [taskvars['main_color'], taskvars['target_color']]:
            taskvars['other_color'] = random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9])
        
        # Generate training and test examples
        num_train = random.randint(3, 6)
        train_examples = []
        test_examples = []
        
        for i in range(num_train + 1):
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            
            example = {'input': input_grid, 'output': output_grid}
            
            if i < num_train:
                train_examples.append(example)
            else:
                test_examples.append(example)
        
        return taskvars, {'train': train_examples, 'test': test_examples}
    
    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        n = taskvars['n']
        main_color = taskvars['main_color']
        target_color = taskvars['target_color']
        other_color = taskvars['other_color']
        
        def generate_valid_grid():
            grid = np.zeros((n, n), dtype=int)
            
            # Calculate max rectangle size based on grid size
            max_rect_size = max(4, n // 4)
            min_rect_size = 3
            
            # Calculate maximum number of rectangles that can fit
            # Estimate based on grid area and average rectangle size with margins
            margin = 2
            avg_rect_size = (min_rect_size + max_rect_size) // 2
            avg_rect_area = avg_rect_size * avg_rect_size
            rect_area_with_margin = (avg_rect_size + margin) * (avg_rect_size + margin)
            max_possible_rectangles = max(2, (n * n) // rect_area_with_margin)
            
            # Limit to a reasonable maximum (e.g., 8) to avoid overcrowding
            max_rectangles = min(8, max_possible_rectangles)
            
            # Create random number of rectangles (between 2 and the calculated maximum)
            num_rectangles = random.randint(2, max_rectangles)
            rectangles = []
            
            for rect_idx in range(num_rectangles):
                attempts = 0
                placed = False
                
                while attempts < 200 and not placed:
                    # Random rectangle size
                    width = random.randint(min_rect_size, max_rect_size)
                    height = random.randint(min_rect_size, max_rect_size)
                    
                    # Ensure different sizes (including rotations)
                    size = (width, height)
                    existing_sizes = [r['size'] for r in rectangles] + [(r['size'][1], r['size'][0]) for r in rectangles]
                    
                    if size in existing_sizes:
                        attempts += 1
                        continue
                    
                    # Try to place rectangle
                    if width > n or height > n:
                        attempts += 1
                        continue
                        
                    row = random.randint(0, n - height)
                    col = random.randint(0, n - width)
                    
                    # Check if placement is valid (no overlap, separated by empty cells)
                    valid = True
                    for r in rectangles:
                        # Check for overlap with margin
                        if not (row + height + margin <= r['row'] or 
                               row >= r['row'] + r['height'] + margin or
                               col + width + margin <= r['col'] or 
                               col >= r['col'] + r['width'] + margin):
                            valid = False
                            break
                    
                    if valid:
                        rectangles.append({
                            'row': row, 'col': col, 'size': size,
                            'width': width, 'height': height
                        })
                        placed = True
                    
                    attempts += 1
                
                if not placed:
                    return None  # Failed to place rectangle
            
            if len(rectangles) < 2:  # Need at least 2 rectangles
                return None
            
            # Fill rectangles with colors
            # Strategy: designate one rectangle as having highest target_color
            # Make sure it does NOT have the highest other_color
            winner_idx = random.randint(0, len(rectangles) - 1)
            target_counts = []
            other_counts = []
            
            for rect_idx, rect in enumerate(rectangles):
                row, col, width, height = rect['row'], rect['col'], rect['width'], rect['height']
                rect_area = width * height
                
                # Ensure we have enough cells for all three colors (at least 6 cells)
                if rect_area < 6:
                    return None
                
                if rect_idx == winner_idx:
                    # Winner rectangle: highest target_count, but NOT highest other_count
                    # Calculate target_count (aim for 30-45% but ensure valid range)
                    min_target = max(2, rect_area // 3)
                    max_target = max(min_target + 1, (rect_area * 45) // 100)
                    target_count = random.randint(min_target, max_target)
                    
                    # Calculate other_count (aim for 15-30% but ensure valid range)
                    remaining_after_target = rect_area - target_count
                    min_other = max(1, (rect_area * 15) // 100)
                    max_other = min((rect_area * 30) // 100, remaining_after_target - 1)
                    
                    if max_other < min_other:
                        max_other = remaining_after_target - 1
                        min_other = max(1, max_other // 2)
                    
                    if max_other < 1 or min_other < 1 or max_other < min_other:
                        return None
                        
                    other_count = random.randint(min_other, max_other)
                    main_count = rect_area - target_count - other_count
                    
                    if main_count < 1:
                        return None
                else:
                    # Non-winner rectangles: lower target_count
                    # Calculate target_count (aim for 10-25%)
                    max_target_loser = max(1, (rect_area * 25) // 100)
                    min_target_loser = max(1, (rect_area * 10) // 100)
                    if max_target_loser < min_target_loser:
                        min_target_loser = 1
                    target_count = random.randint(min_target_loser, max_target_loser)
                    
                    # Some non-winner rectangles should have higher other_count than winner
                    # With 50% probability, make this rectangle have high other_count
                    if random.random() < 0.5:
                        # High other_count (35-50% of area)
                        remaining_after_target = rect_area - target_count
                        min_other = max(1, (rect_area * 35) // 100)
                        max_other = min((rect_area * 50) // 100, remaining_after_target - 1)
                        
                        if max_other < min_other:
                            max_other = remaining_after_target - 1
                            min_other = max(1, remaining_after_target // 2)
                    else:
                        # Normal other_count (15-30% of area)
                        remaining_after_target = rect_area - target_count
                        min_other = max(1, (rect_area * 15) // 100)
                        max_other = min((rect_area * 30) // 100, remaining_after_target - 1)
                        
                        if max_other < min_other:
                            max_other = remaining_after_target - 1
                            min_other = 1
                    
                    if max_other < 1 or min_other < 1 or max_other < min_other:
                        return None
                        
                    other_count = random.randint(min_other, max_other)
                    remaining = rect_area - target_count - other_count
                    
                    if remaining < 1:
                        return None
                    main_count = remaining
                
                if main_count < 1 or target_count < 1 or other_count < 1:
                    return None
                
                target_counts.append(target_count)
                other_counts.append(other_count)
                
                # Create color list
                colors = [main_color] * main_count + [target_color] * target_count + [other_color] * other_count
                random.shuffle(colors)
                
                # Fill the rectangle
                idx = 0
                for r in range(row, row + height):
                    for c in range(col, col + width):
                        grid[r, c] = colors[idx]
                        idx += 1
            
            # Verify constraints
            max_target_count = max(target_counts)
            max_target_idx = target_counts.index(max_target_count)
            max_other_count = max(other_counts)
            max_other_idx = other_counts.index(max_other_count)
            
            # NEW CONSTRAINT: Rectangle with highest target_color does NOT have highest other_color
            if max_target_idx == max_other_idx:
                return None
            
            # Ensure all target_color counts are unique
            if len(set(target_counts)) != len(target_counts):
                return None
            
            return grid
        
        return retry(generate_valid_grid, lambda x: x is not None)
    
    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        target_color = taskvars['target_color']
        
        # Find all connected rectangular objects
        objects = find_connected_objects(grid, diagonal_connectivity=True, background=0, monochromatic=False)
        
        # Count target_color cells in each object
        max_target_count = 0
        selected_object = None
        
        for obj in objects:
            target_count = sum(1 for r, c, color in obj.cells if color == target_color)
            if target_count > max_target_count:
                max_target_count = target_count
                selected_object = obj
        
        if selected_object is None:
            return np.zeros((1, 1), dtype=int)
        
        # Extract the selected rectangle
        bounding_box = selected_object.bounding_box
        output_grid = grid[bounding_box[0], bounding_box[1]].copy()
        
        return output_grid
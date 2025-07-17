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
            "Each input grid contains 3 or 4 rectangular shapes, separated from each other by empty cells (0).",
            "The rectangular shapes in each grid are of different sizes.",
            "Each rectangle contains a random number of cells colored with {color('main_color')}, {color('target_color')}, and {color('other_color')}.",
            "In most rectangles, the majority of cells are colored with {color('main_color')}.",
            "The number of {color('target_color')} cells is different across rectangles within the same grid.",
            "In each input grid, the rectangle that has the highest number of cells colored with {color('target_color')} compared to the other rectangles, also has fewer cells of color {color('other_color')} than {color('target_color')}."
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
            
            # Create 3 or 4 rectangles
            num_rectangles = random.choice([3, 4])
            rectangles = []
            
            # Generate rectangles with different sizes
            for _ in range(num_rectangles):
                attempts = 0
                while attempts < 100:
                    # Random rectangle size (at least 2x2, different sizes)
                    width = random.randint(2, min(8, n//3))
                    height = random.randint(2, min(8, n//3))
                    
                    # Ensure different sizes
                    size = (width, height)
                    if size not in [r['size'] for r in rectangles]:
                        # Try to place rectangle
                        row = random.randint(0, n - height)
                        col = random.randint(0, n - width)
                        
                        # Check if placement is valid (no overlap, separated by empty cells)
                        valid = True
                        for r in rectangles:
                            # Check for overlap or adjacency
                            if not (row + height < r['row'] or row > r['row'] + r['size'][1] or
                                   col + width < r['col'] or col > r['col'] + r['size'][0]):
                                valid = False
                                break
                        
                        if valid:
                            rectangles.append({
                                'row': row, 'col': col, 'size': size,
                                'width': width, 'height': height
                            })
                            break
                    
                    attempts += 1
                
                if attempts >= 100:
                    return None  # Failed to place rectangle
            
            if len(rectangles) != num_rectangles:
                return None
            
            # Fill rectangles with colors
            target_counts = []
            other_counts = []
            
            for rect in rectangles:
                row, col, width, height = rect['row'], rect['col'], rect['width'], rect['height']
                
                # Fill rectangle area
                rect_area = width * height
                
                # Majority should be main_color
                main_count = random.randint(rect_area // 2, rect_area - 2)
                remaining = rect_area - main_count
                
                # Split remaining between target and other colors
                target_count = random.randint(1, remaining - 1)
                other_count = remaining - target_count
                
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
            
            # Check constraint: rectangle with highest target_color count has fewer other_color than target_color
            max_target_idx = target_counts.index(max(target_counts))
            if other_counts[max_target_idx] >= target_counts[max_target_idx]:
                return None
            
            # Ensure different target_color counts across rectangles
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


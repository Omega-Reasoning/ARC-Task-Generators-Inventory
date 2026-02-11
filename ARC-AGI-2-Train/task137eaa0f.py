from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects, GridObjects, GridObject
from input_library import retry
import numpy as np
import random
from typing import Dict, Any, Tuple, List, Set

class Task137eaa0fGenerator(ARCTaskGenerator):
    
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['grid_size']}x{vars['grid_size']}.",
            "Each grid contains several colored objects, where each object is made of 8-way connected cells and the remaining cells are empty (0).",
            "Each object contains exactly one {color('center_cell')} cell. The other cells in the object share the same color, which is unique for each object (i.e., different from the colors of other objects).",
            "The objects are shaped and sized such that they are 8-way connected to the {color('center_cell')} cell — i.e., top, bottom, left, right, top-left, top-right, bottom-left, and bottom-right.",
            "If two cells are placed above and below the {color('center_cell')} cell in one object, then those same positions cannot be used in other objects.",
            "The {color('center_cell')} cell always acts as the center of each object and ensures that the surrounding connected cells occupy unique positions across all objects. This means that if all {color('center_cell')} cells were placed on top of each other, no other colored cells would overlap.",
            "It is not necessary for all 8-way connections around the {color('center_cell')} cell to be filled.",
            "All objects must be completely separated from each other by empty (0) cells."
        ]
        
        transformation_reasoning_chain = [
            "The output grids are of size 3×3.",
            "They are constructed by identifying all colored objects containing a {color('center_cell')} cell and placing each object into a separate output grid.",
            "Each object is positioned such that its {color('center_cell')} cell is at position (1,1) in the output grid.",
            "Objects should be placed exactly as they appear in the input, preserving their original shape and orientation."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Initialize task variables
        taskvars = {
            'grid_size': random.randint(9, 30),
            'center_cell': random.randint(1, 9)
        }
        
        # Create examples using the default method but with custom gridvars
        train_examples = []
        test_examples = []
        
        for i in range(4):  # 3 train + 1 test
            # Create different gridvars for each example to ensure variety
            gridvars = {
                'num_objects': random.randint(2, 4),  # Keep reasonable for 3x3 output arrangement
                'object_colors': self._generate_unique_colors(taskvars['center_cell'], 5)
            }
            
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            example = {
                'input': input_grid,
                'output': output_grid
            }
            
            if i < 3:
                train_examples.append(example)
            else:
                test_examples.append(example)
        
        return taskvars, {
            'train': train_examples,
            'test': test_examples
        }
    
    def _generate_unique_colors(self, center_color: int, num_colors: int) -> List[int]:
        """Generate unique colors different from center_color"""
        available_colors = [c for c in range(1, 10) if c != center_color]
        return random.sample(available_colors, min(num_colors, len(available_colors)))
    
    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        grid_size = taskvars['grid_size']
        center_cell = taskvars['center_cell']
        num_objects = gridvars['num_objects']
        object_colors = gridvars['object_colors']
        
        def generate_valid_grid():
            grid = np.zeros((grid_size, grid_size), dtype=int)
            
            # Generate object patterns that don't overlap when center cells are aligned
            object_patterns = self._generate_non_overlapping_patterns(num_objects)
            
            # Place objects in the grid
            placed_objects = []
            max_attempts = 100
            
            for i, (pattern, color) in enumerate(zip(object_patterns, object_colors[:num_objects])):
                # Try to place this object
                for attempt in range(max_attempts):
                    # Choose random center position (with some margin from edges)
                    center_r = random.randint(2, grid_size - 3)
                    center_c = random.randint(2, grid_size - 3)
                    
                    # Check if we can place this object
                    can_place = True
                    positions_to_fill = [(center_r, center_c)]  # Center gets center_cell color
                    
                    for dr, dc in pattern:
                        new_r, new_c = center_r + dr, center_c + dc
                        if (0 <= new_r < grid_size and 0 <= new_c < grid_size):
                            positions_to_fill.append((new_r, new_c))
                        else:
                            can_place = False
                            break
                    
                    if not can_place:
                        continue
                    
                    # Check for conflicts with existing objects and separation
                    conflict = False
                    for r, c in positions_to_fill:
                        # Check if cell is already occupied
                        if grid[r, c] != 0:
                            conflict = True
                            break
                        
                        # Check for adjacency (objects must be separated by at least 1 empty cell)
                        for dr in [-1, 0, 1]:
                            for dc in [-1, 0, 1]:
                                if dr == 0 and dc == 0:
                                    continue
                                nr, nc = r + dr, c + dc
                                if (0 <= nr < grid_size and 0 <= nc < grid_size and 
                                    grid[nr, nc] != 0 and (nr, nc) not in positions_to_fill):
                                    conflict = True
                                    break
                            if conflict:
                                break
                        if conflict:
                            break
                    
                    if not conflict:
                        # Place the object
                        grid[center_r, center_c] = center_cell
                        for dr, dc in pattern:
                            new_r, new_c = center_r + dr, center_c + dc
                            if (0 <= new_r < grid_size and 0 <= new_c < grid_size):
                                grid[new_r, new_c] = color
                        placed_objects.append((center_r, center_c, pattern, color))
                        break
                else:
                    # Couldn't place this object, try a new grid
                    return None
            
            return grid if len(placed_objects) >= 2 else None
        
        return retry(generate_valid_grid, lambda x: x is not None)
    
    def _generate_non_overlapping_patterns(self, num_objects: int) -> List[List[Tuple[int, int]]]:
        """Generate patterns that don't overlap when centers are aligned"""
        # All possible 8-way connected positions around center (0,0)
        all_positions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        
        patterns = []
        used_positions = set()
        
        for _ in range(num_objects):
            # Available positions for this object
            available = [pos for pos in all_positions if pos not in used_positions]
            
            if not available:
                break
            
            # Create a pattern using 1-3 available positions
            pattern_size = random.randint(1, min(3, len(available)))
            pattern = random.sample(available, pattern_size)
            patterns.append(pattern)
            
            # Mark these positions as used
            used_positions.update(pattern)
        
        return patterns
    
    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        center_cell = taskvars['center_cell']
        
        # Find all objects containing center cells
        objects = find_connected_objects(grid, diagonal_connectivity=True, background=0, monochromatic=False)
        center_objects = objects.filter(lambda obj: center_cell in obj.colors)
        
        # For this task, I'll create a single 3x3 output that represents the "canonical" form
        # where all objects are overlaid with their centers aligned at (1,1)
        output_grid = np.zeros((3, 3), dtype=int)
        
        for obj in center_objects:
            # Find the center cell position
            center_pos = None
            for r, c, color in obj.cells:
                if color == center_cell:
                    center_pos = (r, c)
                    break
            
            if center_pos is None:
                continue
            
            # Place object with center at (1, 1)
            center_r, center_c = center_pos
            for r, c, color in obj.cells:
                output_r = r - center_r + 1
                output_c = c - center_c + 1
                
                # Only place if within 3x3 bounds
                if 0 <= output_r < 3 and 0 <= output_c < 3:
                    if output_grid[output_r, output_c] == 0:  # Don't overwrite existing
                        output_grid[output_r, output_c] = color
        
        return output_grid


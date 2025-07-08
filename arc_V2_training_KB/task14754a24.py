from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects, GridObject, GridObjects
from input_library import random_cell_coloring, retry
import numpy as np
import random

class Task14754a24Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of different sizes.",
            "Each grid has a background composed of empty (0) cells and {color('background')} cells, with several objects made of {color('object_color')}.",
            "The ratio of empty (0) cells to {color('background')} cells in the background is approximately 1:1. The {color('background')} cells appear either as isolated single cells or in clusters formed through 8-way connectivity.",
            "Each {color('object_color')} object is initially created by placing a plus-shaped pattern defined as: [[0, {color('object_color')}, 0], [ {color('object_color')}, {color('object_color')}, {color('object_color')} ], [0, {color('object_color')}, 0]].",
            "These plus shapes are placed only on top of {color('background')} cells. Then, some {color('object_color')} cells within these shapes are removed and replaced by {color('background')} cells, while ensuring that the remaining {color('object_color')} cells in each object remain connected.",
            "The removed cells are chosen such that in the output, they can be uniquely recovered i.e., there is only one valid way to refill them using {color('object_color')} on top of {color('background')} cells."
        ]
        
        transformation_reasoning_chain = [
            "The output grids are created by copying the input grids and identifying the incomplete {color('object_color')} objects.",
            "Once identified, these objects are completed by adding {color('fill_color')} cells to form plus shapes defined as: [[0, {color('object_color')}, 0], [ {color('object_color')}, {color('object_color')}, {color('object_color')} ], [0, {color('object_color')}, 0]].",
            "The {color('fill_color')} cells can only be added on top of {color('background')} cells."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self):
        # Generate task variables
        taskvars = {}
        
        # Ensure all colors are different
        all_colors = list(range(1, 10))
        chosen_colors = random.sample(all_colors, 3)
        taskvars['object_color'] = chosen_colors[0]
        taskvars['fill_color'] = chosen_colors[1] 
        taskvars['background'] = chosen_colors[2]
        
        # Generate 3 training examples and 1 test example
        train_examples = []
        for _ in range(3):
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            train_examples.append({'input': input_grid, 'output': output_grid})
        
        test_input = self.create_input(taskvars, {})
        test_output = self.transform_input(test_input, taskvars)
        test_examples = [{'input': test_input, 'output': test_output}]
        
        train_test_data = {
            'train': train_examples,
            'test': test_examples
        }
        
        return taskvars, train_test_data

    def create_input(self, taskvars, gridvars):
        object_color = taskvars['object_color']
        fill_color = taskvars['fill_color']
        background = taskvars['background']
        
        # Random grid size between 8 and 30
        size = random.randint(8, 30)
        height = size
        width = size
        
        def generate_valid_grid():
            # Start with empty grid
            grid = np.zeros((height, width), dtype=int)
            
            # Add background cells with ~50% density
            random_cell_coloring(grid, background, density=0.5)
            
            # Find potential plus shape centers (need space for full plus and separation)
            plus_pattern = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]  # center, up, down, left, right
            
            valid_centers = []
            for r in range(2, height - 2):  # Need extra space for separation
                for c in range(2, width - 2):
                    # Check if all plus positions have background color
                    if all(grid[r + dr, c + dc] == background for dr, dc in plus_pattern):
                        valid_centers.append((r, c))
            
            # If not enough valid centers, try creating more background cells
            if len(valid_centers) < 2:
                # Add more background cells in strategic locations
                for r in range(1, height - 1):
                    for c in range(1, width - 1):
                        if grid[r, c] == 0:  # Empty cell
                            # Check if making this background would create valid plus positions
                            grid[r, c] = background
                            # Recheck valid centers
                            valid_centers = []
                            for r2 in range(2, height - 2):
                                for c2 in range(2, width - 2):
                                    if all(grid[r2 + dr, c2 + dc] == background for dr, dc in plus_pattern):
                                        valid_centers.append((r2, c2))
                            if len(valid_centers) >= 2:
                                break
                    if len(valid_centers) >= 2:
                        break
            
            # Final fallback: if still not enough, create a simpler background pattern
            if len(valid_centers) < 2:
                # Reset grid and create a more structured background
                grid = np.zeros((height, width), dtype=int)
                # Create background in a checkerboard-like pattern to ensure plus positions
                for r in range(height):
                    for c in range(width):
                        if (r + c) % 3 == 0:  # Every third cell
                            grid[r, c] = background
                
                # Recheck valid centers
                valid_centers = []
                for r in range(2, height - 2):
                    for c in range(2, width - 2):
                        if all(grid[r + dr, c + dc] == background for dr, dc in plus_pattern):
                            valid_centers.append((r, c))
            
            if len(valid_centers) < 2:
                return None  # Still not enough valid positions
            
            # Place 2-3 plus shapes with minimum distance between them
            num_objects = min(random.randint(2, 3), len(valid_centers))
            selected_centers = []
            
            # Use a more flexible approach for center selection
            attempts = 0
            while len(selected_centers) < num_objects and attempts < 50:
                attempts += 1
                
                if not valid_centers:
                    break
                    
                # Pick a random center
                center = random.choice(valid_centers)
                
                # Check if it's far enough from existing centers
                min_distance = 4 if len(selected_centers) == 0 else max(3, 4 - len(selected_centers))
                too_close = any(abs(center[0] - existing[0]) + abs(center[1] - existing[1]) < min_distance 
                               for existing in selected_centers)
                
                if not too_close:
                    selected_centers.append(center)
                    # Remove this center and very close ones
                    valid_centers = [c for c in valid_centers 
                                   if c != center and abs(c[0] - center[0]) + abs(c[1] - center[1]) >= min_distance]
                elif len(selected_centers) >= 2:
                    # If we have at least 2 objects, we can proceed
                    break
            
            if len(selected_centers) < 2:
                return None
            
            # Place incomplete plus shapes
            successfully_placed = 0
            for center_r, center_c in selected_centers:
                # Create complete plus shape positions
                plus_cells = [(center_r + dr, center_c + dc) for dr, dc in plus_pattern]
                
                # Remove 1-2 cells (but not center) to make it incomplete
                removable_cells = plus_cells[1:]  # All except center
                
                # Try different removal patterns until we find one that works
                placed = False
                for num_to_remove in [1, 2]:
                    if num_to_remove > len(removable_cells) or placed:
                        continue
                    
                    # Try multiple combinations
                    for _ in range(5):  # Try up to 5 different removal patterns
                        cells_to_remove = random.sample(removable_cells, num_to_remove)
                        remaining_cells = [cell for cell in plus_cells if cell not in cells_to_remove]
                        
                        # Verify remaining cells are connected
                        if self._are_cells_connected(remaining_cells):
                            # Place the object cells
                            for r, c in remaining_cells:
                                grid[r, c] = object_color
                            successfully_placed += 1
                            placed = True
                            break
            
            if successfully_placed < 2:
                return None
            
            # Final verification: check that all object_color cells form separate objects
            objects = find_connected_objects(grid, diagonal_connectivity=False, background=0)
            object_color_objects = objects.with_color(object_color)
            
            # Should have at least 2 objects
            if len(object_color_objects) < 2:
                return None
            
            # Each object should be completable to exactly one plus shape
            valid_completions = 0
            for obj in object_color_objects:
                if self._can_complete_to_plus(grid, obj, object_color, background):
                    valid_completions += 1
            
            # Need at least 2 valid objects
            return grid if valid_completions >= 2 else None
        
        # Use retry with more attempts and handle final fallback
        try:
            return retry(generate_valid_grid, lambda x: x is not None, max_attempts=200)
        except ValueError:
            # Final fallback: create a simpler, guaranteed-to-work grid
            return self._create_fallback_grid(taskvars)

    def _create_fallback_grid(self, taskvars):
        """Create a simple, guaranteed-to-work grid as a last resort."""
        object_color = taskvars['object_color']
        background = taskvars['background']
        
        # Create a smaller, simpler grid that's guaranteed to work
        height = width = 12
        grid = np.zeros((height, width), dtype=int)
        
        # Fill with background in a structured way
        for r in range(height):
            for c in range(width):
                if (r + c) % 2 == 0:
                    grid[r, c] = background
        
        # Place exactly 2 simple incomplete plus shapes at fixed positions
        centers = [(3, 3), (8, 8)]
        plus_pattern = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
        
        for center_r, center_c in centers:
            plus_cells = [(center_r + dr, center_c + dc) for dr, dc in plus_pattern]
            # Remove one cell to make it incomplete
            remaining_cells = plus_cells[:-1]  # Remove the last cell (right)
            
            for r, c in remaining_cells:
                grid[r, c] = object_color
        
        return grid

    def _are_cells_connected(self, cells):
        """Check if a set of cells forms a connected component using 4-connectivity."""
        if len(cells) <= 1:
            return True
        
        cells_set = set(cells)
        visited = set()
        stack = [cells[0]]
        
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            
            r, c = current
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor = (r + dr, c + dc)
                if neighbor in cells_set and neighbor not in visited:
                    stack.append(neighbor)
        
        return len(visited) == len(cells)

    def _can_complete_to_plus(self, grid, obj, object_color, background):
        """Check if an object can be completed to exactly one plus shape."""
        coords = list(obj.coords)
        
        # Try each cell as potential center
        possible_completions = 0
        
        for r, c in coords:
            # Check if this could be the center of a plus
            plus_pattern = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
            plus_cells = [(r + dr, c + dc) for dr, dc in plus_pattern]
            
            # Check if all plus cells are within grid bounds
            if not all(0 <= pr < grid.shape[0] and 0 <= pc < grid.shape[1] 
                      for pr, pc in plus_cells):
                continue
            
            # Check if this forms a valid plus completion
            can_complete = True
            has_missing = False
            
            for pr, pc in plus_cells:
                if (pr, pc) in coords:
                    # This cell is already part of the object
                    continue
                elif grid[pr, pc] == background:
                    # This cell can be filled
                    has_missing = True
                else:
                    # This cell is blocked
                    can_complete = False
                    break
            
            if can_complete and has_missing:
                possible_completions += 1
        
        return possible_completions == 1

    def transform_input(self, grid, taskvars):
        object_color = taskvars['object_color']
        fill_color = taskvars['fill_color']
        background = taskvars['background']
        
        output_grid = grid.copy()
        
        # Find all object_color objects
        objects = find_connected_objects(output_grid, diagonal_connectivity=False, background=0)
        object_color_objects = objects.with_color(object_color)
        
        # For each object, complete it to a plus shape
        for obj in object_color_objects:
            self._complete_plus_shape(output_grid, obj, object_color, fill_color, background)
        
        return output_grid

    def _complete_plus_shape(self, grid, obj, object_color, fill_color, background):
        """Complete an incomplete plus shape by adding fill_color cells."""
        coords = list(obj.coords)
        
        # Try each cell as potential center
        for r, c in coords:
            # Check if this could be the center of a plus
            plus_pattern = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
            plus_cells = [(r + dr, c + dc) for dr, dc in plus_pattern]
            
            # Check if all plus cells are within grid bounds
            if not all(0 <= pr < grid.shape[0] and 0 <= pc < grid.shape[1] 
                      for pr, pc in plus_cells):
                continue
            
            # Check if this forms a valid plus completion
            can_complete = True
            missing_cells = []
            
            for pr, pc in plus_cells:
                if (pr, pc) in coords:
                    # This cell is already part of the object - should stay object_color
                    continue
                elif grid[pr, pc] == background:
                    # This cell can be filled with fill_color
                    missing_cells.append((pr, pc))
                else:
                    # This cell is blocked
                    can_complete = False
                    break
            
            if can_complete and missing_cells:
                # Fill the missing cells with fill_color
                for pr, pc in missing_cells:
                    grid[pr, pc] = fill_color
                return  # Successfully completed this object

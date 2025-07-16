from arc_task_generator import ARCTaskGenerator
from transformation_library import GridObject, GridObjects, find_connected_objects
from input_library import create_object, retry, random_cell_coloring
import numpy as np
import random

class ExtenderExpansionTaskGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "The input grids are squares of different sizes.",
            "Each input grid contains two key components.",
            "A 2x2 block with each cell having a distinct color, and A connected extender pattern that sprouts from exactly one of the cells in that 2×2 block.",
            "The extending pattern always uses the same color as the cell it is attached to.",
            "Multiple such blocks (with their extenders) may appear in a single grid. No two 2×2 blocks share the exact same arrangement of colors. No two extenders share the exact same shape or orientation within the same grid."
        ]
        
        transformation_reasoning_chain = [
            "The output grid retains the same dimensions as the input grid.",
            "For each extender in the input, Identify the originating cell in the 2×2 block and its color.In the output, reproduce the extender shape in all four cardinal directions (up, down, left, right), each copy rooted at that same cell.Maintain the exact shape and color of the original extender.",
            "Each block+extender group is handled independently, their expanded extenders may overlap if their ranges intersect."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_extender_pattern(self, direction, color):
        """Create an extender pattern in a specific direction"""
        patterns = {
            'up': [
                [(0, 0), (-1, 0), (-2, 0)],  # Vertical line up
                [(0, 0), (-1, 0), (-1, 1)],  # L-shape up-right
                [(0, 0), (-1, 0), (-1, -1), (-1, 1)]  # T-shape up
            ],
            'down': [
                [(0, 0), (1, 0), (2, 0)],  # Vertical line down
                [(0, 0), (1, 0), (1, 1)],  # L-shape down-right
                [(0, 0), (1, 0), (1, -1), (1, 1)]  # T-shape down
            ],
            'left': [
                [(0, 0), (0, -1), (0, -2)],  # Horizontal line left
                [(0, 0), (0, -1), (-1, -1)],  # L-shape left-up
                [(0, 0), (0, -1), (-1, -1), (1, -1)]  # T-shape left
            ],
            'right': [
                [(0, 0), (0, 1), (0, 2)],  # Horizontal line right
                [(0, 0), (0, 1), (-1, 1)],  # L-shape right-up
                [(0, 0), (0, 1), (-1, 1), (1, 1)]  # T-shape right
            ]
        }
        
        pattern = random.choice(patterns[direction])
        return {(r, c, color) for r, c in pattern}
    
    def get_extender_bounds(self, extender_cells, origin_r, origin_c):
        """Calculate the bounds of an extender pattern"""
        min_r = min(r for r, c, _ in extender_cells)
        max_r = max(r for r, c, _ in extender_cells)
        min_c = min(c for c, _, _ in extender_cells)
        max_c = max(c for c, _, _ in extender_cells)
        return (min_r + origin_r, max_r + origin_r, min_c + origin_c, max_c + origin_c)
    
    def create_input(self, taskvars, gridvars):
        """Create an input grid with 2x2 blocks and extenders"""
        grid_size = gridvars['grid_size']
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Number of block+extender groups
        num_groups = random.randint(1, min(3, grid_size // 8))
        
        # Keep track of used color arrangements and extender shapes
        used_color_arrangements = set()
        used_extender_shapes = set()
        placed_groups = []
        
        for _ in range(num_groups):
            # Try to place a group
            max_attempts = 50
            placed = False
            
            for attempt in range(max_attempts):
                # Generate unique color arrangement for 2x2 block
                colors = random.sample(range(1, 10), 4)
                color_arrangement = tuple(colors)
                
                if color_arrangement in used_color_arrangements:
                    continue
                
                # Choose position for 2x2 block (leave space for extenders)
                margin = max(3, grid_size // 6)
                if grid_size <= margin * 2:
                    margin = 1
                
                block_r = random.randint(margin, grid_size - margin - 2)
                block_c = random.randint(margin, grid_size - margin - 2)
                
                # Choose which cell the extender comes from
                extender_origin_idx = random.randint(0, 3)
                extender_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
                origin_offset_r, origin_offset_c = extender_positions[extender_origin_idx]
                extender_color = colors[extender_origin_idx]
                
                # Create extender pattern
                extender_direction = random.choice(['up', 'down', 'left', 'right'])
                extender_cells = self.create_extender_pattern(extender_direction, extender_color)
                
                # Check if this extender shape was already used
                extender_shape = (extender_direction, frozenset((r, c) for r, c, _ in extender_cells))
                if extender_shape in used_extender_shapes:
                    continue
                
                # Calculate actual extender bounds
                origin_r = block_r + origin_offset_r
                origin_c = block_c + origin_offset_c
                extender_bounds = self.get_extender_bounds(extender_cells, origin_r, origin_c)
                
                # Check if extender fits in grid
                if (extender_bounds[0] < 0 or extender_bounds[1] >= grid_size or 
                    extender_bounds[2] < 0 or extender_bounds[3] >= grid_size):
                    continue
                
                # Check for overlaps with existing groups
                overlap = False
                for existing_group in placed_groups:
                    existing_bounds = existing_group['bounds']
                    # Check if bounding boxes overlap with padding
                    padding = 2
                    if not (extender_bounds[1] + padding < existing_bounds[0] or 
                           extender_bounds[0] - padding > existing_bounds[1] or
                           extender_bounds[3] + padding < existing_bounds[2] or 
                           extender_bounds[2] - padding > existing_bounds[3]):
                        overlap = True
                        break
                
                if overlap:
                    continue
                
                # Place the 2x2 block
                grid[block_r:block_r+2, block_c:block_c+2] = np.array(colors).reshape(2, 2)
                
                # Place the extender
                for r, c, color in extender_cells:
                    actual_r = origin_r + r
                    actual_c = origin_c + c
                    if 0 <= actual_r < grid_size and 0 <= actual_c < grid_size:
                        grid[actual_r, actual_c] = color
                
                # Record this placement
                used_color_arrangements.add(color_arrangement)
                used_extender_shapes.add(extender_shape)
                placed_groups.append({
                    'block_pos': (block_r, block_c),
                    'origin_pos': (origin_r, origin_c),
                    'extender_cells': extender_cells,
                    'bounds': extender_bounds,
                    'color': extender_color
                })
                
                placed = True
                break
            
            if not placed:
                break
        
        return grid
    
    def transform_input(self, input_grid, taskvars):
        """Transform input to output by expanding extenders in all 4 directions"""
        output_grid = input_grid.copy()
        
        # Find all connected objects
        objects = find_connected_objects(input_grid, diagonal_connectivity=False, background=0)
        
        # Process each object to find extenders
        for obj in objects:
            coords = obj.coords
            
            # Check if this object is connected to a 2x2 block
            for r in range(input_grid.shape[0] - 1):
                for c in range(input_grid.shape[1] - 1):
                    block_coords = {(r, c), (r, c+1), (r+1, c), (r+1, c+1)}
                    
                    # Check if this object contains exactly one cell from this 2x2 block
                    intersection = coords.intersection(block_coords)
                    if len(intersection) == 1:
                        # Check if all 4 cells of the block are non-zero and have different colors
                        block_colors = [input_grid[br, bc] for br, bc in block_coords]
                        if all(color != 0 for color in block_colors) and len(set(block_colors)) == 4:
                            # This is an extender attached to a 2x2 block
                            origin_r, origin_c = list(intersection)[0]
                            extender_cells = coords - {(origin_r, origin_c)}
                            
                            if extender_cells:  # Make sure there are extender cells
                                color = input_grid[origin_r, origin_c]
                                
                                # Convert extender to relative coordinates
                                relative_extender = [(r - origin_r, c - origin_c) 
                                                   for r, c in extender_cells]
                                
                                # Create extenders in all 4 directions
                                directions = [
                                    (-1, 0),  # up
                                    (1, 0),   # down
                                    (0, -1),  # left
                                    (0, 1)    # right
                                ]
                                
                                for dir_r, dir_c in directions:
                                    for rel_r, rel_c in relative_extender:
                                        # Transform relative coordinates based on direction
                                        if dir_r == -1:  # up
                                            new_r = origin_r - abs(rel_r) if rel_r < 0 else origin_r - rel_r
                                            new_c = origin_c + rel_c
                                        elif dir_r == 1:  # down
                                            new_r = origin_r + abs(rel_r) if rel_r > 0 else origin_r - rel_r
                                            new_c = origin_c + rel_c
                                        elif dir_c == -1:  # left
                                            new_r = origin_r + rel_r
                                            new_c = origin_c - abs(rel_c) if rel_c < 0 else origin_c - rel_c
                                        elif dir_c == 1:  # right
                                            new_r = origin_r + rel_r
                                            new_c = origin_c + abs(rel_c) if rel_c > 0 else origin_c - rel_c
                                        
                                        # Place if within bounds
                                        if (0 <= new_r < input_grid.shape[0] and 
                                            0 <= new_c < input_grid.shape[1]):
                                            output_grid[new_r, new_c] = color
        
        return output_grid
    
    def create_grids(self):
        """Create train and test grids with consistent variables."""
        # Store task variables (empty for this task)
        taskvars = {}
        
        # Generate 3-5 training examples
        num_train_examples = random.randint(3, 5)
        train_examples = []
        
        # Generate grid sizes
        min_size = 10
        max_size = 20
        all_sizes = [random.randint(min_size, max_size) for _ in range(num_train_examples + 1)]
        
        # Create training examples
        for i in range(num_train_examples):
            gridvars = {'grid_size': all_sizes[i]}
            
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Create test example
        test_gridvars = {'grid_size': all_sizes[-1]}
        test_input = self.create_input(taskvars, test_gridvars)
        test_output = self.transform_input(test_input, taskvars)
        
        test_examples = [{
            'input': test_input,
            'output': test_output
        }]
        
        return taskvars, {
            'train': train_examples,
            'test': test_examples
        }

# Test the generator
if __name__ == "__main__":
    generator = ExtenderExpansionTaskGenerator()
    taskvars, train_test_data = generator.create_grids()
    
    # Visualize using the parent class method
    print("Training examples:")
    for i, example in enumerate(train_test_data['train']):
        print(f"\nTrain {i+1}:")
        print("Input:")
        print(example['input'])
        print("Output:")
        print(example['output'])
    
    print("\nTest example:")
    print("Input:")
    print(train_test_data['test'][0]['input'])
    print("Output:")
    print(train_test_data['test'][0]['output'])
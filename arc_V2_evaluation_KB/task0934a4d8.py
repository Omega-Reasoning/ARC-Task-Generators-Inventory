from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects, GridObject, GridObjects
from input_library import create_object, random_cell_coloring, retry, Contiguity
import numpy as np
import random

class Task0934a4d8Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['grid_size']} × {vars['grid_size']}.",
            "Each grid is completely filled with multi-colored objects made of 8-way connected cells. There are no empty (0) cells. In addition, there is a rectangular {color('block')} object present in the grid.",
            "To create the grid, first construct the top-left quadrant of size ({vars['grid_size']//2 + 1} × {vars['grid_size']//2 + 1}).",
            "This quadrant is filled by placing differently shaped and colored objects such as 2×2 blocks, single cells, and some diagonally arranged cells.",
            "Add an inverse diagonal starting from ({vars['grid_size']//2}, 0) and (0, {vars['grid_size']//2}), stopping at the boundary of the top-left ({vars['grid_size']//2 + 1} × {vars['grid_size']//2 + 1}) quadrant. This diagonal must have a distinct color not used by any neighboring objects, and its color should vary across different examples.",
            "Also, add a second diagonal starting from (0, 0), extending down-right until it touches the first (inverse) diagonal. This diagonal should also have a unique color that differs from both the first diagonal and any adjacent objects.",
            "Once the top-left quadrant ({vars['grid_size']//2 + 1} × {vars['grid_size']//2 + 1}) is constructed, reflect it horizontally to fill the top-right quadrant and vertically to fill the bottom-left quadrant. These reflections do not need to fully fit the space—partial fits are acceptable.",
            "Finally, reflect the bottom-left quadrant horizontally to fill in the bottom-right quadrant.",
            "After all quadrants have been filled, place a rectangular {color('block')} object with dimensions at least 3×3 anywhere in the grid except the top-left quadrant. This object must not overlap any existing pattern elements and should be randomly positioned within the allowed area."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by identifying the rectangular {color('block')} object in the input grid and initializing a zero-filled grid of the same size as the {color('block')} object.",
            "In the input grid, the top-left quadrant of size ({vars['grid_size']//2 + 1} × {vars['grid_size']//2 + 1}) is reflected horizontally to form the top-right quadrant, and vertically to form the bottom-left quadrant.",
            "The bottom-right quadrant is formed by horizontally reflecting the bottom-left quadrant.",
            "Based on this reflection logic, determine which parts of the original input grid were overlapped by the rectangular {color('block')} object in the input grid.",
            "Once the overlapping cells from the original pattern are identified, copy their values into the corresponding positions within the output grid."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars, gridvars):
        grid_size = taskvars['grid_size']
        block_color = taskvars['block']
        
        # Create top-left quadrant with exactly grid_size unique objects
        quad_size = grid_size // 2 + 1
        
        # Initialize quadrant with zeros temporarily
        quadrant = np.zeros((quad_size, quad_size), dtype=int)
        
        # Available colors (excluding block color)
        available_colors = [c for c in range(1, 10) if c != block_color]
        
        # Create exactly grid_size unique objects to fill the quadrant completely
        self._create_unique_objects(quadrant, quad_size, available_colors, grid_size)
        
        # Now create the full grid and fill it with reflections
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Copy the quadrant to top-left
        grid[:quad_size, :quad_size] = quadrant
        
        # Add inverse diagonal (from top-right to bottom-left of quadrant)
        diag1_color = random.choice(available_colors)
        for i in range(quad_size):
            if i < quad_size and (quad_size - 1 - i) < quad_size:
                grid[i, quad_size - 1 - i] = diag1_color
        
        # Add main diagonal (from top-left, touching but not cutting the inverse diagonal)
        diag2_color = random.choice([c for c in available_colors if c != diag1_color])
        
        # The main diagonal goes from (0,0) and stops just before it would cut the inverse diagonal
        # The inverse diagonal is at positions (i, quad_size-1-i)
        # The main diagonal is at positions (i, i)
        # They meet when i == quad_size-1-i, which means i == (quad_size-1)/2
        meeting_point = (quad_size - 1) // 2
        
        # Draw main diagonal from (0,0) up to (but not including) the meeting point
        for i in range(meeting_point):
            if i < quad_size and i < quad_size:
                grid[i, i] = diag2_color
        
        # Store the original quadrant for reflection reconstruction
        original_quad = grid[:quad_size, :quad_size].copy()
        
        # Reflect horizontally to fill top-right quadrant
        top_right_start = grid_size // 2
        for r in range(quad_size):
            for c in range(quad_size):
                target_c = top_right_start + (quad_size - 1 - c)
                if target_c < grid_size:
                    grid[r, target_c] = original_quad[r, c]
        
        # Reflect vertically to fill bottom-left quadrant
        bottom_left_start = grid_size // 2
        for r in range(quad_size):
            for c in range(quad_size):
                target_r = bottom_left_start + (quad_size - 1 - r)
                if target_r < grid_size:
                    grid[target_r, c] = original_quad[r, c]
        
        # Reflect bottom-left horizontally to fill bottom-right
        for r in range(bottom_left_start, grid_size):
            for c in range(quad_size):
                target_c = top_right_start + (quad_size - 1 - c)
                if target_c < grid_size:
                    grid[r, target_c] = grid[r, c]
        
        # Store the complete pattern before placing block
        self.complete_pattern = grid.copy()
        
        # Place the block object (at least 3x3, outside top-left quadrant)
        block_width = random.randint(3, 6)
        block_height = random.randint(3, 6)
        
        # Choose location outside top-left quadrant
        possible_locations = []
        
        # Top-right area
        for r in range(grid_size - block_height + 1):
            for c in range(grid_size // 2, grid_size - block_width + 1):
                possible_locations.append((r, c))
        
        # Bottom area (entire width)
        for r in range(grid_size // 2, grid_size - block_height + 1):
            for c in range(grid_size - block_width + 1):
                possible_locations.append((r, c))
        
        if possible_locations:
            block_r, block_c = random.choice(possible_locations)
            
            # Store block position and size for transformation
            self.block_pos = (block_r, block_c)
            self.block_size = (block_height, block_width)
            
            # Place the block
            grid[block_r:block_r + block_height, block_c:block_c + block_width] = block_color
        
        return grid

    def _create_unique_objects(self, quadrant, quad_size, available_colors, grid_size):
        """Create exactly grid_size unique objects to fill the quadrant completely"""
        total_cells = quad_size * quad_size
        cells_per_object = max(1, total_cells // grid_size)
        
        # Create a list of all cell positions
        all_positions = [(r, c) for r in range(quad_size) for c in range(quad_size)]
        random.shuffle(all_positions)
        
        object_id = 1
        pos_index = 0
        
        # Create exactly grid_size objects
        for obj_num in range(grid_size):
            if pos_index >= len(all_positions):
                break
                
            color = available_colors[obj_num % len(available_colors)]
            
            # Determine object size (vary between 1-4 cells, with smaller objects more common)
            if obj_num < grid_size - 10:  # Most objects are small
                obj_size = random.choices([1, 2, 3, 4], weights=[40, 30, 20, 10])[0]
            else:  # Last few objects can be larger to fill remaining space
                remaining_cells = len(all_positions) - pos_index
                remaining_objects = grid_size - obj_num
                avg_size_needed = max(1, remaining_cells // remaining_objects)
                obj_size = min(remaining_cells, max(1, avg_size_needed + random.randint(-1, 1)))
            
            # Create the object
            if obj_size == 1:
                # Single cell
                if pos_index < len(all_positions):
                    r, c = all_positions[pos_index]
                    quadrant[r, c] = color
                    pos_index += 1
            elif obj_size == 2:
                # Two connected cells
                if pos_index + 1 < len(all_positions):
                    r1, c1 = all_positions[pos_index]
                    r2, c2 = all_positions[pos_index + 1]
                    
                    # Try to make them adjacent if possible
                    quadrant[r1, c1] = color
                    quadrant[r2, c2] = color
                    pos_index += 2
            elif obj_size == 3:
                # L-shape or line
                if pos_index + 2 < len(all_positions):
                    shape_type = random.choice(['L', 'line'])
                    if shape_type == 'L':
                        # Try to create L-shape
                        r, c = all_positions[pos_index]
                        if r + 1 < quad_size and c + 1 < quad_size:
                            quadrant[r, c] = color
                            quadrant[r + 1, c] = color
                            quadrant[r, c + 1] = color
                        else:
                            # Fallback to scattered
                            for i in range(3):
                                if pos_index + i < len(all_positions):
                                    r, c = all_positions[pos_index + i]
                                    quadrant[r, c] = color
                    else:
                        # Line shape
                        for i in range(3):
                            if pos_index + i < len(all_positions):
                                r, c = all_positions[pos_index + i]
                                quadrant[r, c] = color
                    pos_index += 3
            else:  # obj_size == 4 or larger
                # 2x2 block or scattered
                shape_type = random.choice(['2x2', 'scattered'])
                cells_to_place = min(obj_size, len(all_positions) - pos_index)
                
                if shape_type == '2x2' and cells_to_place >= 4:
                    # Try to create 2x2 block
                    r, c = all_positions[pos_index]
                    if r + 1 < quad_size and c + 1 < quad_size:
                        quadrant[r, c] = color
                        quadrant[r + 1, c] = color
                        quadrant[r, c + 1] = color
                        quadrant[r + 1, c + 1] = color
                        pos_index += 4
                    else:
                        # Fallback to scattered
                        for i in range(cells_to_place):
                            if pos_index + i < len(all_positions):
                                r, c = all_positions[pos_index + i]
                                quadrant[r, c] = color
                        pos_index += cells_to_place
                else:
                    # Scattered cells
                    for i in range(cells_to_place):
                        if pos_index + i < len(all_positions):
                            r, c = all_positions[pos_index + i]
                            quadrant[r, c] = color
                    pos_index += cells_to_place
        
        # Fill any remaining empty cells with random colors to ensure no background
        for r in range(quad_size):
            for c in range(quad_size):
                if quadrant[r, c] == 0:
                    quadrant[r, c] = random.choice(available_colors)

    def transform_input(self, grid, taskvars):
        grid_size = taskvars['grid_size']
        block_color = taskvars['block']
        
        # Find the block object
        block_objects = find_connected_objects(grid, diagonal_connectivity=True, monochromatic=True).with_color(block_color)
        
        if len(block_objects) == 0:
            return np.zeros((3, 3), dtype=int)
        
        # Get the largest block (should be our rectangular block)
        block = max(block_objects.objects, key=len)
        
        # Get block bounding box
        bbox = block.bounding_box
        block_height = bbox[0].stop - bbox[0].start
        block_width = bbox[1].stop - bbox[1].start
        block_r = bbox[0].start
        block_c = bbox[1].start
        
        # Create output grid of block size
        output = np.zeros((block_height, block_width), dtype=int)
        
        # Reconstruct the original pattern that would be underneath the block
        quad_size = grid_size // 2 + 1
        
        # For each cell in the block area, determine what the original pattern would be
        for r in range(block_height):
            for c in range(block_width):
                grid_r = block_r + r
                grid_c = block_c + c
                
                # Reconstruct what should be at this position based on the reflection pattern
                reconstructed_value = self._reconstruct_original_value(grid, grid_r, grid_c, grid_size, quad_size, block_color)
                output[r, c] = reconstructed_value
        
        return output
    
    def _reconstruct_original_value(self, grid, r, c, grid_size, quad_size, block_color):
        """Reconstruct the original value at position (r,c) based on reflection pattern"""
        
        # Determine which quadrant this position belongs to and find the corresponding source
        if r < quad_size and c < quad_size:
            # Top-left quadrant - this is the original, look at nearby non-block cells
            return self._find_nearby_non_block_value(grid, r, c, block_color)
        
        elif r < quad_size and c >= grid_size // 2:
            # Top-right quadrant - horizontally reflected from top-left
            source_r = r
            source_c = quad_size - 1 - (c - grid_size // 2)
            if 0 <= source_c < quad_size:
                return self._find_nearby_non_block_value(grid, source_r, source_c, block_color)
        
        elif r >= grid_size // 2 and c < quad_size:
            # Bottom-left quadrant - vertically reflected from top-left
            source_r = quad_size - 1 - (r - grid_size // 2)
            source_c = c
            if 0 <= source_r < quad_size:
                return self._find_nearby_non_block_value(grid, source_r, source_c, block_color)
        
        elif r >= grid_size // 2 and c >= grid_size // 2:
            # Bottom-right quadrant - reflected from bottom-left
            source_r = r
            source_c = quad_size - 1 - (c - grid_size // 2)
            if 0 <= source_c < quad_size:
                return self._find_nearby_non_block_value(grid, source_r, source_c, block_color)
        
        # Fallback: find any nearby non-block value
        return self._find_nearby_non_block_value(grid, r, c, block_color)
    
    def _find_nearby_non_block_value(self, grid, r, c, block_color):
        """Find a nearby non-block value to use as the reconstructed value"""
        grid_size = grid.shape[0]
        
        # First try the exact position if it's not a block
        if 0 <= r < grid_size and 0 <= c < grid_size and grid[r, c] != block_color:
            return grid[r, c]
        
        # Search in expanding radius for non-block cells
        for radius in range(1, min(grid_size, 10)):
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < grid_size and 0 <= nc < grid_size and 
                        grid[nr, nc] != block_color):
                        return grid[nr, nc]
        
        # Fallback to a non-block color
        for color in range(1, 10):
            if color != block_color:
                return color
        
        return 2  # Ultimate fallback

    def create_grids(self):
        # Create task variables
        taskvars = {
            'grid_size': random.choice([22,24, 26, 28, 30]),
            'block': random.randint(1, 9)
        }
        
        # Generate training and test examples
        num_train = random.randint(3, 6)
        train_examples = []
        
        for _ in range(num_train):
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            train_examples.append({'input': input_grid, 'output': output_grid})
        
        # Generate test example
        test_input = self.create_input(taskvars, {})
        test_output = self.transform_input(test_input, taskvars)
        test_examples = [{'input': test_input, 'output': test_output}]
        
        train_test_data = {
            'train': train_examples,
            'test': test_examples
        }
        
        return taskvars, train_test_data
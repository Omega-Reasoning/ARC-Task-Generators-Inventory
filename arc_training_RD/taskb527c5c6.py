from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects, GridObjects, GridObject
from input_library import retry, random_cell_coloring
import numpy as np
import random

class PointerBlockExtensionGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "The input grids are squares of different sizes.",
            "Each grid has two rectangular elongated blocks, one vertically and the other horizontally. Both blocks are of {color('block_color')} color.",
            "Each block has a small pointer mark (a single cell) of {color('pointer_color')} within the block, not at the corners. The pointer is within the boundary and acts like a marking. These blocks and their pointers may be used for alignment, connection, or movement clues."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is a copy of the input grid.",
            "Each block uses its pointer (anchor cell) as a clue for movement. The pointer extends in the opposite direction of the blocks orientation.",
            "If the pointer is on the horizontal block, it moves vertically; if on the vertical block, it moves horizontally.",
            "The pointer extends along its path using a unique path color of {color('pointer_color')}, while the block itself also extends in the same direction, using its original color.",
            "The block extends by several cells equal to the number of rows (for horizontal blocks) or columns (for vertical blocks) in the grid."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def color_name(self, color: int) -> str:
        color_map = {
            0: "black",
            1: "blue", 
            2: "red",
            3: "green",
            4: "yellow",
            5: "gray",
            6: "magenta",
            7: "orange",
            8: "cyan",
            9: "brown"
        }
        return color_map.get(color, f"color_{color}")

    def will_blocks_overlap_after_extension(self, h_block_area, v_block_area, h_pointer_pos, v_pointer_pos, h_pointer_dir, v_pointer_dir, grid_size):
        """Check if blocks will overlap after extension"""
        h_start_row, h_start_col, h_thickness, h_length = h_block_area
        v_start_row, v_start_col, v_length, v_thickness = v_block_area
        
        # Calculate extended areas
        h_extended_area = set()
        v_extended_area = set()
        
        # Original horizontal block
        for r in range(h_start_row, h_start_row + h_thickness):
            for c in range(h_start_col, h_start_col + h_length):
                h_extended_area.add((r, c))
        
        # Extended horizontal block
        for i in range(1, grid_size + 1):
            new_row = h_start_row + (i * h_pointer_dir)
            if 0 <= new_row < grid_size:
                for c in range(h_start_col, h_start_col + h_length):
                    h_extended_area.add((new_row, c))
        
        # Original vertical block
        for r in range(v_start_row, v_start_row + v_length):
            for c in range(v_start_col, v_start_col + v_thickness):
                v_extended_area.add((r, c))
        
        # Extended vertical block
        for i in range(1, grid_size + 1):
            new_col = v_start_col + (i * v_pointer_dir)
            if 0 <= new_col < grid_size:
                for r in range(v_start_row, v_start_row + v_length):
                    v_extended_area.add((r, new_col))
        
        # Check for overlap
        return bool(h_extended_area.intersection(v_extended_area))

    def create_input(self, taskvars):
        block_color = taskvars["block_color"]
        pointer_color = taskvars["pointer_color"]
        # path_color = taskvars["path_color"]
        
        # Try multiple times to create a valid grid
        for attempt in range(50):
            # Random grid size between 10 and 25
            size = random.randint(10, 25)
            grid = np.zeros((size, size), dtype=int)
            
            try:
                # Create horizontal block (uniformly spaced from edges)
                min_block = 3
                max_block = min(size // 3, 8)  # Limit max block size to avoid overlap issues
                if max_block < min_block:
                    continue
                
                h_block_length = random.randint(min_block, max_block)
                h_block_thickness = random.randint(1, 2)
                
                # Position horizontal block with uniform spacing
                margin = 3
                if size - h_block_thickness - 2*margin <= 0:
                    continue
                h_start_row = random.randint(margin, size - h_block_thickness - margin)
                
                start = margin
                end = size - h_block_length - margin
                if end < start:
                    continue
                h_start_col = random.randint(start, end)
                
                # Draw horizontal block
                grid[h_start_row:h_start_row + h_block_thickness, 
                     h_start_col:h_start_col + h_block_length] = block_color
                
                # Add pointer WITHIN horizontal block (not at corners)
                if h_block_length <= 2:  # Need at least 3 cells to avoid corners
                    continue
                
                # Choose position within the block
                pointer_col = random.randint(h_start_col + 1, h_start_col + h_block_length - 2)
                pointer_row = random.randint(h_start_row, h_start_row + h_block_thickness - 1)
                
                grid[pointer_row, pointer_col] = pointer_color
                h_pointer_pos = (pointer_row, pointer_col)
                
                # Determine extension direction - can go up or down
                h_pointer_dir = random.choice([-1, 1])
                
                # Create vertical block (uniformly spaced from edges)
                v_block_length = random.randint(min_block, max_block)
                v_block_thickness = random.randint(1, 2)
                
                # Position vertical block with uniform spacing, avoiding horizontal block
                overlap_found = True
                for attempt_v in range(50):
                    if size - v_block_length - 2*margin <= 0 or size - v_block_thickness - 2*margin <= 0:
                        break
                    v_start_row = random.randint(margin, size - v_block_length - margin)
                    v_start_col = random.randint(margin, size - v_block_thickness - margin)
                    
                    # Check if it overlaps with horizontal block
                    h_block_area_set = set()
                    for r in range(h_start_row, h_start_row + h_block_thickness):
                        for c in range(h_start_col, h_start_col + h_block_length):
                            h_block_area_set.add((r, c))
                    
                    v_block_area_set = set()
                    for r in range(v_start_row, v_start_row + v_block_length):
                        for c in range(v_start_col, v_start_col + v_block_thickness):
                            v_block_area_set.add((r, c))
                    
                    if not h_block_area_set.intersection(v_block_area_set):
                        overlap_found = False
                        break
                
                if overlap_found:
                    continue
                
                # Draw vertical block
                grid[v_start_row:v_start_row + v_block_length, 
                     v_start_col:v_start_col + v_block_thickness] = block_color
                
                # Add pointer WITHIN vertical block (not at corners)
                if v_block_length <= 2:  # Need at least 3 cells to avoid corners
                    continue
                
                # Choose position within the block
                pointer_row = random.randint(v_start_row + 1, v_start_row + v_block_length - 2)
                pointer_col = random.randint(v_start_col, v_start_col + v_block_thickness - 1)
                
                grid[pointer_row, pointer_col] = pointer_color
                v_pointer_pos = (pointer_row, pointer_col)
                
                # Determine extension direction - can go left or right
                v_pointer_dir = random.choice([-1, 1])
                
                # Check if blocks will overlap after extension
                h_block_area = (h_start_row, h_start_col, h_block_thickness, h_block_length)
                v_block_area = (v_start_row, v_start_col, v_block_length, v_block_thickness)
                
                if self.will_blocks_overlap_after_extension(h_block_area, v_block_area, h_pointer_pos, v_pointer_pos, h_pointer_dir, v_pointer_dir, size):
                    continue  # Try again with different positions/directions
                
                # Store metadata for transformation
                self.h_block_area = h_block_area
                self.v_block_area = v_block_area
                self.h_pointer_pos = h_pointer_pos
                self.v_pointer_pos = v_pointer_pos
                self.h_pointer_dir = h_pointer_dir
                self.v_pointer_dir = v_pointer_dir
                self.block_color = block_color
                self.pointer_color = pointer_color
                # self.path_color = path_color
                self.grid_size = size
                
                return grid
                
            except Exception:
                continue
        
        # If we couldn't create a valid grid after many attempts, raise an error
        raise ValueError("Could not generate a valid input grid after multiple attempts")

    def transform_input(self, grid, taskvars):
        block_color = taskvars["block_color"]
        pointer_color = taskvars["pointer_color"]
        # path_color = taskvars["path_color"]
        
        output_grid = grid.copy()
        
        # Extension distance equals grid size
        extension_distance = self.grid_size
        
        # Extend horizontal block and its pointer
        h_start_row, h_start_col, h_thickness, h_length = self.h_block_area
        h_pointer_row, h_pointer_col = self.h_pointer_pos
        
        # Extend horizontal block vertically
        for i in range(1, extension_distance + 1):
            new_row = h_start_row + (i * self.h_pointer_dir)
            if 0 <= new_row < self.grid_size:
                for c in range(h_start_col, h_start_col + h_length):
                    if 0 <= c < self.grid_size:
                        output_grid[new_row, c] = block_color
        
        # Extend pointer path vertically
        for i in range(1, extension_distance + 1):
            new_row = h_pointer_row + (i * self.h_pointer_dir)
            if 0 <= new_row < self.grid_size:
                output_grid[new_row, h_pointer_col] = pointer_color
        
        # Extend vertical block and its pointer
        v_start_row, v_start_col, v_length, v_thickness = self.v_block_area
        v_pointer_row, v_pointer_col = self.v_pointer_pos
        
        # Extend vertical block horizontally
        for i in range(1, extension_distance + 1):
            new_col = v_start_col + (i * self.v_pointer_dir)
            if 0 <= new_col < self.grid_size:
                for r in range(v_start_row, v_start_row + v_length):
                    if 0 <= r < self.grid_size:
                        output_grid[r, new_col] = block_color
        
        # Extend pointer path horizontally
        for i in range(1, extension_distance + 1):
            new_col = v_pointer_col + (i * self.v_pointer_dir)
            if 0 <= new_col < self.grid_size:
                output_grid[v_pointer_row, new_col] = pointer_color
        
        return output_grid

    def create_grids(self):
        num_train_pairs = random.randint(3, 5)
        train_pairs = []
        
        # Randomly choose colors
        block_color = random.randint(1, 9)
        pointer_color = random.choice([c for c in range(1, 10) if c != block_color])
        # path_color = random.choice([c for c in range(1, 10) if c != block_color and c != pointer_color])
        
        taskvars = {
            "block_color": block_color,
            "pointer_color": pointer_color,
            # "path_color": path_color
        }
        
        # Helper for reasoning chain formatting
        def color_fmt(key):
            color_id = taskvars[key]
            return f"{self.color_name(color_id)} ({color_id})"

        # Replace color placeholders in reasoning chains
        self.input_reasoning_chain = [
            chain.replace("{color('block_color')}", color_fmt('block_color'))
                 .replace("{color('pointer_color')}", color_fmt('pointer_color'))
                #  .replace("{color('path_color')}", color_fmt('path_color'))
            for chain in self.input_reasoning_chain
        ]
        
        self.transformation_reasoning_chain = [
            chain.replace("{color('block_color')}", color_fmt('block_color'))
                 .replace("{color('pointer_color')}", color_fmt('pointer_color'))
                #  .replace("{color('path_color')}", color_fmt('path_color'))
            for chain in self.transformation_reasoning_chain
        ]
        
        # Generate training pairs
        for i in range(num_train_pairs):
            input_grid = self.create_input(taskvars)
            output_grid = self.transform_input(input_grid, taskvars)
            train_pairs.append(GridPair(input=input_grid, output=output_grid))
        
        # Generate test pair
        test_input = self.create_input(taskvars)
        test_output = self.transform_input(test_input, taskvars)
        test_pairs = [GridPair(input=test_input, output=test_output)]
        
        return taskvars, TrainTestData(train=train_pairs, test=test_pairs)


# Test the generator
if __name__ == "__main__":
    generator = PointerBlockExtensionGenerator()
    taskvars, train_test_data = generator.create_grids()
    ARCTaskGenerator.visualize_train_test_data(train_test_data)
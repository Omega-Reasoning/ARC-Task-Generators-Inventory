from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from transformation_library import find_connected_objects
from input_library import retry, Contiguity

class Taskbdad9b1fGenerator(ARCTaskGenerator):
    def __init__(self):
        # Use placeholders in the reasoning chains
        input_reasoning_chain = [
            "Input grids are square grids of same sizes.", 
            "Each grid contains :", 
            "Two-celled horizontal blocks with {color('h_color')} color.",
            "Two-celled vertical blocks with {color('v_color')} color.", 
            "These horizontal and vertical blocks are positioned such that when extended in their respective directions, they intersect at one or more grid cells."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is copied from the input grid.",
            "Each horizontal and vertical block is then extended fully:",
            "Horizontal blocks extend left and right across the row, maintaining the {color('h_color')} color.",
            "Vertical blocks extend up and down across the column, maintaining the {color('v_color')} color.",
            "Wherever a horizontal and vertical extension intersect, the cell at the intersection is assigned a distinct {color('i_color')} color."
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
    
    def create_input(self, taskvars, gridvars):
        # Random grid size (preferring even numbers and more than 5 rows/columns)
        size = random.choice([6, 8, 10, 12, 14, 16])
        
        # Colors for horizontal, vertical blocks and intersections
        h_color = taskvars["h_color"]
        v_color = taskvars["v_color"]
        i_color = taskvars["i_color"]
        
        # Initialize empty grid
        grid = np.zeros((size, size), dtype=int)
        
        # Create one horizontal block
        h_row = random.randint(0, size-1)
        h_col_start = random.randint(0, size-2)
        grid[h_row, h_col_start:h_col_start+2] = h_color
        
        # Create one vertical block
        # Decide whether the blocks should intersect in the input (randomly)
        should_intersect_in_input = random.choice([True, False])
        
        if should_intersect_in_input:
            # Place vertical block to overlap with horizontal block
            v_col = random.choice([h_col_start, h_col_start+1])
            v_row_start = random.randint(0, size-2)
            
            # Make sure at least one cell of the vertical block doesn't overlap with the horizontal
            # (we need two distinct blocks, not just one L-shape)
            if v_row_start <= h_row and v_row_start + 1 >= h_row:
                if h_row > 0 and v_row_start > 0:
                    v_row_start = h_row - 1
                elif h_row < size - 2:
                    v_row_start = h_row + 1
                else:
                    v_row_start = max(0, h_row - 1)
            
            grid[v_row_start:v_row_start+2, v_col] = v_color
            
            # If they overlap in the input, we need to correct that cell
            if (h_row >= v_row_start and h_row < v_row_start+2 and 
                v_col >= h_col_start and v_col < h_col_start+2):
                # We'll prioritize the vertical block at the intersection
                grid[h_row, v_col] = v_color
        else:
            # Place vertical block to not overlap with horizontal block, but to create intersection when extended
            v_col = random.randint(h_col_start, h_col_start+1)  # Ensure intersection when extended
            v_row_start = random.randint(0, size-2)
            
            # Ensure they don't overlap in the input
            while (h_row >= v_row_start and h_row < v_row_start+2):
                v_row_start = random.randint(0, size-2)
                
            grid[v_row_start:v_row_start+2, v_col] = v_color
        
        return grid
    
    def transform_input(self, grid, taskvars):
        # Extract colors from taskvars
        h_color = taskvars["h_color"]
        v_color = taskvars["v_color"]
        i_color = taskvars["i_color"]
        
        # Create a copy of the input grid
        output_grid = np.copy(grid)
        
        # Find the horizontal block
        horizontal_blocks = find_connected_objects(grid, diagonal_connectivity=False, background=0)
        horizontal_blocks = horizontal_blocks.filter(lambda obj: 
                                                   h_color in obj.colors and 
                                                   obj.width == 2 and
                                                   obj.height == 1)
        
        # Find the vertical block
        vertical_blocks = find_connected_objects(grid, diagonal_connectivity=False, background=0)
        vertical_blocks = vertical_blocks.filter(lambda obj: 
                                               v_color in obj.colors and 
                                               obj.width == 1 and
                                               obj.height == 2)
        
        # Extend horizontal block
        for h_block in horizontal_blocks:
            row = list(h_block.cells)[0][0]  # Get the row of the block
            output_grid[row, :] = h_color
        
        # Extend vertical block
        for v_block in vertical_blocks:
            col = list(v_block.cells)[0][1]  # Get the column of the block
            output_grid[:, col] = v_color
        
        # Mark intersections
        for h_block in horizontal_blocks:
            row = list(h_block.cells)[0][0]
            for v_block in vertical_blocks:
                col = list(v_block.cells)[0][1]
                output_grid[row, col] = i_color
        
        return output_grid
    
    def create_grids(self):
        # Define task variables with distinct colors
        h_color = random.randint(1, 9)
        v_color = random.choice([c for c in range(1, 10) if c != h_color])
        i_color = random.choice([c for c in range(1, 10) if c != h_color and c != v_color])
        
        taskvars = {
            "h_color": h_color,
            "v_color": v_color,
            "i_color": i_color
        }
        
        # Helper for reasoning chain formatting
        def color_fmt(key):
            color_id = taskvars[key]
            return f"{self.color_name(color_id)} ({color_id})"
        
        # Replace placeholders in reasoning chains
        self.input_reasoning_chain = [
            chain.replace("{color('h_color')}", color_fmt('h_color'))
                 .replace("{color('v_color')}", color_fmt('v_color'))
                 .replace("{color('i_color')}", color_fmt('i_color'))
            for chain in self.input_reasoning_chain
        ]
        self.transformation_reasoning_chain = [
            chain.replace("{color('h_color')}", color_fmt('h_color'))
                 .replace("{color('v_color')}", color_fmt('v_color'))
                 .replace("{color('i_color')}", color_fmt('i_color'))
            for chain in self.transformation_reasoning_chain
        ]
        
        nr_train = random.randint(3, 5)
        nr_test = random.randint(1, 2)

        # Use the same pattern as the reference code to ensure diversity
        def generate_examples(n, is_test=False):
            examples = []
            attempts = 0
            max_attempts = 100 
            
            while len(examples) < n and attempts < max_attempts:
                input_grid = self.create_input(taskvars, {})
                output_grid = self.transform_input(input_grid, taskvars)
                
                # For test examples, always add the first one and ensure second is different
                if is_test and len(examples) == 0:
                    examples.append({'input': input_grid, 'output': output_grid})
                    attempts += 1
                    continue
                    
                # Check if input equals output
                if not np.array_equal(input_grid, output_grid):
                    examples.append({'input': input_grid, 'output': output_grid})
                else:
                    # For training set, allow at most one equal case
                    equal_cases = sum(1 for ex in examples if np.array_equal(ex['input'], ex['output']))
                    if equal_cases == 0:
                        examples.append({'input': input_grid, 'output': output_grid})
                        
                attempts += 1
                
            if attempts >= max_attempts:
                raise RuntimeError("Could not generate enough diverse examples")
                
            return examples

        train_examples = generate_examples(nr_train)
        test_examples = generate_examples(nr_test, is_test=True)
        
        # Convert to the expected format
        train_pairs = [GridPair(input=ex['input'], output=ex['output']) for ex in train_examples]
        test_pairs = [GridPair(input=ex['input'], output=ex['output']) for ex in test_examples]
        
        return taskvars, TrainTestData(train=train_pairs, test=test_pairs)


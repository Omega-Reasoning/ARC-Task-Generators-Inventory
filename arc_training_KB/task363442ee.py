from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from transformation_library import find_connected_objects
from input_library import create_object, retry, Contiguity

class Task363442eeGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['rows']}x{vars['cols']}..",
            "They contain a completely filled fourth column with {color('cell_color1')} colored cells and a 3x3 block made of multi-colored (1-9) cells, positioned at the top-left corner of the grid.",
            "Several {color('cell_color2')} cells are placed on the right side of the {color('cell_color1')} column.",
            "Each {color('cell_color2')} cell must have at least two consecutive empty (0) cells connected to it in all 8-way directions."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid and pasting the 3x3 multi-colored block from the top-left corner onto all {color('cell_color2')} cells.",
            "The center of the 3x3 multi-colored block must always be aligned with the {color('cell_color2')} cell it is pasted on, while the rest of the grid remains unchanged."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        rows, cols = taskvars['rows'], taskvars['cols']
        cell_color1, cell_color2 = taskvars['cell_color1'], taskvars['cell_color2']
        grid = np.zeros((rows, cols), dtype=int)

        # Fill the fourth column with cell_color1
        grid[:, 3] = cell_color1

        # Create a 3x3 multi-colored block in the top-left corner
        block_colors = list(set(range(1, 10)) - {cell_color1, cell_color2})
        block = np.random.choice(block_colors, (3, 3))
        grid[:3, :3] = block

        # Place several cell_color2 cells ensuring the required spacing
        # By default create_input builds a base grid with the block and column filled.
        # Placement of the {cell_color2} cells is handled in create_grids so that
        # we can control counts and ensure the test count differs from the train counts.
        # Here we only store the generated block so other code can reuse it if needed.
        gridvars.setdefault('base_block', grid[:3, :3].copy())
        return grid
    
    def transform_input(self, grid: np.ndarray, gridvars: dict) -> np.ndarray:
        cell_color2_positions = gridvars['cell_color2_positions']
        transformed_grid = grid.copy()
        block = grid[:3, :3]  # Extract the original block

        for r, c in cell_color2_positions:
            transformed_grid[r-1:r+2, c-1:c+2] = block

        return transformed_grid
    
    def create_grids(self) -> tuple:
        rows, cols = random.randint(8, 30), random.randint(11, 30)
        
        valid_colors = list(range(1, 10))
        cell_color1, cell_color2 = random.sample(valid_colors, 2)
        
        taskvars = {
            'rows': rows,
            'cols': cols,
            'cell_color1': cell_color1,
            'cell_color2': cell_color2
        }
        
        # Build a shared base grid with only the fourth column filled. The 3x3 block
        # will be re-randomized per example so train and test do not all share the
        # exact same block colors.
        base_grid = np.zeros((rows, cols), dtype=int)
        base_grid[:, 3] = cell_color1
        block_colors = list(set(range(1, 10)) - {cell_color1, cell_color2})

        def new_block(used_blocks: set):
            # Re-roll the 3x3 block until it differs from previously used ones.
            for _ in range(10):
                candidate = np.random.choice(block_colors, (3, 3))
                key = tuple(candidate.flatten())
                if key not in used_blocks:
                    used_blocks.add(key)
                    return candidate
            # Fallback if variety is exhausted (unlikely): return last candidate.
            used_blocks.add(tuple(candidate.flatten()))
            return candidate

        used_blocks = set()

        # Candidate positions to the right of the fourth column where a 5x5 neighborhood fits
        available_positions = [(r, c) for r in range(2, rows-2) for c in range(6, cols-2)]

        def place_positions_for_count(base, count):
            # Try to place exactly `count` cells respecting the 5x5 empty neighborhood rule.
            grid = base.copy()
            positions = []
            candidates = available_positions.copy()
            random.shuffle(candidates)
            for r, c in candidates:
                if (grid[r-2:r+3, c-2:c+3] == 0).all():
                    grid[r, c] = cell_color2
                    positions.append((r, c))
                    if len(positions) >= count:
                        break
            # If we couldn't place enough, return what we have (caller may handle)
            return grid, positions

        # Choose a test count that will be strictly different from all train counts.
        allowed_counts = list(range(3, 7))  # 3..6
        n_train = random.randint(3, 4)
        test_count = random.choice(allowed_counts)

        # For train examples, pick counts that are NOT equal to test_count
        train_allowed = [c for c in allowed_counts if c != test_count]
        train_examples = []
        train_counts = []
        for _ in range(n_train):
            count = random.choice(train_allowed)
            train_counts.append(count)
            gridvars = {}
            base_with_block = base_grid.copy()
            base_with_block[:3, :3] = new_block(used_blocks)
            inp_grid, positions = place_positions_for_count(base_with_block, count)
            gridvars['cell_color2_positions'] = positions
            # If placement failed to reach requested count, try again with more shuffles up to a few times
            attempts = 0
            while len(positions) < count and attempts < 3:
                inp_grid, positions = place_positions_for_count(base_with_block, count)
                gridvars['cell_color2_positions'] = positions
                attempts += 1
            # finalize example (even if positions < count it's still a valid grid)
            out_grid = inp_grid.copy()
            train_examples.append({'input': inp_grid, 'output': self.transform_input(inp_grid, gridvars)})

        # Create test example with a count different from all train counts
        gridvars = {}
        base_with_block = base_grid.copy()
        base_with_block[:3, :3] = new_block(used_blocks)
        test_grid, test_positions = place_positions_for_count(base_with_block, test_count)
        gridvars['cell_color2_positions'] = test_positions
        attempts = 0
        while len(test_positions) < test_count and attempts < 3:
            test_grid, test_positions = place_positions_for_count(base_with_block, test_count)
            gridvars['cell_color2_positions'] = test_positions
            attempts += 1
        test_output = self.transform_input(test_grid, gridvars)
        
        train_test_data = {
            'train': train_examples,
            'test': [{'input': test_grid, 'output': test_output}]
        }
        
        return taskvars, train_test_data
    


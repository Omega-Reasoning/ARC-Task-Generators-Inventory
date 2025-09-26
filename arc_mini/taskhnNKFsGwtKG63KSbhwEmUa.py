from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects
from input_library import retry
import numpy as np
import random
from typing import Dict, Any, Tuple, List

class TaskhnNKFsGwtKG63KSbhwEmUaGenerator(ARCTaskGenerator):

    def __init__(self):
        # 1) Input reasoning chain (copy from input)
        input_reasoning_chain = [
            "Input grids are of size {vars['grid_size']}x{vars['grid_size']}.",
            "Each input grid contains the main diagonal (from top-left to bottom-right) completely filled with cells of {color('object_color1')} and {color('object_color2')} color and several non-diagonal cells of {color('object_color1')} and {color('object_color2')} color.",
            "The non-diagonal cells are always below the main diagonal.",
            "The remaining cells are empty (0)."
        ]

        # 2) Transformation reasoning chain (copy from input)
        transformation_reasoning_chain = [
            "To construct the output grid, copy the input grid and iterate through each non-diagonal cell.",
            "If a non-diagonal cell is 4-way connected to a diagonal cell of the same color, mirror it across the main diagonal (line of reflection) in the output grid; otherwise, leave it unchanged."
            
        ]

        # 3) super call
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self,
                     taskvars,
                     gridvars) -> np.ndarray:
        """
        Create an input grid according to the input reasoning chain 
        given the task (taskvars) and grid-specific (gridvars) variables.
        """
        size = taskvars['grid_size']
        color1 = taskvars['object_color1']
        color2 = taskvars['object_color2']
        ensure_color1_4way_conn = gridvars.get('ensure_color1_4way_conn', False)
        
        # Start with an empty grid
        grid = np.zeros((size, size), dtype=int)
        
        # 1. Fill the main diagonal with random picks of color1 or color2
        #    so that each cell i,i is either color1 or color2
        for i in range(size):
            grid[i, i] = random.choice([color1, color2])
        
        # 2. Insert 3-5 non-diagonal cells, all below the diagonal (r>c).
        #    They must not be 4-way adjacent to each other.
        #    We also want at least one to be color1 and 4-way connected to 
        #    a diagonal cell with color1 if ensure_color1_4way_conn == True.
        num_non_diag = random.randint(3, 5)
        
        # We'll store chosen below-diagonal cells (r,c,color).
        non_diagonal_cells = []
        
        attempts = 0
        max_attempts = 100
        while len(non_diagonal_cells) < num_non_diag and attempts < max_attempts:
            attempts += 1
            r = random.randint(1, size-1)
            c = random.randint(0, r-1)  # ensures r>c (below main diagonal)
            # Decide color. We'll choose either color1 or color2 randomly.
            # But if we need to ensure adjacency for color1, we might prefer color1 with adjacency.
            chosen_color = random.choice([color1, color2])
            
            # Check if already occupied or diagonal or out-of-bounds
            if grid[r, c] != 0:
                continue
            
            # Check no adjacency with previously chosen non-diagonal cells
            # (We want them not 4-way adjacent to each other.)
            def not_adj_to_others(rr, cc):
                for (rr2, cc2, _) in non_diagonal_cells:
                    if abs(rr - rr2) + abs(cc - cc2) == 1:
                        return False
                return True
            
            if not not_adj_to_others(r, c):
                continue
            
            # If we must ensure that at least one color1 cell is 4-way connected to a diagonal cell 
            # with color1, let's try to place such a cell:
            if ensure_color1_4way_conn and chosen_color == color1:
                # We can attempt to place it near a diagonal cell that also has color1
                # i.e. (k+1, k) if grid[k, k] == color1
                # We'll see if we can forcibly ensure adjacency:
                # If r-1 == c, that means it's right below the diagonal cell at (c, c).
                # We'll check if the diagonal cell has color1. If so, it's good.
                if (r - 1 == c) and (grid[c, c] == color1):
                    # This is good for adjacency
                    pass
                else:
                    # If we can't easily place it, skip or override chosen_color
                    continue
            
            # If we are not specifically forcing adjacency, or adjacency conditions are met, place it.
            grid[r, c] = chosen_color
            non_diagonal_cells.append((r, c, chosen_color))
        
        # If we didn't succeed in placing enough cells:
        if len(non_diagonal_cells) < num_non_diag:
            # fallback: just place fewer if repeated attempts fail
            pass
        
        return grid

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        """
        Transform the input grid according to the transformation reasoning chain:
        1) Copy the input
        2) For each non-diagonal cell that is 4-way connected to a diagonal cell
           of the same color, mirror it across the diagonal.
        """
        size = taskvars['grid_size']
        out_grid = grid.copy()
        
        # We'll iterate over all cells below the diagonal (r>c).
        for r in range(size):
            for c in range(size):
                if r > c and grid[r, c] != 0:
                    color = grid[r, c]
                    # Check 4-way connectivity with a diagonal cell of the same color.
                    # A diagonal cell is at (k, k). For them to be 4-way connected,
                    # either (r,c) must be adjacent to (k,k). That implies
                    # |r - k|+|c - k|=1. Hence r-k=1,c-k=0 or r-k=0,c-k=1 or negative eq.
                    # But we know r>c, so the simplest adjacency is likely (k+1, k).
                    
                    # We'll check if there's any diagonal (k,k) with the same color 
                    # such that (r,c) is 4way-adj to (k,k).
                    # We can do a small loop or direct check:
                    connected = False
                    for k in range(size):
                        if grid[k, k] == color:
                            # Up, down, left, right
                            # (k-1, k), (k+1, k), (k, k-1), (k, k+1)
                            adj_positions = [
                                (k-1, k),
                                (k+1, k),
                                (k, k-1),
                                (k, k+1)
                            ]
                            if (r, c) in adj_positions:
                                connected = True
                                break
                    
                    if connected:
                        # mirror across diagonal => out_grid[c, r] = color
                        out_grid[c, r] = color
        
        return out_grid

    def create_grids(self) -> (dict, TrainTestData):
        """
        Initialise task variables used in the templates (vars[key]) and create 
        train/test data grids. We must produce a dictionary of taskvars plus the 
        train/test data.
        
        We want 3-4 training grids and 1 test grid. We ensure at least one training 
        grid has a color1 cell 4way connected to a diagonal cell with color1.
        """
        # Randomly select grid size, color1, color2
        grid_size = random.randint(7, 30)
        object_color1 = random.randint(1, 9)
        # ensure color2 is distinct
        while True:
            object_color2 = random.randint(1, 9)
            if object_color2 != object_color1:
                break
        
        taskvars = {
            'grid_size': grid_size,
            'object_color1': object_color1,
            'object_color2': object_color2
        }
        
        # Randomly choose how many training examples: 3 or 4
        nr_train_examples = random.choice([3, 4])
        nr_test_examples = 1
        
        # We want to ensure at least one training example has color1 adjacency.
        # We'll generate the first training example with ensure_color1_4way_conn=True
        train_examples = []
        
        # first example ensures adjacency
        gridvars_first = {'ensure_color1_4way_conn': True}
        input_grid_1 = self.create_input(taskvars, gridvars_first)
        output_grid_1 = self.transform_input(input_grid_1, taskvars)
        train_examples.append({'input': input_grid_1, 'output': output_grid_1})
        
        # the rest training examples can be default
        for _ in range(nr_train_examples - 1):
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            train_examples.append({'input': input_grid, 'output': output_grid})
        
        # Now create test example (usually we do not want to reveal the output, 
        # but for completeness, we'll fill it in as well).
        # The test example can be arbitrary.
        input_grid_test = self.create_input(taskvars, {})
        output_grid_test = self.transform_input(input_grid_test, taskvars)

        test_examples = [{'input': input_grid_test, 'output': output_grid_test}]
        
        train_test_data = {
            'train': train_examples,
            'test': test_examples
        }
        
        return taskvars, train_test_data


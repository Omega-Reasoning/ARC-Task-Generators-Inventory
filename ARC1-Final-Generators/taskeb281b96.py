from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

class Taskeb281b96Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size rows x {vars['columns']}.",
            "A wave-like pattern is filled using one random color. The pattern is defined as follows: In the bottom row, all cells in columns where the index j satisfies j % 4 == 0 are colored. In the top row, the column located midway between each pair of consecutive colored columns in the bottom row is identified, and the corresponding cell is colored. In all intermediate rows (excluding the top and bottom), all cells in columns with an odd index are colored."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed through a series of vertical flips and concatenations on the input grid.",
            "The input grid, excluding its bottom row, is first flipped vertically.",
            "The vertically flipped grid is then appended to the bottom of the original input grid.",
            "The resulting grid (original + flipped), excluding its new bottom row, is flipped vertically again.",
            "This second flipped version is then appended to the bottom of the current grid."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars, gridvars):
        rows = gridvars['rows']
        columns = taskvars['columns']
        color = gridvars['color']
        
        # Create empty grid
        grid = np.zeros((rows, columns), dtype=int)
        
        # Bottom row: color columns where j % 4 == 0
        for j in range(columns):
            if j % 4 == 0:
                grid[rows-1, j] = color
        
        # Top row: color midpoint columns between consecutive colored columns in bottom row
        if rows > 1:
            bottom_colored_cols = [j for j in range(columns) if j % 4 == 0]
            for i in range(len(bottom_colored_cols) - 1):
                midpoint = (bottom_colored_cols[i] + bottom_colored_cols[i+1]) // 2
                grid[0, midpoint] = color
        
        # Intermediate rows: color all odd-indexed columns
        if rows > 2:
            for row in range(1, rows-1):
                for j in range(columns):
                    if j % 2 == 1:
                        grid[row, j] = color
        
        return grid

    def transform_input(self, grid, taskvars):
        # Step 1: Take input grid excluding bottom row and flip vertically
        without_bottom = grid[:-1, :]
        flipped1 = np.flipud(without_bottom)
        
        # Step 2: Append flipped version to bottom of original grid
        combined1 = np.vstack([grid, flipped1])
        
        # Step 3: Take the combined grid excluding its new bottom row and flip vertically
        without_new_bottom = combined1[:-1, :]
        flipped2 = np.flipud(without_new_bottom)
        
        # Step 4: Append this second flipped version to bottom of current grid
        final_grid = np.vstack([combined1, flipped2])
        
        return final_grid

    def create_grids(self):
        num_examples = random.randint(3, 6)
        
        # Generate valid columns that allow at least 5 rows
        # We need columns // 2 >= 5, so columns >= 10
        # Valid columns satisfying both columns % 4 == 1 and columns >= 10: 13, 17, 21, 25, 29
        valid_columns = [c for c in range(13, 31, 4)]  # 13, 17, 21, 25, 29
        
        if not valid_columns:  # Safety fallback
            valid_columns = [13]
            
        columns = random.choice(valid_columns)
        
        taskvars = {
            'columns': columns
        }
        
        # Generate training examples with different colors and row counts
        train_examples = []
        used_colors = set()
        
        for _ in range(num_examples):
            # Choose a unique color for this example
            available_colors = [c for c in range(1, 10) if c not in used_colors]
            if not available_colors:
                used_colors.clear()
                available_colors = list(range(1, 10))
            
            color = random.choice(available_colors)
            used_colors.add(color)
            
            # Generate rows for this specific grid (constraint: rows <= columns/2)
            max_rows = min(30, columns // 2)
            # Ensure max_rows >= 5
            max_rows = max(5, max_rows)
            rows = random.randint(5, max_rows)
            
            gridvars = {'color': color, 'rows': rows}
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Generate test example
        available_colors = [c for c in range(1, 10) if c not in used_colors]
        if not available_colors:
            available_colors = [random.randint(1, 9)]
        
        test_color = random.choice(available_colors)
        max_rows = min(30, columns // 2)
        max_rows = max(5, max_rows)  # Ensure max_rows >= 5
        test_rows = random.randint(5, max_rows)
        
        test_gridvars = {'color': test_color, 'rows': test_rows}
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


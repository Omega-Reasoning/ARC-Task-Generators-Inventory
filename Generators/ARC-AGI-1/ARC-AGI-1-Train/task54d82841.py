from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from transformation_library import find_connected_objects, GridObject
from input_library import create_object, Contiguity, retry

class Task54d82841Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['rows']}x{vars['cols']}.",
            "They contain one or more n-shaped objects, where an n-shaped object follows the form: [[c,c,c],[c,0,c]] for a color c.",
            "Each n-shaped object is positioned within its own set of three consecutive columns, ensuring that no two objects share the same column space.",
            "The colors of the n-shaped objects within the same grid may sometimes be the same, but they should vary across examples.",
            "All other cells remain empty (0)."
        ]
        
        transformation_reasoning_chain = [
            "Output grids are constructed by copying the input grids and identifying the n-shaped objects within the grid, where an n-shaped object follows the form: [[c,c,c],[c,0,c]] for a color c.",
            "Once identified, for each object, a {color('fill_color')} cell is placed in the last row, exactly in the middle column of the respective object."
        ]
        
        taskvars_definitions = {
            'rows': lambda: random.randint(5, 30),
            'cols': lambda: random.randint(5, 30),
            'fill_color': lambda: random.randint(1, 9)
        }
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self):
        # Initialize task variables
        taskvars = {
            'rows': random.randint(5, 30),
            'cols': random.randint(5, 30),
            'fill_color': random.randint(1, 9)
        }
        
        # Number of train examples
        nr_train_examples = 3

        # Determine maximum possible objects and allowed counts
        max_objects = taskvars['cols'] // 3
        possible_counts = list(range(1, max(1, min(5, max_objects)) + 1))

        # Generate train examples, retry a few times if they accidentally cover all possible counts
        train_examples = []
        for attempt in range(10):
            train_examples = []
            for _ in range(nr_train_examples):
                # Create input grid with random number of N-shaped objects
                input_grid = self.create_input(taskvars, {})

                # Ensure fill_color is different from any color in the grid
                unique_colors = np.unique(input_grid)
                unique_colors = unique_colors[unique_colors != 0]  # Exclude background

                # If fill_color is among the grid colors, choose a different one
                if taskvars['fill_color'] in unique_colors:
                    available_colors = [c for c in range(1, 10) if c not in unique_colors]
                    if available_colors:
                        taskvars['fill_color'] = random.choice(available_colors)
                    else:
                        # If all colors are used, just pick a different one
                        new_fill_color = random.randint(1, 9)
                        while new_fill_color in unique_colors:
                            new_fill_color = random.randint(1, 9)
                        taskvars['fill_color'] = new_fill_color

                # Transform input grid to get output grid
                output_grid = self.transform_input(input_grid.copy(), taskvars)
                train_examples.append({
                    'input': input_grid,
                    'output': output_grid
                })

            # Compute train counts and ensure they don't cover all possible counts (so a distinct test count exists)
            train_counts = [self._count_n_shapes(ex['input'], taskvars['fill_color']) for ex in train_examples]
            if set(train_counts) != set(possible_counts):
                break

        # If after retries train examples still cover all possible counts, reduce nr_train_examples by 1
        if set(train_counts) == set(possible_counts):
            nr_train_examples = max(1, nr_train_examples - 1)
            train_examples = train_examples[:nr_train_examples]

        # Now pick a test count that's not present in any train example
        train_counts = [self._count_n_shapes(ex['input'], taskvars['fill_color']) for ex in train_examples]
        remaining_counts = [c for c in possible_counts if c not in train_counts]

        if not remaining_counts:
            # Fallback: regenerate a single test grid and ensure its count differs by regenerating one of the train examples
            # (This is extremely unlikely given retries above)
            test_input_grid = self.create_input(taskvars, {})
            test_count = self._count_n_shapes(test_input_grid, taskvars['fill_color'])
            if test_count in train_counts:
                # Replace first train example with a regenerated one that doesn't match test_count
                for _ in range(20):
                    new_train_grid = self.create_input(taskvars, {})
                    new_count = self._count_n_shapes(new_train_grid, taskvars['fill_color'])
                    if new_count != test_count:
                        new_output = self.transform_input(new_train_grid.copy(), taskvars)
                        train_examples[0] = {'input': new_train_grid, 'output': new_output}
                        break
        else:
            desired_count = random.choice(remaining_counts)
            test_input_grid = self.create_input(taskvars, {'num_objects': desired_count})

        # Ensure fill_color is different from any color in test grid
        unique_colors = np.unique(test_input_grid)
        unique_colors = unique_colors[unique_colors != 0]  # Exclude background

        if taskvars['fill_color'] in unique_colors:
            available_colors = [c for c in range(1, 10) if c not in unique_colors]
            if available_colors:
                taskvars['fill_color'] = random.choice(available_colors)
            else:
                # If all colors are used, just pick a different one
                new_fill_color = random.randint(1, 9)
                while new_fill_color in unique_colors:
                    new_fill_color = random.randint(1, 9)
                taskvars['fill_color'] = new_fill_color

        test_output_grid = self.transform_input(test_input_grid.copy(), taskvars)
        test_examples = [{
            'input': test_input_grid,
            'output': test_output_grid
        }]
        
        return taskvars, {
            'train': train_examples,
            'test': test_examples
        }
    
    def create_input(self, taskvars, gridvars):
        rows = taskvars['rows']
        cols = taskvars['cols']
        
        # Create an empty grid
        grid = np.zeros((rows, cols), dtype=int)
        
        # Determine number of N-shaped objects to place
        max_objects = cols // 3  # Each object needs 3 columns
        allowed_max = max(1, min(5, max_objects))

        # Allow callers to force a specific number of objects via gridvars
        if gridvars and 'num_objects' in gridvars:
            try:
                forced = int(gridvars.get('num_objects'))
            except Exception:
                forced = None
            if forced is not None:
                # Clamp to valid range
                num_objects = max(1, min(forced, allowed_max))
            else:
                num_objects = random.randint(1, allowed_max)
        else:
            num_objects = random.randint(1, allowed_max)
        
        # Choose colors for the objects, ensuring they're different from fill_color
        fill_color = taskvars['fill_color']
        available_colors = [c for c in range(1, 10) if c != fill_color]
        colors = [random.choice(available_colors) for _ in range(num_objects)]
        
        # Determine positions for the N-shaped objects
        available_column_sets = list(range(0, cols - 2, 3))
        random.shuffle(available_column_sets)
        
        for i in range(num_objects):
            if not available_column_sets:
                break
                
            col_start = available_column_sets.pop(0)
            
            # Check if there's enough height to place the object
            if rows < 3:  # Need at least 3 rows (object height=2 plus last row for output)
                continue
                
            # Randomly choose a row to place the object, but never in the last two rows
            # (N-shape has height 2, so last two rows should be avoided)
            max_row_start = rows - 3
            if max_row_start < 0:
                continue  # Skip if not enough rows
                
            row_start = random.randint(0, max_row_start)
            
            # Place the N-shaped object
            color = colors[i]
            # Top row: [c,c,c]
            grid[row_start, col_start:col_start+3] = color
            # Bottom row: [c,0,c]
            grid[row_start+1, col_start] = color
            grid[row_start+1, col_start+2] = color
            
        # Ensure each same-colored object is not 4-way connected to another
        self._separate_same_colored_objects(grid, fill_color)
            
        return grid
    
    def _separate_same_colored_objects(self, grid, fill_color):
        # Find all objects of each color
        for color in range(1, 10):
            # Skip the fill_color
            if color == fill_color:
                continue
                
            # Create a mask for this color
            color_mask = grid == color
            if not np.any(color_mask):
                continue
                
            # Find connected components
            objects = find_connected_objects(grid, diagonal_connectivity=False, background=0)
            objects = objects.with_color(color)
            
            # No need to separate if there's only one object of this color
            if len(objects) <= 1:
                continue
                
            # Check if objects are too close (4-way connected)
            for i, obj1 in enumerate(objects):
                for j, obj2 in enumerate(objects):
                    if i >= j:
                        continue
                    
                    # Check if they touch
                    if obj1.touches(obj2, diag=False):
                        # Objects are touching, change one object's color
                        available_colors = [c for c in range(1, 10) if c != fill_color and c != color]
                        if available_colors:
                            new_color = random.choice(available_colors)
                        else:
                            continue  # Skip if no other colors available
                            
                        # Replace the color in the grid
                        for r, c, _ in obj2:
                            grid[r, c] = new_color

    def _count_n_shapes(self, grid, fill_color):
        """Count N-shaped objects in a grid using the same detection rules as transform_input.

        An N-shaped object has the pattern:
        [c,c,c]
        [c,0,c]
        and object color must not be the fill_color.
        """
        rows, cols = grid.shape
        count = 0
        for r in range(rows - 1):
            for c in range(cols - 2):
                if (c+2 < cols and r+1 < rows and
                    grid[r, c] != 0 and grid[r, c] != fill_color and
                    grid[r, c+1] == grid[r, c] and
                    grid[r, c+2] == grid[r, c] and
                    grid[r+1, c] == grid[r, c] and
                    grid[r+1, c+1] == 0 and
                    grid[r+1, c+2] == grid[r, c]):
                    count += 1
        return count
    
    def transform_input(self, grid, taskvars):
        # Copy the input grid
        output_grid = grid.copy()
        
        rows, cols = output_grid.shape
        fill_color = taskvars['fill_color']
        
        # Find all possible N-shaped objects
        for r in range(rows - 1):
            for c in range(cols - 2):
                # Check if this is an N-shaped object
                # Pattern should be:
                # [c,c,c]
                # [c,0,c]
                if (c+2 < cols and r+1 < rows and
                    grid[r, c] != 0 and grid[r, c] != fill_color and  # Ensure object color isn't fill_color
                    grid[r, c+1] == grid[r, c] and 
                    grid[r, c+2] == grid[r, c] and
                    grid[r+1, c] == grid[r, c] and
                    grid[r+1, c+1] == 0 and
                    grid[r+1, c+2] == grid[r, c]):
                    
                    # Found an N-shaped object, add fill_color to the last row at the same column
                    # as the empty cell in the second row
                    output_grid[rows-1, c+1] = fill_color
        
        return output_grid


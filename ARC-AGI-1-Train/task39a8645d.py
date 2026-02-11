from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from input_library import create_object, Contiguity
from transformation_library import find_connected_objects, GridObject

class Task39a8645dGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['grid_size']}x{vars['grid_size']}.",
            "They contain at least three colored objects, where each object is made of 8-way connected cells, with the remaining cells being empty (0).",
            "Each object is confined within a {vars['square_size']}x{vars['square_size']} subgrid and is completely surrounded by empty (0) cells.",
            "There are two or three different colors used in each input grid, with one color appearing more frequently than the others.",
            "Objects of the same color must have the exact same shape and structure."
        ]
        
        transformation_reasoning_chain = [
            "Output grids are always of size {vars['square_size']}x{vars['square_size']}.",
            "The output grid is constructed by identifying the most frequently occurring object (by color and shape) and copying the confined {vars['square_size']}x{vars['square_size']} subgrid of that object into the output grid."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self) -> tuple:
        # Initialize task variables
        grid_size = random.randint(10, 30)
        # Choose square_size (3, 4, or 5) depending on grid_size so objects fit sensibly
        if grid_size < 12:
            square_size = 3
        elif grid_size < 22:
            square_size = 4
        else:
            square_size = 5

        taskvars = {'grid_size': grid_size, 'square_size': square_size}
        
        # Randomize number of training examples
        num_train = random.randint(3, 6)

        # Ensure every grid (train + test) has a different number of objects,
        # and each grid must contain more than three objects (i.e., >= 4).
        total_examples = num_train + 1  # include one test example

        # Conservative estimate for how many 3x3 objects can fit with spacing
        max_possible = max(4, (grid_size // 4) ** 2)
        upper = max(4, min(max_possible, 12))

        # If range is too small, expand the upper bound so we can pick distinct counts
        if (upper - 4 + 1) < total_examples:
            upper = max(4 + total_examples - 1, upper)

        possible_counts = list(range(4, upper + 1))
        if len(possible_counts) < total_examples:
            # fallback incremental distinct counts
            desired_counts = [4 + i for i in range(total_examples)]
        else:
            desired_counts = random.sample(possible_counts, total_examples)

        train_examples = []
        for i in range(num_train):
            desired = desired_counts[i]
            input_grid = self.create_input(taskvars, {'desired_total_objects': desired})
            output_grid = self.transform_input(input_grid, taskvars)
            train_examples.append({'input': input_grid, 'output': output_grid})

        test_examples = []
        test_input = self.create_input(taskvars, {'desired_total_objects': desired_counts[-1]})
        test_output = self.transform_input(test_input, taskvars)
        test_examples.append({'input': test_input, 'output': test_output})
        
        train_test_data = {
            'train': train_examples,
            'test': test_examples
        }
        
        return taskvars, train_test_data
    
    def create_object_template(self, color):
        """Create a random object template of specified color that fits in a square of given size.

        This method will be called with the size value stored in `taskvars['square_size']`.
        """
        # We'll fetch size from the current taskvars if available, otherwise default to 3
        size = getattr(self, '_current_template_size', 3)

        # Create an empty `size x size` template
        template = np.zeros((size, size), dtype=int)

        # Choose number of colored cells: at least `size`, at most a reasonable fraction of the area
        min_cells = max(4, size)
        max_cells = min(size * size - 1, size * 2)
        num_cells = random.randint(min_cells, max_cells)

        cells = [(r, c) for r in range(size) for c in range(size)]
        selected_cells = random.sample(cells, num_cells)
        for r, c in selected_cells:
            template[r, c] = color

        # Ensure object is 8-way connected
        connected = find_connected_objects(template, diagonal_connectivity=True, background=0)
        if len(connected.objects) != 1:
            return self.create_object_template(color)

        # Ensure object occupies all rows and all columns of the subgrid
        rows_used = set(r for r, c in [(r, c) for r in range(size) for c in range(size) if template[r, c] != 0])
        cols_used = set(c for r, c in [(r, c) for r in range(size) for c in range(size) if template[r, c] != 0])

        if len(rows_used) < size or len(cols_used) < size:
            return self.create_object_template(color)

        return template
    
    def create_input(self, taskvars, gridvars):
        grid_size = taskvars['grid_size']
        square_size = taskvars.get('square_size', 3)
        # make the current template size accessible to create_object_template
        self._current_template_size = square_size

        # Determine desired total number of objects for this grid (must be >=4)
        desired_total = gridvars.get('desired_total_objects')
        if desired_total is None or desired_total < 4:
            # pick a reasonable default if none provided
            desired_total = random.randint(4, max(4, min(12, (grid_size // 4) ** 2)))

        # Determine number of colors (exactly 2 or 3)
        num_colors = random.randint(2, 3)
        color_choices = random.sample([1, 2, 3, 4, 5, 6, 7, 8, 9], num_colors)

        # Create template objects for each color
        object_templates = {}
        for color in color_choices:
            object_templates[color] = self.create_object_template(color)

        # We need to split desired_total into positive integers for each color
        # such that there is a unique maximum (one most frequent color).
        color_counts = None
        attempts = 0
        while attempts < 200:
            attempts += 1
            if desired_total < num_colors:
                # impossible to give at least 1 to each color; increase grid and retry
                taskvars['grid_size'] = min(30, grid_size + 4)
                return self.create_input(taskvars, gridvars)

            # random composition: choose (num_colors - 1) cut points
            if num_colors == 1:
                parts = [desired_total]
            else:
                cuts = sorted(random.sample(range(1, desired_total), num_colors - 1))
                parts = []
                prev = 0
                for cut in cuts:
                    parts.append(cut - prev)
                    prev = cut
                parts.append(desired_total - prev)

            # require a unique maximum so there is a clear most frequent color
            if parts.count(max(parts)) == 1:
                # map the largest part to a chosen most frequent color
                max_idx = parts.index(max(parts))
                most_frequent_color = random.choice(color_choices)
                # reorder color_choices so that index max_idx maps to most_frequent_color
                ordered_colors = [most_frequent_color] + [c for c in color_choices if c != most_frequent_color]
                # rotate parts so that parts[0] corresponds to most_frequent_color
                if max_idx != 0:
                    parts = parts[max_idx:] + parts[:max_idx]

                color_counts = {ordered_colors[i]: parts[i] for i in range(num_colors)}
                break

        if color_counts is None:
            # fallback: simple distribution with one clear leader
            leader = random.choice(color_choices)
            color_counts = {c: 1 for c in color_choices}
            remaining = desired_total - num_colors
            color_counts[leader] += remaining

        # Create a grid filled with zeros
        grid = np.zeros((grid_size, grid_size), dtype=int)

        # Place objects on the grid
        successful_placements = {color: 0 for color in color_choices}

        for color in color_choices:
            template = object_templates[color]
            target = color_counts.get(color, 0)
            for _ in range(target):
                # Try to place the object until successful
                placed = False
                attempts_place = 0
                while not placed and attempts_place < 200:
                    # Choose a random position ensuring a 1-cell border around object
                    # valid top-left range: 1 .. grid_size - (square_size + 1)
                    if grid_size - (square_size + 1) < 1:
                        # grid too small for this object + border -- force retry by enlarging grid
                        taskvars['grid_size'] = min(30, grid_size + 4)
                        return self.create_input(taskvars, gridvars)

                    r = random.randint(1, grid_size - (square_size + 1))
                    c = random.randint(1, grid_size - (square_size + 1))

                    # Check if the area (including surrounding cells) is clear
                    clear = True
                    for check_r in range(r-1, r + square_size + 1):
                        for check_c in range(c-1, c + square_size + 1):
                            if 0 <= check_r < grid_size and 0 <= check_c < grid_size:
                                if grid[check_r, check_c] != 0:
                                    clear = False
                                    break
                        if not clear:
                            break

                    if clear:
                        # Place the template
                        for tr in range(square_size):
                            for tc in range(square_size):
                                if template[tr, tc] != 0:
                                    grid[r+tr, c+tc] = template[tr, tc]
                        placed = True
                        successful_placements[color] += 1

                    attempts_place += 1

        # Verify we placed the desired total and that there is a clear most frequent color
        placed_total = sum(successful_placements.values())
        if placed_total != desired_total:
            # Increase grid size and retry to try to achieve the requested pattern
            taskvars['grid_size'] = min(30, grid_size + 4)
            return self.create_input(taskvars, gridvars)

        # ensure the mode is unique (one most frequent color)
        counts_list = list(successful_placements.values())
        max_count = max(counts_list) if counts_list else 0
        if counts_list.count(max_count) != 1:
            taskvars['grid_size'] = min(30, grid_size + 4)
            return self.create_input(taskvars, gridvars)

        # Verify we have both color options present
        if len([c for c, count in successful_placements.items() if count > 0]) < 2:
            # Try again if we don't have at least 2 colors
            return self.create_input(taskvars, gridvars)

        return grid
    
    def transform_input(self, grid, taskvars):
        # Find all objects in the grid
        objects = find_connected_objects(grid, diagonal_connectivity=True, background=0)
        size = taskvars.get('square_size', 3)

        # Group objects by color
        color_objects = {}
        for obj in objects.objects:
            # Verify object occupies a square subgrid of the expected size
            box = obj.bounding_box
            if (box[0].stop - box[0].start != size) or (box[1].stop - box[1].start != size):
                continue
                
            # Get the object's color
            color = list(obj.colors)[0]
            
            if color not in color_objects:
                color_objects[color] = []
            
            color_objects[color].append(obj)
        
        # Count the number of objects of each color
        color_counts = {color: len(objs) for color, objs in color_objects.items()}
        
        # Find the most frequent color
        if not color_counts:
            # If no valid objects found, return empty grid of the appropriate size
            return np.zeros((size, size), dtype=int)
            
        most_frequent_color = max(color_counts, key=color_counts.get)
        
        # Return the first object of the most frequent color
        if color_objects[most_frequent_color]:
            return color_objects[most_frequent_color][0].to_array()
        else:
            # This should never happen given our checks, but just in case
            return np.zeros((size, size), dtype=int)


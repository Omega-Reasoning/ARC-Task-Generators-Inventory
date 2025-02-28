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
            "Each object is confined within a 3x3 subgrid and is completely surrounded by empty (0) cells.",
            "There are two or three different colors used in each input grid, with one color appearing more frequently than the others.",
            "Objects of the same color must have the exact same shape and structure."
        ]
        
        transformation_reasoning_chain = [
            "Output grids are always of size 3x3.",
            "The output grid is constructed by identifying the most frequently occurring object and copying it into the grid."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self) -> tuple:
        # Initialize task variables
        grid_size = random.randint(10, 30)
        taskvars = {'grid_size': grid_size}
        
        # Randomize number of training examples
        num_train = random.randint(3, 6)
        
        train_examples = []
        for _ in range(num_train):
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            train_examples.append({'input': input_grid, 'output': output_grid})
        
        test_examples = []
        test_input = self.create_input(taskvars, {})
        test_output = self.transform_input(test_input, taskvars)
        test_examples.append({'input': test_input, 'output': test_output})
        
        train_test_data = {
            'train': train_examples,
            'test': test_examples
        }
        
        return taskvars, train_test_data
    
    def create_object_template(self, color):
        """Create a random object template of specified color that fits in a 3x3 grid"""
        # Create a 3x3 grid with cells set to 0
        template = np.zeros((3, 3), dtype=int)
        
        # Randomly add 4-6 colored cells
        num_cells = random.randint(4, 6)
        cells = []
        for r in range(3):
            for c in range(3):
                cells.append((r, c))
        
        selected_cells = random.sample(cells, num_cells)
        for r, c in selected_cells:
            template[r, c] = color
            
        # Ensure object is 8-way connected
        connected = find_connected_objects(template, diagonal_connectivity=True, background=0)
        if len(connected.objects) != 1:
            # Try again if not connected
            return self.create_object_template(color)
        
        # Ensure object occupies all 3 rows and all 3 columns
        rows_used = set(r for r, c in [(r, c) for r in range(3) for c in range(3) if template[r, c] != 0])
        cols_used = set(c for r, c in [(r, c) for r in range(3) for c in range(3) if template[r, c] != 0])
        
        if len(rows_used) < 3 or len(cols_used) < 3:
            # Try again if not using all rows and columns
            return self.create_object_template(color)
        
        return template
    
    def create_input(self, taskvars, gridvars):
        grid_size = taskvars['grid_size']
        
        # Determine number of colors (exactly 2 or 3)
        num_colors = random.randint(2, 3)
        color_choices = random.sample([1, 2, 3, 4, 5, 6, 7, 8, 9], num_colors)
        
        # Create template objects for each color
        object_templates = {}
        for color in color_choices:
            object_templates[color] = self.create_object_template(color)
        
        # Choose the most frequent color
        most_frequent_color = random.choice(color_choices)
        
        # Distribute objects to ensure a clear frequency pattern
        # We'll use at least 3 objects of the most frequent color and 1-2 of the others
        min_objects_per_color = 1
        min_lead = 2  # The most frequent color should have at least this many more than any other
        
        # Set the counts for each color
        color_counts = {}
        
        # First, assign the minimum count to each color
        for color in color_choices:
            color_counts[color] = min_objects_per_color
        
        # Set an appropriate count for the most frequent color
        other_max = 0
        for color in color_choices:
            if color != most_frequent_color:
                # Randomly decide how many objects to add (beyond the minimum)
                extra = random.randint(0, 1)
                color_counts[color] += extra
                other_max = max(other_max, color_counts[color])
        
        # Make sure the most frequent color has a clear lead
        color_counts[most_frequent_color] = other_max + min_lead
        
        # Create a grid filled with zeros
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Place objects on the grid
        successful_placements = {color: 0 for color in color_choices}
        
        for color in color_choices:
            template = object_templates[color]
            for _ in range(color_counts[color]):
                # Try to place the object until successful
                placed = False
                attempts = 0
                while not placed and attempts < 100:
                    # Choose a random position ensuring a 1-cell border around object
                    r = random.randint(1, grid_size - 4)
                    c = random.randint(1, grid_size - 4)
                    
                    # Check if the area (including surrounding cells) is clear
                    clear = True
                    for check_r in range(r-1, r+4):  # check 5x5 area (3x3 object + surrounding cells)
                        for check_c in range(c-1, c+4):
                            if 0 <= check_r < grid_size and 0 <= check_c < grid_size:
                                if grid[check_r, check_c] != 0:
                                    clear = False
                                    break
                        if not clear:
                            break
                    
                    if clear:
                        # Place the template
                        for tr in range(3):
                            for tc in range(3):
                                if template[tr, tc] != 0:
                                    grid[r+tr, c+tc] = template[tr, tc]
                        placed = True
                        successful_placements[color] += 1
                    
                    attempts += 1
        
        # Verify we have the correct frequency pattern with at least 3 objects
        if (sum(successful_placements.values()) < 3 or 
            successful_placements[most_frequent_color] <= max(successful_placements[c] for c in color_choices if c != most_frequent_color)):
            # Try again with a larger grid if we couldn't achieve the desired pattern
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
        
        # Group objects by color
        color_objects = {}
        for obj in objects.objects:
            # Verify object occupies a 3x3 subgrid
            box = obj.bounding_box
            if (box[0].stop - box[0].start != 3) or (box[1].stop - box[1].start != 3):
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
            # If no valid objects found, return empty grid
            return np.zeros((3, 3), dtype=int)
            
        most_frequent_color = max(color_counts, key=color_counts.get)
        
        # Return the first object of the most frequent color
        if color_objects[most_frequent_color]:
            return color_objects[most_frequent_color][0].to_array()
        else:
            # This should never happen given our checks, but just in case
            return np.zeros((3, 3), dtype=int)


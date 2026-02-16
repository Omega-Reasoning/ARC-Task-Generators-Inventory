from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from Framework.transformation_library import find_connected_objects
from Framework.input_library import retry, create_object, Contiguity

class Task00dbd492Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are square and can have different sizes.",
            "They contain several {color('object_color')} objects, with the remaining cells being empty (0).",
            "Each {color('object_color')} object has a different size, with possible size options of 5x5, 7x7, and 9x9 and is shaped as a one-cell wide square frame, with an empty (0) interior except for one {color('object_color')} cell exactly in the center of the frame.",
            "All {color('object_color')} objects must be completely separated from each other.",
            "Each {color('object_color')} object should be placed differently across the grids."
        ]
        
        transformation_reasoning_chain = [
            "Output grids are constructed by copying the input grids and identifying the {color('object_color')} square frames, which can be sized as 5x5, 7x7 and 9x9.",
            "The empty (0) cells contained inside each {color('object_color')} square frame are colored based on the size of the square frames.",
            "The 5x5 frame changes the empty (0) cells to {color('fill_color1')}, the 7x7 frame changes the empty (0) cells to {color('fill_color2')}, and the 9x9 frame changes the empty (0) cells to {color('fill_color3')}.",
            "The transformation leaves all {color('object_color')} cells unchanged."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self):
        # Generate task variables
        taskvars = self._generate_task_variables()
        
        # Determine grid size range
        min_grid_size = 10
        max_grid_size = 30
        
        # Create 4 train examples
        train_data = []
        # Keep track of how many frames each train example has so we can
        # guarantee the test example uses a different count.
        train_frame_counts = []

        # Ensure at least one grid has multiple frames (at least 2)
        multi_frame_index = random.randint(0, 3)  # Choose which of the 4 grids will have multiple frames

        for i in range(4):
            # Random grid size between 10x10 and 30x30, but larger for multi-frame grid
            if i == multi_frame_index:
                # For the multi-frame grid, ensure we have enough space
                grid_size = random.randint(20, max_grid_size)  # Larger minimum size
                min_frames = 2  # At least 2 frames for this grid
                max_frames = min(5, (grid_size // 10) + 2)  # Allow more frames
                num_frames = random.randint(min_frames, max_frames)
            else:
                grid_size = random.randint(min_grid_size, max_grid_size)
                min_frames = 1
                max_frames = min(3, (grid_size // 10) + 1)
                num_frames = random.randint(min_frames, max_frames)

            # Create input grid with randomized frame placements
            input_grid = self.create_input(taskvars, {'grid_size': grid_size, 'num_frames': num_frames})

            # Transform the input grid to create the output grid
            output_grid = self.transform_input(input_grid, taskvars)

            train_data.append({
                'input': input_grid,
                'output': output_grid
            })

            # Record how many frames were placed for this training example
            train_frame_counts.append(num_frames)

        # Create 1 test example. Ensure the number of {object_color} frames in the
        # test grid is different from every train grid's frame count.

        # Allowed frame counts: use 1..5 (5 is the maximum we allow elsewhere).
        used_counts = set(train_frame_counts)
        available_counts = [n for n in range(1, 6) if n not in used_counts]

        # If somehow all 1..5 are used (very unlikely with 4 train examples),
        # fall back to allowing 1..5 (can't satisfy uniqueness in that degenerate case).
        if not available_counts:
            available_counts = [1, 2, 3, 4, 5]

        # Choose a test number of frames that is not equal to any of the train counts
        test_num_frames = random.choice(available_counts)

        # Choose a test grid size that can accommodate the chosen number of frames.
        # For placement we use the more generous max formula similar to multi-frame
        # logic: max_frames = min(5, (grid_size // 10) + 2)
        candidate_sizes = [s for s in range(15, max_grid_size + 1)
                           if min(5, (s // 10) + 2) >= test_num_frames]

        if candidate_sizes:
            test_grid_size = random.choice(candidate_sizes)
        else:
            # Fallback to max size if no candidate size found (shouldn't happen)
            test_grid_size = max_grid_size

        test_input_grid = self.create_input(taskvars, {'grid_size': test_grid_size, 'num_frames': test_num_frames})
        test_output_grid = self.transform_input(test_input_grid, taskvars)

        test_data = [{
            'input': test_input_grid,
            'output': test_output_grid
        }]
        
        return taskvars, {
            'train': train_data,
            'test': test_data
        }
    
    def _generate_task_variables(self):
        # Choose distinct colors for objects and fills
        colors = random.sample(range(1, 10), 4)
        
        return {
            'object_color': colors[0],
            'fill_color1': colors[1],  # For 5x5 frames
            'fill_color2': colors[2],  # For 7x7 frames
            'fill_color3': colors[3],  # For 9x9 frames
        }
    
    def create_input(self, taskvars, gridvars):
        grid_size = gridvars.get('grid_size', random.randint(10, 30))
        num_frames = gridvars.get('num_frames', random.randint(1, min(5, (grid_size // 10) + 1)))
        
        # Initialize empty grid
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Possible frame sizes
        frame_sizes = [5, 7, 9]
        
        # Try to include all three frame sizes if we have enough frames
        chosen_sizes = []
        if num_frames >= 3:
            # Include one of each size
            chosen_sizes = frame_sizes.copy()
            # Fill remaining slots randomly
            while len(chosen_sizes) < num_frames:
                chosen_sizes.append(random.choice(frame_sizes))
        else:
            # Pick random sizes, but try to have variety
            for _ in range(num_frames):
                # Prefer sizes we haven't chosen yet
                available_sizes = [size for size in frame_sizes if size not in chosen_sizes]
                if not available_sizes:  # If all sizes used, just pick randomly
                    available_sizes = frame_sizes
                chosen_sizes.append(random.choice(available_sizes))
        
        # Shuffle the sizes
        random.shuffle(chosen_sizes)
        
        # Keep track of placed frames
        placed_frames = 0
        
        # Place each frame
        for frame_size in chosen_sizes:
            max_attempts = 100  # Increased attempts to ensure we can place frames
            for attempt in range(max_attempts):
                # Random position for top-left corner
                r = random.randint(0, grid_size - frame_size)
                c = random.randint(0, grid_size - frame_size)
                
                # Check if this position would overlap with existing frames
                can_place = True
                for i in range(-1, frame_size + 1):  # Check with 1-cell buffer
                    for j in range(-1, frame_size + 1):
                        ri, cj = r + i, c + j
                        if 0 <= ri < grid_size and 0 <= cj < grid_size:
                            if grid[ri, cj] != 0:
                                can_place = False
                                break
                    if not can_place:
                        break
                
                if can_place:
                    # Place the frame
                    for i in range(frame_size):
                        for j in range(frame_size):
                            # Place the frame borders
                            if (i == 0 or i == frame_size - 1 or 
                                j == 0 or j == frame_size - 1):
                                grid[r + i, c + j] = taskvars['object_color']
                    
                    # Place the center cell
                    center = frame_size // 2
                    grid[r + center, c + center] = taskvars['object_color']
                    
                    placed_frames += 1
                    break
            
            # If we couldn't place a frame after max attempts, just continue
        
        # If we couldn't place any frames (unlikely but possible), place at least one
        if placed_frames == 0:
            frame_size = random.choice(frame_sizes)
            r = random.randint(0, grid_size - frame_size)
            c = random.randint(0, grid_size - frame_size)
            
            # Place the frame
            for i in range(frame_size):
                for j in range(frame_size):
                    if (i == 0 or i == frame_size - 1 or j == 0 or j == frame_size - 1):
                        grid[r + i, c + j] = taskvars['object_color']
            
            # Place the center cell
            center = frame_size // 2
            grid[r + center, c + center] = taskvars['object_color']
        
        return grid
    
    def transform_input(self, grid, taskvars):
        # Copy the input grid
        output_grid = np.copy(grid)
        object_color = taskvars['object_color']
        
        # Find all connected objects
        objects = find_connected_objects(grid, diagonal_connectivity=False, 
                                        background=0, monochromatic=False)
        
        # Process each object
        for obj in objects:
            # Only consider objects of the specified color
            if object_color not in obj.colors:
                continue
            
            # Get bounding box to determine frame size
            box = obj.bounding_box
            height = box[0].stop - box[0].start
            width = box[1].stop - box[1].start
            
            # Only process square frames
            if height != width:
                continue
            
            # Determine frame size and corresponding fill color
            frame_size = height
            fill_color = 0  # Default (no fill)
            
            if frame_size == 5:
                fill_color = taskvars['fill_color1']
            elif frame_size == 7:
                fill_color = taskvars['fill_color2']
            elif frame_size == 9:
                fill_color = taskvars['fill_color3']
            else:
                continue  # Skip if not a recognized frame size
            
            # Fill the interior of the frame with the appropriate color
            r_start, c_start = box[0].start, box[1].start
            for r in range(1, frame_size - 1):
                for c in range(1, frame_size - 1):
                    # Skip the center cell
                    if r == frame_size // 2 and c == frame_size // 2:
                        continue
                    output_grid[r_start + r, c_start + c] = fill_color
        
        return output_grid


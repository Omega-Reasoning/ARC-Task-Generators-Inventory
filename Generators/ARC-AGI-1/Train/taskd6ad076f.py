from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.transformation_library import find_connected_objects, GridObjects
from Framework.input_library import retry
import numpy as np
import random
from typing import Dict, Any, Tuple

class Taskd6ad076fGenerator(ARCTaskGenerator):
    
    def __init__(self):
        input_reasoning_chain = [
            "Input grids can have different sizes.",
            "Each grid contains exactly two colored rectangular shapes, each filled with a different color.",
            "One of the two rectangles, in each grid, has a larger length than the other.",
            "The size and color of the rectangular shapes varies across different input grids.",
            "The two rectangles share the same orientation.",
            "If the rectangles are both vertical, then they are separated by at least one column of empty (0) cells, and the smaller rectangle is aligned within the vertical span of the larger rectangle.",
            "If the rectangles are both horizontal, then they are separated by at least one row of empty (0) cells, and the smaller rectangle is aligned within the horizontal span of the larger rectangle.",
            "All the remaining cells are empty."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid.",
            "Two rectangular shapes inside the grid and their colors are identified.",
            "If the rectangles are positioned vertically, the columns between them are filled with color {color('color_3')}, limited to the same horizontal range as the rectangle with smaller length. The first and last rows of the newly filled area are then set to empty (0).",
            "If the rectangles are positioned horizontally, the rows between them are filled with color {color('color_3')}, limited to the same vertical range as the rectangle with smaller length. The first and last columns of the newly filled area are then set to empty (0)."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Generate random task variables
        taskvars = {
            'color_3': random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9])  # Fill color
        }
        
        # Generate training examples (3-6)
        num_train = random.randint(3, 6)
        train_examples = []
        
        for _ in range(num_train):
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Generate test example
        test_input = self.create_input(taskvars, {})
        test_output = self.transform_input(test_input, taskvars)
        test_examples = [{
            'input': test_input,
            'output': test_output
        }]
        
        return taskvars, {
            'train': train_examples,
            'test': test_examples
        }
    
    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        def generate_valid_grid():
            # Random grid size - ensure minimum size to fit two rectangles with gap
            min_size = 12  # Minimum to fit two rectangles with gap and some margin
            height = random.randint(min_size, 30)
            width = random.randint(min_size, 30)
            grid = np.zeros((height, width), dtype=int)
            
            # Choose rectangle orientation (True = vertical rectangles, False = horizontal rectangles)
            rectangles_are_vertical = random.choice([True, False])
            
            # Choose colors for rectangles (different from fill color and each other)
            available_colors = [c for c in range(1, 10) if c != taskvars['color_3']]
            if len(available_colors) < 2:
                return None
            rect_colors = random.sample(available_colors, 2)
            
            if rectangles_are_vertical:
                # Both rectangles are VERTICAL (height > width)
                # They will be placed side by side (separated by columns)
                
                # For vertical rectangles: height > width, and we compare heights to determine "larger length"
                min_width = 3
                max_width = min(6, (width - 3) // 2)  # Leave space for both rectangles and gap
                if max_width < min_width:
                    return None
                
                min_height = 4  # Ensure height > width for vertical orientation
                max_height = min(15, height - 2)
                if max_height < min_height + 1:
                    return None
                
                # Generate rectangle dimensions - both vertical (height > width)
                rect1_width = random.randint(min_width, max_width)
                rect1_height = random.randint(max(min_height, rect1_width + 1), max_height)  # Ensure height > width
                
                rect2_width = random.randint(min_width, max_width)
                rect2_height = random.randint(max(min_height, rect2_width + 1), max_height)  # Ensure height > width
                
                # Ensure different heights (since we compare by height for vertical rectangles)
                while rect2_height == rect1_height:
                    rect2_height = random.randint(max(min_height, rect2_width + 1), max_height)
                    if rect2_height == rect1_height and max_height > min_height:
                        # Try to adjust slightly
                        if rect2_height < max_height:
                            rect2_height += 1
                        elif rect2_height > min_height:
                            rect2_height -= 1
                        else:
                            return None  # Can't make them different
                
                # Determine which is larger by height
                if rect1_height > rect2_height:
                    large_height, small_height = rect1_height, rect2_height
                    large_width, small_width = rect1_width, rect2_width
                    large_color, small_color = rect_colors[0], rect_colors[1]
                else:
                    large_height, small_height = rect2_height, rect1_height
                    large_width, small_width = rect2_width, rect1_width
                    large_color, small_color = rect_colors[1], rect_colors[0]
                
                # Position rectangles side by side
                # Position large rectangle first
                large_row = random.randint(0, height - large_height)
                max_large_col = width - large_width - small_width - 2  # Leave space for small rect + gap
                if max_large_col < 0:
                    return None
                large_col = random.randint(0, max_large_col)
                
                # Position small rectangle (aligned within large rectangle's vertical span)
                align_start = max(0, large_row)
                align_end = min(height - small_height, large_row + large_height - small_height)
                if align_start > align_end:
                    return None
                small_row = random.randint(align_start, align_end)
                
                min_small_col = large_col + large_width + 1
                max_small_col = width - small_width
                if min_small_col > max_small_col:
                    return None
                small_col = random.randint(min_small_col, max_small_col)
                
                # Place rectangles
                grid[large_row:large_row + large_height, large_col:large_col + large_width] = large_color
                grid[small_row:small_row + small_height, small_col:small_col + small_width] = small_color
                
            else:
                # Both rectangles are HORIZONTAL (width > height)
                # They will be placed vertically stacked (separated by rows)
                
                # For horizontal rectangles: width > height, and we compare widths to determine "larger length"
                min_height = 3
                max_height = min(6, (height - 3) // 2)  # Leave space for both rectangles and gap
                if max_height < min_height:
                    return None
                
                min_width = 4  # Ensure width > height for horizontal orientation
                max_width = min(15, width - 2)
                if max_width < min_width + 1:
                    return None
                
                # Generate rectangle dimensions - both horizontal (width > height)
                rect1_height = random.randint(min_height, max_height)
                rect1_width = random.randint(max(min_width, rect1_height + 1), max_width)  # Ensure width > height
                
                rect2_height = random.randint(min_height, max_height)
                rect2_width = random.randint(max(min_width, rect2_height + 1), max_width)  # Ensure width > height
                
                # Ensure different widths (since we compare by width for horizontal rectangles)
                while rect2_width == rect1_width:
                    rect2_width = random.randint(max(min_width, rect2_height + 1), max_width)
                    if rect2_width == rect1_width and max_width > min_width:
                        # Try to adjust slightly
                        if rect2_width < max_width:
                            rect2_width += 1
                        elif rect2_width > min_width:
                            rect2_width -= 1
                        else:
                            return None  # Can't make them different
                
                # Determine which is larger by width
                if rect1_width > rect2_width:
                    large_width, small_width = rect1_width, rect2_width
                    large_height, small_height = rect1_height, rect2_height
                    large_color, small_color = rect_colors[0], rect_colors[1]
                else:
                    large_width, small_width = rect2_width, rect1_width
                    large_height, small_height = rect2_height, rect1_height
                    large_color, small_color = rect_colors[1], rect_colors[0]
                
                # Position rectangles vertically stacked
                # Position large rectangle first
                max_large_row = height - large_height - small_height - 2  # Leave space for small rect + gap
                if max_large_row < 0:
                    return None
                large_row = random.randint(0, max_large_row)
                large_col = random.randint(0, width - large_width)
                
                # Position small rectangle (aligned within large rectangle's horizontal span)
                align_start = max(0, large_col)
                align_end = min(width - small_width, large_col + large_width - small_width)
                if align_start > align_end:
                    return None
                small_col = random.randint(align_start, align_end)
                
                min_small_row = large_row + large_height + 1
                max_small_row = height - small_height
                if min_small_row > max_small_row:
                    return None
                small_row = random.randint(min_small_row, max_small_row)
                
                # Place rectangles
                grid[large_row:large_row + large_height, large_col:large_col + large_width] = large_color
                grid[small_row:small_row + small_height, small_col:small_col + small_width] = small_color
            
            return grid
        
        # Use retry to ensure we generate a valid grid
        return retry(generate_valid_grid, lambda x: x is not None, max_attempts=100)
    
    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        output_grid = grid.copy()
        
        # Find the two rectangles
        objects = find_connected_objects(grid, diagonal_connectivity=False, background=0, monochromatic=True)
        
        if len(objects) != 2:
            return output_grid
        
        rect1, rect2 = objects[0], objects[1]
        
        # Get bounding boxes
        rect1_box = rect1.bounding_box
        rect2_box = rect2.bounding_box
        
        rect1_height = rect1_box[0].stop - rect1_box[0].start
        rect1_width = rect1_box[1].stop - rect1_box[1].start
        rect2_height = rect2_box[0].stop - rect2_box[0].start
        rect2_width = rect2_box[1].stop - rect2_box[1].start
        
        # Determine rectangle orientation by checking if they are vertical or horizontal
        rect1_is_vertical = rect1_height > rect1_width
        rect2_is_vertical = rect2_height > rect2_width
        
        # Both rectangles should have the same orientation
        if rect1_is_vertical and rect2_is_vertical:
            # Both rectangles are vertical - they should be side by side
            # Compare by height to find the one with larger length
            if rect1_height > rect2_height:
                large_rect, small_rect = rect1, rect2
                large_box, small_box = rect1_box, rect2_box
            else:
                large_rect, small_rect = rect2, rect1
                large_box, small_box = rect2_box, rect1_box
            
            # Fill columns between rectangles, limited to small rectangle's row range
            rect1_left, rect1_right = rect1_box[1].start, rect1_box[1].stop
            rect2_left, rect2_right = rect2_box[1].start, rect2_box[1].stop
            
            if rect1_right <= rect2_left:
                fill_col_start = rect1_right
                fill_col_end = rect2_left
            elif rect2_right <= rect1_left:
                fill_col_start = rect2_right
                fill_col_end = rect1_left
            else:
                return output_grid  # Rectangles overlap, shouldn't happen
            
            fill_row_start = small_box[0].start
            fill_row_end = small_box[0].stop
            
            # Fill the area
            if fill_col_start < fill_col_end and fill_row_end - fill_row_start > 2:
                output_grid[fill_row_start:fill_row_end, fill_col_start:fill_col_end] = taskvars['color_3']
                
                # Remove first and last rows of filled area
                output_grid[fill_row_start, fill_col_start:fill_col_end] = 0
                output_grid[fill_row_end - 1, fill_col_start:fill_col_end] = 0
                    
        elif not rect1_is_vertical and not rect2_is_vertical:
            # Both rectangles are horizontal - they should be stacked vertically
            # Compare by width to find the one with larger length
            if rect1_width > rect2_width:
                large_rect, small_rect = rect1, rect2
                large_box, small_box = rect1_box, rect2_box
            else:
                large_rect, small_rect = rect2, rect1
                large_box, small_box = rect2_box, rect1_box
            
            # Fill rows between rectangles, limited to small rectangle's column range
            rect1_top, rect1_bottom = rect1_box[0].start, rect1_box[0].stop
            rect2_top, rect2_bottom = rect2_box[0].start, rect2_box[0].stop
            
            if rect1_bottom <= rect2_top:
                fill_row_start = rect1_bottom
                fill_row_end = rect2_top
            elif rect2_bottom <= rect1_top:
                fill_row_start = rect2_bottom
                fill_row_end = rect1_top
            else:
                return output_grid  # Rectangles overlap, shouldn't happen
            
            fill_col_start = small_box[1].start
            fill_col_end = small_box[1].stop
            
            # Fill the area
            if fill_row_start < fill_row_end and fill_col_end - fill_col_start > 2:
                output_grid[fill_row_start:fill_row_end, fill_col_start:fill_col_end] = taskvars['color_3']
                
                # Remove first and last columns of filled area
                output_grid[fill_row_start:fill_row_end, fill_col_start] = 0
                output_grid[fill_row_start:fill_row_end, fill_col_end - 1] = 0
        
        return output_grid

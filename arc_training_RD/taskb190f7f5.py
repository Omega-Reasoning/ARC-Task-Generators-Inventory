from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from transformation_library import find_connected_objects, GridObject, GridObjects
from input_library import create_object, Contiguity, retry

class Taskb190f7f5Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Each input grid has variable size with width being twice the height.",
            "The grid is conceptually split into two equal halves.",
            "The left half contains a pattern made up of a {color('template_color')} color or a uniform shape.",
            "The right half contains a shape that consists of multiple colors arranged meaningfully (e.g., in a T or cross shape).",
            "These roles may also be interchanged.",
            "The left half might have the multi-colored pattern while the right half contains the single-color shape.",
            "The {color('template_color')} color shape uses the same color across all input grids."
        ]
        
        transformation_reasoning_chain = [
            "The goal is to use the {color('template_color')} color shape as a template and the multi-colored shape as a layout guide.",
            "The {color('template_color')} color shape as a template or building block.",
            "The multi-colored shape as a layout guide to decide where and how to place copies of the building block.",
            "Create a new output grid that is square with size equal to the square of input height (row² × row²).",
            "For each colored cell in the multi-colored shape.",
            "Place a copy of the template shape at the corresponding location in the output.",
            "But recolor the copy to match the color of that cell in the multi-colored shape."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_single_color_pattern(self, template_color, half_size):
        """Create a pattern with the specified template color and size"""
        
        # Define complete pattern types that guarantee visibility
        if half_size == 2:
            patterns = {
                'plus': np.array([
                    [0, 1],
                    [1, 0]
                ]),
                'corner_l': np.array([
                    [1, 0],
                    [1, 1]
                ]),
                'diagonal': np.array([
                    [1, 0],
                    [0, 1]
                ]),
                'line': np.array([
                    [1, 1],
                    [0, 0]
                ]),
                'small_square': np.array([
                    [1, 1],
                    [1, 0]
                ])
            }
        elif half_size == 3:
            patterns = {
                'plus': np.array([
                    [0, 1, 0],
                    [1, 1, 1], 
                    [0, 1, 0]
                ]),
                'corner_l': np.array([
                    [1, 0, 0],
                    [1, 0, 0],
                    [1, 1, 1]
                ]),
                'diagonal': np.array([
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]
                ]),
                'line_vertical': np.array([
                    [0, 1, 0],
                    [0, 1, 0],
                    [0, 1, 0]
                ]),
                'line_horizontal': np.array([
                    [0, 0, 0],
                    [1, 1, 1],
                    [0, 0, 0]
                ]),
                'small_square': np.array([
                    [1, 1, 0],
                    [1, 1, 0],
                    [0, 0, 0]
                ]),
                'corners': np.array([
                    [1, 0, 1],
                    [0, 0, 0],
                    [1, 0, 1]
                ]),
                'border': np.array([
                    [1, 1, 1],
                    [1, 0, 1],
                    [1, 1, 1]
                ])
            }
        elif half_size == 4:
            patterns = {
                'plus': np.array([
                    [0, 0, 1, 0],
                    [0, 0, 1, 0],
                    [1, 1, 1, 1],
                    [0, 0, 1, 0]
                ]),
                'corner_l': np.array([
                    [1, 0, 0, 0],
                    [1, 0, 0, 0],
                    [1, 0, 0, 0],
                    [1, 1, 1, 1]
                ]),
                'diagonal': np.array([
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]
                ]),
                'border': np.array([
                    [1, 1, 1, 1],
                    [1, 0, 0, 1],
                    [1, 0, 0, 1],
                    [1, 1, 1, 1]
                ]),
                'small_square': np.array([
                    [1, 1, 0, 0],
                    [1, 1, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0]
                ])
            }
        else:
            # Fallback for other sizes
            patterns = {
                'diagonal': np.eye(half_size, dtype=int),
                'center': np.zeros((half_size, half_size))
            }
            patterns['center'][half_size//2, half_size//2] = 1
        
        pattern_name = random.choice(list(patterns.keys()))
        pattern = patterns[pattern_name] * template_color
        
        return pattern

    def create_multicolor_pattern(self, layout_colors, half_size):
        """Create a pattern with the specified layout colors and size"""
        
        # Create layouts that guarantee all colors are used
        if len(layout_colors) == 2:
            if half_size == 2:
                layouts = {
                    'diagonal': [(0, 0), (1, 1)],
                    'adjacent': [(0, 0), (0, 1)],
                    'vertical': [(0, 0), (1, 0)]
                }
            else:
                layouts = {
                    'two_line': [(0, 1), (1, 1)],
                    'diagonal_pair': [(0, 0), (2, 2)],
                    'adjacent': [(1, 0), (1, 1)]
                }
        elif len(layout_colors) == 3:
            if half_size >= 3:
                layouts = {
                    'line_three': [(1, 0), (1, 1), (1, 2)],
                    'triangle': [(0, 1), (1, 0), (1, 2)],
                    'diagonal_three': [(0, 0), (1, 1), (2, 2)],
                    'corner_three': [(0, 0), (0, 2), (2, 1)]
                }
            else:
                layouts = {
                    'three_pos': [(0, 0), (0, 1), (1, 0)]
                }
        else:  # 4+ colors
            if half_size >= 3:
                layouts = {
                    'corners_four': [(0, 0), (0, 2), (2, 0), (2, 2)],
                    'cross_four': [(0, 1), (1, 0), (1, 2), (2, 1)],
                    'square_four': [(0, 0), (0, 1), (1, 0), (1, 1)]
                }
            else:
                layouts = {
                    'four_pos': [(0, 0), (0, 1), (1, 0), (1, 1)]
                }
        
        layout_name = random.choice(list(layouts.keys()))
        positions = layouts[layout_name]
        
        pattern = np.zeros((half_size, half_size), dtype=int)
        # Assign each color to a position
        for i, (r, c) in enumerate(positions):
            if i < len(layout_colors) and r < half_size and c < half_size:
                pattern[r, c] = layout_colors[i]
        
        return pattern

    def create_input(self, taskvars):
        """Create input grid with template and layout halves"""
        # Randomly select grid dimensions for this specific input
        half_size = random.randint(2, 4)  # Half size can be 2, 3, or 4
        input_height = half_size
        input_width = half_size * 2  # Width is always twice the half_size
        
        grid = np.zeros((input_height, input_width), dtype=int)
        
        # Use the consistent template color from taskvars
        template_color = taskvars['template_color']
        
        # Generate 2-4 different colors for the layout (excluding template color)
        max_layout_colors = min(4, half_size * half_size - 1)
        num_layout_colors = random.randint(2, max_layout_colors)
        available_colors = [c for c in range(1, 10) if c != template_color]
        layout_colors = random.sample(available_colors, num_layout_colors)
        
        # Create the two patterns using these colors
        template_pattern = self.create_single_color_pattern(template_color, half_size)
        layout_pattern = self.create_multicolor_pattern(layout_colors, half_size)
        
        # Verify patterns are not empty
        if np.all(template_pattern == 0):
            # Fallback: create a simple plus pattern
            template_pattern = np.zeros((half_size, half_size), dtype=int)
            if half_size == 2:
                template_pattern = np.array([[0, template_color], [template_color, 0]])
            elif half_size == 3:
                template_pattern = np.array([
                    [0, template_color, 0],
                    [template_color, template_color, template_color], 
                    [0, template_color, 0]
                ])
            else:
                template_pattern[0, 0] = template_color
        
        if np.all(layout_pattern == 0):
            # Fallback: create a simple two-color pattern
            layout_pattern = np.zeros((half_size, half_size), dtype=int)
            layout_pattern[0, 0] = layout_colors[0]
            if len(layout_colors) > 1 and half_size > 1:
                layout_pattern[1, 1] = layout_colors[1]
        
        # Randomly decide which half gets which pattern
        template_on_left = random.choice([True, False])
        
        if template_on_left:
            grid[:half_size, :half_size] = template_pattern
            grid[:half_size, half_size:] = layout_pattern
        else:
            grid[:half_size, :half_size] = layout_pattern  
            grid[:half_size, half_size:] = template_pattern
            
        return grid

    def transform_input(self, grid, taskvars):
        """Transform input grid by extracting template and layout, then assembling output"""
        template_color = taskvars['template_color']
        
        # Determine grid dimensions from the input
        input_height, input_width = grid.shape
        half_size = input_height  # Since height = half_size in our design
        
        # Split the grid into two halves
        left_half = grid[:half_size, :half_size]
        right_half = grid[:half_size, half_size:]
        
        # Determine which half contains the template (single color pattern)
        left_has_template = np.any(left_half == template_color) and len(np.unique(left_half[left_half != 0])) == 1
        right_has_template = np.any(right_half == template_color) and len(np.unique(right_half[right_half != 0])) == 1
        
        if left_has_template:
            template_half = left_half
            layout_half = right_half
        else:
            template_half = right_half
            layout_half = left_half
        
        # Get the template pattern (remove color, keep shape)
        template_shape = (template_half != 0).astype(int)
        
        # Calculate output size - row² × row²
        output_size = input_height * input_height
        output_grid = np.zeros((output_size, output_size), dtype=int)
        
        # For each colored cell in the layout pattern
        for r in range(half_size):
            for c in range(half_size):
                if layout_half[r, c] != 0:
                    cell_color = layout_half[r, c]
                    
                    # Place a copy of the template at the corresponding location
                    # Each layout cell maps to a half_size x half_size block in the output
                    start_r = r * half_size
                    start_c = c * half_size
                    
                    # Ensure we don't go out of bounds of the row²×row² output
                    end_r = min(start_r + half_size, output_size)
                    end_c = min(start_c + half_size, output_size)
                    
                    # Get the portion of template that fits
                    template_height = end_r - start_r
                    template_width = end_c - start_c
                    
                    # Create colored copy of template (or portion that fits)
                    if template_height > 0 and template_width > 0:
                        colored_template = template_shape[:template_height, :template_width] * cell_color
                        
                        # Place it in the output grid
                        output_grid[start_r:end_r, start_c:end_c] = colored_template
        
        return output_grid

    def create_grids(self):
        num_train_pairs = random.randint(3, 5)
        train_pairs = []

        # Choose a consistent template color for all grids
        template_color = random.randint(1, 9)
        
        taskvars = {
            'template_color': template_color
        }
        
        # Generate training pairs with validation
        attempts = 0
        while len(train_pairs) < num_train_pairs and attempts < 50:
            try:
                input_grid = self.create_input(taskvars)
                output_grid = self.transform_input(input_grid.copy(), taskvars)
                
                # Validate the generated grids
                input_height, input_width = input_grid.shape
                expected_output_size = input_height * input_height  # row² × row²
                
                if (input_width == 2 * input_height and  # Width should be twice height
                    output_grid.shape == (expected_output_size, expected_output_size) and  # row² × row²
                    output_grid.size > 0 and 
                    np.any(input_grid != 0) and
                    np.any(output_grid != 0)):
                    
                    train_pairs.append(GridPair(input=input_grid, output=output_grid))
                
            except Exception as e:
                print(f"Failed to generate training pair: {e}")
            
            attempts += 1
        
        # Ensure we have at least one training pair
        if len(train_pairs) == 0:
            raise ValueError("Failed to generate any valid training pairs")
        
        # Generate test pair with validation
        test_attempts = 0
        test_pairs = []
        
        while len(test_pairs) == 0 and test_attempts < 20:
            try:
                test_input = self.create_input(taskvars)
                test_output = self.transform_input(test_input.copy(), taskvars)
                
                # Validate test grid
                test_input_height, test_input_width = test_input.shape
                test_expected_output_size = test_input_height * test_input_height  # row² × row²
                
                if (test_input_width == 2 * test_input_height and  # Width should be twice height
                    test_output.shape == (test_expected_output_size, test_expected_output_size) and  # row² × row²
                    test_output.size > 0 and 
                    np.any(test_input != 0) and
                    np.any(test_output != 0)):
                    
                    test_pairs = [GridPair(input=test_input, output=test_output)]
                
            except Exception as e:
                print(f"Failed to generate test pair: {e}")
            
            test_attempts += 1
        
        if len(test_pairs) == 0:
            raise ValueError("Failed to generate valid test pair")
        
        return taskvars, TrainTestData(train=train_pairs, test=test_pairs)


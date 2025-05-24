from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from transformation_library import find_connected_objects, GridObject, GridObjects
from input_library import create_object, Contiguity, retry

class TaskTemplateReplicationGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Each input grid is of fixed size: 6 columns × 3 rows.",
            "The grid is conceptually split into two equal halves:",
            "- The left half (3×3) contains a pattern made up of {color('template_color')} color or a uniform shape.",
            "- The right half (3×3) contains a shape that consists of multiple colors arranged meaningfully (e.g., in a T or cross shape).",
            "These roles may also be interchanged:",
            "- The left half might have the multi-colored pattern while the right half contains the {color('template_color')} shape."
        ]
        
        transformation_reasoning_chain = [
            "The goal is to use:",
            "- The {color('template_color')} shape as a template or building block.",
            "- The multi-colored shape as a layout guide to decide where and how to place copies of the building block.",
            "Count the number of distinct colors in the multi-colored half. Let this number be N.",
            "Create a new output grid of size (N × 3) × (N × 3) — because the original shape block is 3×3, and we are replicating it N times in each direction.",
            "For each colored cell in the multi-colored shape:",
            "- Place a copy of the {color('template_color')} shape at the corresponding location in the output.",
            "- But recolor the copy to match the color of that cell in the multi-colored shape."
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

    def create_single_color_pattern(self, template_color):
        """Create a 3x3 pattern with the specified template color"""
        
        # Define complete pattern types that guarantee visibility
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
        
        pattern_name = random.choice(list(patterns.keys()))
        pattern = patterns[pattern_name] * template_color
        
        return pattern

    def create_multicolor_pattern(self, layout_colors):
        """Create a 3x3 pattern with the specified layout colors"""
        
        # Create layouts that guarantee all colors are used
        if len(layout_colors) == 2:
            layouts = {
                'two_line': [(0, 1), (1, 1)],
                'diagonal_pair': [(0, 0), (2, 2)],
                'corners_pair': [(0, 0), (2, 2)],
                'adjacent': [(1, 0), (1, 1)]
            }
        elif len(layout_colors) == 3:
            layouts = {
                'line_three': [(1, 0), (1, 1), (1, 2)],
                'triangle': [(0, 1), (1, 0), (1, 2)],
                'diagonal_three': [(0, 0), (1, 1), (2, 2)],
                'corner_three': [(0, 0), (0, 2), (2, 1)]
            }
        else:  # 4 colors
            layouts = {
                'corners_four': [(0, 0), (0, 2), (2, 0), (2, 2)],
                'cross_four': [(0, 1), (1, 0), (1, 2), (2, 1)],
                'square_four': [(0, 0), (0, 1), (1, 0), (1, 1)]
            }
        
        layout_name = random.choice(list(layouts.keys()))
        positions = layouts[layout_name]
        
        pattern = np.zeros((3, 3), dtype=int)
        # Assign each color to a position
        for i, (r, c) in enumerate(positions):
            if i < len(layout_colors):
                pattern[r, c] = layout_colors[i]
        
        return pattern

    def create_input(self, taskvars):
        """Create a 6x3 input grid with template and layout halves"""
        grid = np.zeros((3, 6), dtype=int)
        
        # Use the SAME colors for this specific grid generation
        template_color = taskvars['template_color']
        layout_colors = taskvars['layout_colors']
        
        # Create the two patterns using consistent colors
        template_pattern = self.create_single_color_pattern(template_color)
        layout_pattern = self.create_multicolor_pattern(layout_colors)
        
        # Verify patterns are not empty
        if np.all(template_pattern == 0):
            # Fallback: create a simple plus pattern
            template_pattern = np.array([
                [0, template_color, 0],
                [template_color, template_color, template_color], 
                [0, template_color, 0]
            ])
        
        if np.all(layout_pattern == 0):
            # Fallback: create a simple two-color pattern
            layout_pattern = np.array([
                [layout_colors[0], 0, 0],
                [0, layout_colors[1] if len(layout_colors) > 1 else layout_colors[0], 0],
                [0, 0, 0]
            ])
        
        # Randomly decide which half gets which pattern
        template_on_left = random.choice([True, False])
        
        if template_on_left:
            grid[:, :3] = template_pattern
            grid[:, 3:] = layout_pattern
        else:
            grid[:, :3] = layout_pattern  
            grid[:, 3:] = template_pattern
            
        # Store which side has the template for transformation
        taskvars['template_on_left'] = template_on_left
        
        return grid

    def transform_input(self, input_grid, taskvars):
        """Transform input grid according to the replication rules"""
        template_color = taskvars['template_color']
        layout_colors = taskvars['layout_colors'] 
        template_on_left = taskvars['template_on_left']
        
        # Extract the two halves
        if template_on_left:
            template_half = input_grid[:, :3]
            layout_half = input_grid[:, 3:]
        else:
            template_half = input_grid[:, 3:]
            layout_half = input_grid[:, :3]
        
        # Get the template pattern (remove color, keep shape)
        template_shape = (template_half != 0).astype(int)
        
        # Find actual unique colors present in the layout half
        layout_flat = layout_half.flatten()
        unique_colors = [c for c in np.unique(layout_flat) if c != 0]
        
        # Ensure we have the expected number of colors
        if len(unique_colors) == 0:
            unique_colors = layout_colors[:1]  # Use at least one color
        
        # Calculate maximum position in the layout to determine output size
        layout_positions = []
        for r in range(3):
            for c in range(3):
                if layout_half[r, c] != 0:
                    layout_positions.append((r, c))
        
        if not layout_positions:
            # Fallback if no positions found
            layout_positions = [(0, 0)]
        
        # Calculate output grid size based on maximum position and template size
        max_r = max(pos[0] for pos in layout_positions)
        max_c = max(pos[1] for pos in layout_positions)
        
        # Output size should accommodate the farthest template placement
        output_height = (max_r + 1) * 3
        output_width = (max_c + 1) * 3
        
        # Ensure minimum size and reasonable maximum
        output_height = max(6, min(output_height, 30))
        output_width = max(6, min(output_width, 30))
        
        output_grid = np.zeros((output_height, output_width), dtype=int)
        
        # For each colored cell in the layout pattern
        for r in range(3):
            for c in range(3):
                if layout_half[r, c] != 0:
                    cell_color = layout_half[r, c]
                    
                    # Place a copy of the template at the corresponding location
                    # Each layout cell maps to a 3x3 block in the output
                    start_r = r * 3
                    start_c = c * 3
                    
                    # Ensure we don't go out of bounds
                    end_r = min(start_r + 3, output_height)
                    end_c = min(start_c + 3, output_width)
                    
                    # Get the portion of template that fits
                    template_height = end_r - start_r
                    template_width = end_c - start_c
                    
                    # Create colored copy of template (or portion that fits)
                    colored_template = template_shape[:template_height, :template_width] * cell_color
                    
                    # Place it in the output grid
                    output_grid[start_r:end_r, start_c:end_c] = colored_template
        
        return output_grid

    def create_grids(self):
        num_train_pairs = random.randint(3, 5)
        train_pairs = []
        
        # Generate colors for the ENTIRE TASK (not per grid)
        template_color = random.randint(1, 9)
        
        # Generate 2-4 different colors for the layout (excluding template color)
        num_layout_colors = random.randint(2, 4)
        available_colors = [c for c in range(1, 10) if c != template_color]
        layout_colors = random.sample(available_colors, num_layout_colors)
        
        # Task variables - these stay the same for ALL grids in this task
        taskvars = {
            'template_color': template_color,
            'layout_colors': layout_colors,
        }
        
        # Helper for reasoning chain formatting
        def color_fmt(key):
            color_id = taskvars[key]
            return f"{self.color_name(color_id)} ({color_id})"

        # Replace color placeholders in reasoning chains
        self.input_reasoning_chain = [
            chain.replace("{color('template_color')}", color_fmt('template_color'))
            for chain in self.input_reasoning_chain
        ]
        
        self.transformation_reasoning_chain = [
            chain.replace("{color('template_color')}", color_fmt('template_color'))
            for chain in self.transformation_reasoning_chain
        ]
        
        # Generate training pairs with validation
        attempts = 0
        while len(train_pairs) < num_train_pairs and attempts < 50:
            try:
                # Use the SAME taskvars for each grid (only template_on_left varies)
                grid_taskvars = taskvars.copy()
                
                input_grid = self.create_input(grid_taskvars)
                output_grid = self.transform_input(input_grid, grid_taskvars)
                
                # Validate the generated grids
                if (input_grid.shape == (3, 6) and 
                    output_grid.size > 0 and 
                    output_grid.shape[0] <= 30 and 
                    output_grid.shape[1] <= 30 and
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
                # Use the SAME taskvars for test grid too
                test_taskvars = taskvars.copy()
                test_input = self.create_input(test_taskvars)
                test_output = self.transform_input(test_input, test_taskvars)
                
                # Validate test grid
                if (test_input.shape == (3, 6) and 
                    test_output.size > 0 and 
                    test_output.shape[0] <= 30 and 
                    test_output.shape[1] <= 30 and
                    np.any(test_input != 0) and
                    np.any(test_output != 0)):
                    
                    test_pairs = [GridPair(input=test_input, output=test_output)]
                
            except Exception as e:
                print(f"Failed to generate test pair: {e}")
            
            test_attempts += 1
        
        if len(test_pairs) == 0:
            raise ValueError("Failed to generate valid test pair")
        
        return taskvars, TrainTestData(train=train_pairs, test=test_pairs)

# Test code
if __name__ == "__main__":
    generator = TaskTemplateReplicationGenerator()
    taskvars, train_test_data = generator.create_grids()
    print("Task Variables:", taskvars)
    ARCTaskGenerator.visualize_train_test_data(train_test_data)
from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from input_library import retry
import numpy as np
import random
from typing import Dict, List, Any, Tuple

class Taskeb5a1d5d(ARCTaskGenerator):
    
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['rows']} x {vars['columns']}.",
            "Each input grid contains multiple nested rectangular layers, placed concentrically within one another.",
            "Each layer is filled with a distinct color.",
            "The input grid contains n layers: Layer₁, Layer₂, ..., Layerₙ. Layerₙ is the outermost layer and spans the entire grid or a major portion of it. Layer₁ is the innermost layer and is fully enclosed by all preceding layers.",
            "Each Layerᵢ (for 1 ≤ i < n) satisfies the following conditions: It is a rectangle aligned with the grid. It completely contains Layerᵢ₋₁. It is filled entirely with a single, unique color.",
            "The size of layers varies as long as they meet the above conditions.",
            "The number of layers n varies across different input grid examples.",
            "Layer colors vary between input grids to ensure diversity."
        ]
        
        transformation_reasoning_chain = [
            "The transformation is initiated by identifying the n nested layers present in the input grid, ordered from the outermost (Layerₙ) to the innermost (Layer₁).",
            "The output grid is constructed as a square grid of size (2n - 1) × (2n - 1).",
            "Within the output grid, n nested square layers are arranged concentrically: Each Layerᵢ (for 1 ≤ i ≤ n) is defined as a square of size (2i - 1) × (2i - 1). These layers are centered within the output grid.",
            "The color of each Layerᵢ in the output grid is taken directly from the corresponding Layerᵢ in the input grid.",
            "As a result, a normalized square representation of the nested structure from the input is produced, with the order and color of the original layers preserved"
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Task variables - consistent across all examples
        rows = random.randint(8, 30)  # Ensure enough space for multiple layers
        cols = random.randint(8, 30)
        
        taskvars = {
            'rows': rows,
            'columns': cols
        }
        
        # Create 3-6 training examples and 1 test example
        num_train = random.randint(3, 6)
        num_test = 1
        
        train_examples = []
        test_examples = []
        
        # Generate training examples with varying layers and colors
        used_layer_counts = set()
        
        for _ in range(num_train):
            # Random number of layers (2-6, ensure variety)
            max_possible_layers = min(6, min(rows//2, cols//2))
            num_layers = random.randint(2, max_possible_layers)
            
            # Try to use different layer counts for variety
            if len(used_layer_counts) < num_train and num_layers in used_layer_counts:
                for alt_layers in range(2, max_possible_layers + 1):
                    if alt_layers not in used_layer_counts:
                        num_layers = alt_layers
                        break
            used_layer_counts.add(num_layers)
            
            # Generate unique colors for each layer (avoid reusing color sets)
            colors = random.sample(range(1, 10), num_layers)
            
            gridvars = {
                'num_layers': num_layers,
                'colors': colors
            }
            
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Generate test example with different layer count and colors
        max_possible_layers = min(6, min(rows//2, cols//2))
        test_num_layers = random.randint(2, max_possible_layers)
        test_colors = random.sample(range(1, 10), test_num_layers)
        
        test_gridvars = {
            'num_layers': test_num_layers,
            'colors': test_colors
        }
        
        test_input = self.create_input(taskvars, test_gridvars)
        test_output = self.transform_input(test_input, taskvars)
        
        test_examples.append({
            'input': test_input,
            'output': test_output
        })
        
        return taskvars, {
            'train': train_examples,
            'test': test_examples
        }
    
    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        rows = taskvars['rows']
        cols = taskvars['columns']
        num_layers = gridvars['num_layers']
        colors = gridvars['colors']
        
        # Start with grid filled with outermost layer color (Layerₙ - ensures no empty cells)
        # colors[n-1] is the outermost layer (Layerₙ), colors[0] is innermost (Layer₁)
        grid = np.full((rows, cols), colors[num_layers-1], dtype=int)
        
        # Generate nested rectangles with varied positioning
        layer_bounds = []
        
        # Outermost layer (Layerₙ) always spans the entire grid
        outer_top, outer_bottom = 0, rows - 1
        outer_left, outer_right = 0, cols - 1
        
        current_top, current_bottom = outer_top, outer_bottom
        current_left, current_right = outer_left, outer_right
        
        # Generate each layer from outermost to innermost
        for i in range(num_layers):
            layer_bounds.append((current_top, current_bottom, current_left, current_right))
            
            if i < num_layers - 1:  # Not the last (innermost) layer
                # Calculate space needed for remaining layers
                remaining_layers = num_layers - i - 1
                
                # Available space
                available_height = current_bottom - current_top + 1
                available_width = current_right - current_left + 1
                
                # Minimum space needed for remaining layers (at least 1 pixel each)
                min_height_needed = remaining_layers * 2 + 1
                min_width_needed = remaining_layers * 2 + 1
                
                if available_height <= min_height_needed or available_width <= min_width_needed:
                    break
                
                # Calculate shrink amounts with some randomness
                max_shrink_top = min(3, (available_height - min_height_needed) // 2)
                max_shrink_bottom = min(3, (available_height - min_height_needed) // 2)
                max_shrink_left = min(3, (available_width - min_width_needed) // 2)
                max_shrink_right = min(3, (available_width - min_width_needed) // 2)
                
                # Allow asymmetric shrinking for more variety
                shrink_top = random.randint(1, max(1, max_shrink_top))
                shrink_bottom = random.randint(1, max(1, max_shrink_bottom))
                shrink_left = random.randint(1, max(1, max_shrink_left))
                shrink_right = random.randint(1, max(1, max_shrink_right))
                
                # Update bounds for next inner layer
                current_top += shrink_top
                current_bottom -= shrink_bottom
                current_left += shrink_left
                current_right -= shrink_right
        
        # Fill inner layers (skip the outermost layer since grid is already filled with that color)
        # layer_bounds[0] corresponds to Layerₙ (outermost), layer_bounds[n-1] to Layer₁ (innermost)
        for i in range(1, len(layer_bounds)):
            top, bottom, left, right = layer_bounds[i]
            # colors[num_layers-1-i] gives us the correct color for this layer
            color = colors[num_layers-1-i]
            grid[top:bottom+1, left:right+1] = color
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        # Extract layers from input grid (ordered from outermost to innermost)
        layers_info = self._extract_layers_ordered(grid)
        n = len(layers_info)
        
        # Create output grid of size (2n-1) x (2n-1)
        output_size = 2 * n - 1
        output_grid = np.zeros((output_size, output_size), dtype=int)
        
        # Create concentric square layers centered in output
        center = output_size // 2
        
        # Fill layers from largest to smallest (Layerₙ to Layer₁)
        # According to reasoning chain: Layerᵢ has size (2i - 1) × (2i - 1)
        # Layer₁ (innermost) gets size 1×1, Layer₂ gets 3×3, ..., Layerₙ gets (2n-1)×(2n-1)
        
        for i in range(n):  # i=0 is outermost (Layerₙ), i=n-1 is innermost (Layer₁)
            color, _ = layers_info[i]
            # Layer index in reasoning chain: layer_num = n - i (so outermost Layerₙ -> innermost Layer₁)
            layer_num = n - i
            layer_size = 2 * layer_num - 1
            half_size = layer_size // 2
            
            # Fill the square layer centered in output grid
            top = center - half_size
            bottom = center + half_size
            left = center - half_size
            right = center + half_size
            
            output_grid[top:bottom+1, left:right+1] = color
        
        return output_grid
    
    def _extract_layers_ordered(self, grid: np.ndarray) -> List[Tuple[int, Tuple[int, int, int, int]]]:
        """Extract layer information from input grid as (color, bounds) tuples, ordered from outermost to innermost."""
        rows, cols = grid.shape
        layers = []
        
        # Find layers by examining nested rectangles from outside to inside
        current_bounds = (0, rows - 1, 0, cols - 1)
        
        while True:
            top, bottom, left, right = current_bounds
            
            if top > bottom or left > right:
                break
                
            # Get the color at the current border
            border_color = grid[top, left]
            layers.append((border_color, current_bounds))
            
            # Find the next inner rectangle with a different color
            found_inner = False
            
            # Look for the boundary of the next inner layer
            for offset in range(1, min(bottom - top, right - left) // 2 + 1):
                inner_top = top + offset
                inner_bottom = bottom - offset
                inner_left = left + offset
                inner_right = right - offset
                
                if (inner_top <= inner_bottom and inner_left <= inner_right and
                    grid[inner_top, inner_left] != border_color):
                    
                    # Found a different color - this is the next layer
                    current_bounds = (inner_top, inner_bottom, inner_left, inner_right)
                    found_inner = True
                    break
            
            if not found_inner:
                break
        
        return layers


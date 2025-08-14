from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from input_library import random_cell_coloring, retry
import numpy as np
import random
from typing import Dict, Any, Tuple


class Task833dafe3Generator(ARCTaskGenerator):
    
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['rows']} by {vars['cols']}. They mostly form square grids having rows and columns the same.",
            "Each input grid has a pattern which starts from the first column and runs until the last column.",
            "The columns can have any number of cells filled with any color. But it has consistent colors maintaining a neat pattern."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is twice the size of the input grid.",
            "The grid is now filled in such a way that, the first top-left quadrant of the grid, copies the entire input grid.",
            "The second quadrant that is the top-right section of the grid, is just the mirrored or chiral version of the first quadrant.",
            "The third and fourth quadrants are now again a mirror or chiral of the first and the second quadrant.",
            "Imagine folding a painted paper which is filled only in the top-quadrant and is folded into half vertically. Then the painting gets printed on the other half as a mirror image. Similarly, now fold the paper into half horizontally, such that the upper half transfers its print to the lower half but mirrored."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Generate task variables - preferably square grids but allow some variation
        size = random.choice([3, 4, 5, 6])
        
        # Sometimes make it non-square for variety
        if random.random() < 0.3:
            rows = size
            cols = size + random.choice([-1, 1])
            cols = max(2, min(cols, 6))  # Keep within reasonable bounds
        else:
            rows = cols = size
            
        task_variables = {
            'rows': rows,
            'cols': cols
        }
        
        # Generate 3-5 training examples and 1 test example
        num_train = random.randint(3, 5)
        
        train_pairs = []
        for _ in range(num_train):
            input_grid = self.create_input(task_variables, {})
            output_grid = self.transform_input(input_grid, task_variables)
            train_pairs.append({'input': input_grid, 'output': output_grid})
        
        test_pairs = []
        input_grid = self.create_input(task_variables, {})
        output_grid = self.transform_input(input_grid, task_variables)
        test_pairs.append({'input': input_grid, 'output': output_grid})
        
        train_test_data = {
            'train': train_pairs,
            'test': test_pairs
        }
        
        return task_variables, train_test_data
    
    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        rows = taskvars['rows']
        cols = taskvars['cols']
        
        # Create a grid with structured patterns
        grid = np.zeros((rows, cols), dtype=int)
        
        # Get available colors (excluding background 0)
        available_colors = list(range(1, 10))
        
        # Create column-based patterns - each column gets densely filled with 1-2 colors
        for c in range(cols):
            # Choose 1-2 colors for this column
            num_colors = random.choice([1, 2])
            column_colors = random.sample(available_colors, num_colors)
            
            # Choose a column pattern type
            pattern_type = random.choice([
                'solid_dense', 'alternating_dense', 'block_segments', 
                'gradient_fill', 'top_bottom_split', 'scattered_dense'
            ])
            
            if pattern_type == 'solid_dense':
                # Most cells filled with one color, some gaps
                color = column_colors[0]
                fill_density = random.uniform(0.6, 0.9)  # 60-90% filled
                for r in range(rows):
                    if random.random() < fill_density:
                        grid[r, c] = color
                        
            elif pattern_type == 'alternating_dense':
                # Dense alternating pattern with 1-2 colors
                if len(column_colors) == 2:
                    for r in range(rows):
                        if random.random() > 0.2:  # 80% chance to fill
                            grid[r, c] = column_colors[r % 2]
                else:
                    # Single color with gaps
                    for r in range(rows):
                        if r % 2 == 0 and random.random() > 0.3:
                            grid[r, c] = column_colors[0]
                            
            elif pattern_type == 'block_segments':
                # Column divided into blocks of different colors
                if len(column_colors) == 2:
                    mid_point = random.randint(1, rows - 1)
                    # Top segment
                    for r in range(mid_point):
                        if random.random() > 0.2:  # 80% fill rate
                            grid[r, c] = column_colors[0]
                    # Bottom segment  
                    for r in range(mid_point, rows):
                        if random.random() > 0.2:  # 80% fill rate
                            grid[r, c] = column_colors[1]
                else:
                    # Single color in segments
                    segment_start = random.randint(0, rows // 3)
                    segment_end = random.randint(segment_start + 1, rows)
                    for r in range(segment_start, segment_end):
                        if random.random() > 0.2:
                            grid[r, c] = column_colors[0]
                            
            elif pattern_type == 'gradient_fill':
                # Gradually changing density from top to bottom
                color = column_colors[0]
                for r in range(rows):
                    # Higher probability at top, lower at bottom (or vice versa)
                    direction = random.choice(['top_heavy', 'bottom_heavy'])
                    if direction == 'top_heavy':
                        prob = 0.9 - (r / rows) * 0.7  # 90% to 20%
                    else:
                        prob = 0.2 + (r / rows) * 0.7  # 20% to 90%
                    
                    if random.random() < prob:
                        grid[r, c] = color
                        
            elif pattern_type == 'top_bottom_split':
                # Dense at top and bottom, sparse in middle
                color = column_colors[0]
                second_color = column_colors[1] if len(column_colors) == 2 else color
                
                third = rows // 3
                # Top third - dense
                for r in range(third):
                    if random.random() > 0.2:  # 80% chance
                        grid[r, c] = color
                # Middle third - sparse
                for r in range(third, 2 * third):
                    if random.random() > 0.6:  # 40% chance
                        grid[r, c] = second_color
                # Bottom third - dense
                for r in range(2 * third, rows):
                    if random.random() > 0.2:  # 80% chance
                        grid[r, c] = second_color
                        
            elif pattern_type == 'scattered_dense':
                # Randomly scattered but dense filling
                fill_density = random.uniform(0.5, 0.8)  # 50-80% filled
                positions = random.sample(range(rows), int(rows * fill_density))
                
                for r in positions:
                    color = random.choice(column_colors)
                    grid[r, c] = color
        
        # Ensure every column has at least one filled cell (should already be guaranteed)
        for c in range(cols):
            if np.all(grid[:, c] == 0):  # If column is empty (unlikely with new patterns)
                # Fill a random cell in this column
                r = random.randint(0, rows - 1)
                grid[r, c] = random.choice(available_colors)
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        rows, cols = grid.shape
        
        # Create output grid twice the size
        output = np.zeros((rows * 2, cols * 2), dtype=int)
        
        # Top-left quadrant: original grid
        output[:rows, :cols] = grid
        
        # Top-right quadrant: horizontal mirror
        output[:rows, cols:] = np.fliplr(grid)
        
        # Bottom-left quadrant: vertical mirror  
        output[rows:, :cols] = np.flipud(grid)
        
        # Bottom-right quadrant: both horizontal and vertical mirror
        output[rows:, cols:] = np.flipud(np.fliplr(grid))
        
        return output


# Test code
if __name__ == "__main__":
    generator = QuadrantMirroringTaskGenerator()
    task_variables, train_test_data = generator.create_grids()
    
    print(f"Task variables: {task_variables}")
    print(f"Number of training examples: {len(train_test_data['train'])}")
    print(f"Number of test examples: {len(train_test_data['test'])}")
    
    # Visualize the results
    ARCTaskGenerator.visualize_train_test_data(train_test_data)
from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from transformation_library import find_connected_objects, GridObject, GridObjects
from input_library import Contiguity, retry, create_object, random_cell_coloring

class Taskb27ca6d3yGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of different sizes.",
            "The grid consists of scattered cells of {color('object_color')}.",
            "Most cells are individual and uniformly scattered with proper spacing.",
            "Some cells appear in pairs that are 4-way-connected to each other.",
            "Each pair is placed with enough surrounding space for a complete boundary."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is copied from the input grid.",
            "Individual scattered cells remain unchanged.",
            "For each pair of adjacent cells",
            "If the pair is not on any grid edge, create a complete boundary of {color('bound_color')} around both cells",
            "If the pair touches a grid edge, create a partial boundary only on the non-edge sides",
            "The boundary includes both orthogonal and diagonal positions around the pair",
            "Boundaries never overlap with existing cells or other boundaries"
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        """Create a grid with scattered individual cells and pairs of connected cells."""
        object_color = taskvars["object_color"]
        
        # Generate a random grid size
        height = random.randint(8, 12)  # Increased minimum size for better spacing
        width = random.randint(8, 12)
        
        grid = np.zeros((height, width), dtype=int)
        
        # Calculate balanced distribution with more spacing
        grid_area = height * width
        total_cells = grid_area // 8  # Further reduced density for better spacing
        
        # Determine ratio of pairs vs individual cells
        pair_ratio = random.uniform(0.3, 0.5)  # Reduced maximum ratio for better balance
        num_pairs = int(total_cells * pair_ratio) // 2
        num_individual = total_cells - (num_pairs * 2)
        
        # First place pairs with proper spacing
        pairs_placed = 0
        attempts = 0
        while pairs_placed < num_pairs and attempts < 150:
            r = random.randint(1, height - 2)  # Keep away from edges initially
            c = random.randint(1, width - 2)
            
            if grid[r, c] == 0:
                # Check 3x3 neighborhood to ensure space for boundary
                valid = True
                for dr in range(-2, 3):
                    for dc in range(-2, 3):
                        nr, nc = r + dr, c + dc
                        if (0 <= nr < height and 0 <= nc < width and 
                            grid[nr, nc] != 0):
                            valid = False
                            break
                
                if valid:
                    # Try to place second cell
                    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
                    random.shuffle(directions)
                    
                    for dr, dc in directions:
                        new_r, new_c = r + dr, c + dc
                        if (0 <= new_r < height and 0 <= new_c < width and 
                            grid[new_r, new_c] == 0):
                            # Check space for boundary around second cell
                            valid_second = True
                            for d2r in range(-1, 2):
                                for d2c in range(-1, 2):
                                    check_r, check_c = new_r + d2r, new_c + d2c
                                    if (0 <= check_r < height and 0 <= check_c < width and 
                                        check_r != r and check_c != c and
                                        grid[check_r, check_c] != 0):
                                        valid_second = False
                                        break
                            
                            if valid_second:
                                grid[r, c] = object_color
                                grid[new_r, new_c] = object_color
                                pairs_placed += 1
                                break
            
            attempts += 1
        
        # Then place individual cells with spacing
        placed_cells = 0
        attempts = 0
        while placed_cells < num_individual and attempts < 100:
            r = random.randint(0, height - 1)
            c = random.randint(0, width - 1)
            
            if grid[r, c] == 0:
                # Check 2x2 neighborhood for spacing
                valid = True
                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        nr, nc = r + dr, c + dc
                        if (0 <= nr < height and 0 <= nc < width and 
                            grid[nr, nc] != 0):
                            valid = False
                            break
                
                if valid:
                    grid[r, c] = object_color
                    placed_cells += 1
            
            attempts += 1
        
        return grid
    
    def transform_input(self, grid, taskvars):
        """Transform input by adding boundaries around pairs of connected cells."""
        object_color = taskvars["object_color"]
        bound_color = taskvars["bound_color"]
        
        output_grid = grid.copy()
        objects = find_connected_objects(grid, diagonal_connectivity=False, background=0, monochromatic=True)
        colored_objects = objects.with_color(object_color)
        height, width = grid.shape
        
        for obj in colored_objects:
            if len(obj) == 2:  # Handle pairs
                coords = list(obj.coords)
                boundary_positions = set()
                
                # Check if the pair is on the edge
                edges = {'left': False, 'right': False, 'top': False, 'bottom': False}
                for r, c in coords:
                    if r == 0: edges['top'] = True
                    if r == height-1: edges['bottom'] = True
                    if c == 0: edges['left'] = True
                    if c == width-1: edges['right'] = True
                
                is_edge_pair = any(edges.values())
                
                # Generate boundary positions based on edge status
                for r, c in coords:
                    if is_edge_pair:
                        # Add orthogonal directions except where there's an edge
                        if not edges['top'] and r > 0: boundary_positions.add((r-1, c))
                        if not edges['bottom'] and r < height-1: boundary_positions.add((r+1, c))
                        if not edges['left'] and c > 0: boundary_positions.add((r, c-1))
                        if not edges['right'] and c < width-1: boundary_positions.add((r, c+1))
                        
                        # Add diagonals only for non-edge corners
                        if not edges['top'] and not edges['left'] and r > 0 and c > 0:
                            boundary_positions.add((r-1, c-1))
                        if not edges['top'] and not edges['right'] and r > 0 and c < width-1:
                            boundary_positions.add((r-1, c+1))
                        if not edges['bottom'] and not edges['left'] and r < height-1 and c > 0:
                            boundary_positions.add((r+1, c-1))
                        if not edges['bottom'] and not edges['right'] and r < height-1 and c < width-1:
                            boundary_positions.add((r+1, c+1))
                    else:
                        # Full boundary for non-edge pairs
                        for dr in [-1, 0, 1]:
                            for dc in [-1, 0, 1]:
                                if dr != 0 or dc != 0:
                                    nr, nc = r + dr, c + dc
                                    if (0 <= nr < height and 0 <= nc < width and
                                        (nr, nc) not in coords):
                                        boundary_positions.add((nr, nc))
                
                # Apply boundary only where there's no existing color
                for r, c in boundary_positions:
                    if output_grid[r, c] == 0:
                        output_grid[r, c] = bound_color
        
        return output_grid
    
    def create_grids(self) -> tuple[dict, dict]:
        """Create train and test grids with consistent variables."""
        # Randomly choose colors
        object_color = random.randint(1, 9)
        bound_color = random.choice([c for c in range(1, 10) if c != object_color])
        
        taskvars = {
            "object_color": object_color,
            "bound_color": bound_color
        }
        
        # Generate 3-5 training examples with same task variables
        num_train_examples = random.randint(3, 5)
        train_examples = []
        
        for _ in range(num_train_examples):
            gridvars = {}
            
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Create test example with same task variables
        test_gridvars = {}
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

# Test code
if __name__ == "__main__":
    generator = Taskb27ca6d3yGenerator()
    taskvars, train_test_data = generator.create_grids()
    
    print("Task Variables:", taskvars)
    ARCTaskGenerator.visualize_train_test_data(train_test_data)
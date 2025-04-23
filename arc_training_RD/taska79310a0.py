from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from input_library import create_object, retry, Contiguity
from transformation_library import GridObject, GridObjects, find_connected_objects
import numpy as np
import random
from typing import Dict, List, Any, Tuple

class LetterMoveDownTaskGenerator(ARCTaskGenerator):
    def __init__(self):
        # Initialize reasoning chains
        input_reasoning_chain = [
            "Input grids are squares and can have different sizes.",
            "They contain only one object of {color('object_color')} color and empty (0) cells.",
            "The {color('object_color')} can form shapes like alphabets, specifically 'Z', 'S', 'T', 'U', 'O', 'L', 'I'.",
            "These objects can be rotated and inverted."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid object by just moving the object position to one row below.",
            "The pushed object will have a color {color('fill_color')}"
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_letter_shape(self, letter: str, size: int) -> np.ndarray:
        """Create a letter shape matrix for the given letter."""
        # Create empty matrix
        mat = np.zeros((size, size), dtype=int)
        
        # Define letter patterns
        if letter == 'Z':
            for i in range(size):
                mat[0, i] = 1  # Top horizontal
                mat[-1, i] = 1  # Bottom horizontal
                mat[i, size-1-i] = 1  # Diagonal
        elif letter == 'S':
            for i in range(size):
                mat[0, i] = 1  # Top horizontal
                mat[-1, i] = 1  # Bottom horizontal
                if i < size // 2:
                    mat[i, 0] = 1  # Left vertical (top half)
                else:
                    mat[i, -1] = 1  # Right vertical (bottom half)
        elif letter == 'T':
            for i in range(size):
                mat[0, i] = 1  # Top horizontal
            mat[1:, size // 2] = 1  # Vertical (ensure connectivity)
        elif letter == 'U':
            for i in range(size):
                mat[i, 0] = 1  # Left vertical
                mat[i, -1] = 1  # Right vertical
            mat[-1, :] = 1  # Bottom horizontal
        elif letter == 'O':
            for i in range(size):
                mat[0, i] = 1  # Top horizontal
                mat[-1, i] = 1  # Bottom horizontal
                mat[i, 0] = 1  # Left vertical
                mat[i, -1] = 1  # Right vertical
        elif letter == 'L':
            mat[:, 0] = 1  # Left vertical
            mat[-1, :] = 1  # Bottom horizontal
        elif letter == 'I':
            mat[:, size // 2] = 1  # Vertical (ensure connectivity)
        
        return mat

    def create_input(self, gridvars: Dict[str, Any]) -> np.ndarray:
        """Create an input grid with a single letter-shaped object."""
        # Get parameters from gridvars
        size = gridvars.get('size', random.randint(5, 20))
        object_color = gridvars.get('object_color', random.randint(1, 9))
        letter = gridvars.get('letter', random.choice(['Z', 'S', 'T', 'U', 'O', 'L', 'I']))
        rotation = gridvars.get('rotation', random.randint(0, 3))
        
        # Create base letter shape
        letter_size = random.randint(3, min(7, size - 2))  # Smaller than grid size
        mat = self.create_letter_shape(letter, letter_size)
        
        # Apply rotation
        mat = np.rot90(mat, k=rotation)
        
        # Create empty grid
        grid = np.zeros((size, size), dtype=int)
        
        # Find position where the object fits
        obj_height, obj_width = mat.shape
        max_row = size - obj_height
        max_col = size - obj_width
        
        if max_row < 0 or max_col < 0:
            # If object is too big for grid, scale it down
            scale_factor = min(size // obj_height, size // obj_width)
            mat = np.kron(mat, np.ones((scale_factor, scale_factor), dtype=int))
            mat = (mat > 0).astype(int)  # Ensure binary values after scaling
            obj_height, obj_width = mat.shape
            max_row = size - obj_height
            max_col = size - obj_width
        
        # Random position ensuring object can move down
        start_row = random.randint(0, max(0, max_row - 1))
        start_col = random.randint(0, max_col)
        
        # Place object in grid
        grid[start_row:start_row + obj_height, start_col:start_col + obj_width] = mat * object_color

        # Validate that the grid contains exactly one object
        objects = find_connected_objects(grid, background=0)
        if len(objects) != 1:
            raise ValueError(
                f"Grid creation failed: Expected 1 object, found {len(objects)} objects. "
                f"Grid:\n{grid}\nLetter: {letter}, Size: {size}, Rotation: {rotation}"
            )

        return grid

    def transform_input(self, input_grid: np.ndarray, gridvars: Dict[str, Any]) -> np.ndarray:
        """Transform input by moving object down one row and changing color."""
        output_grid = np.zeros_like(input_grid)
        fill_color = gridvars['fill_color']
        
        # Find the object
        objects = find_connected_objects(input_grid, background=0)
        if len(objects) != 1:
            raise ValueError("Input grid should contain exactly one object")
        
        obj = objects[0]
        
        # Create new object one row below with new color
        new_cells = {(r + 1, c, fill_color) for (r, c, _) in obj.cells}
        new_obj = GridObject(new_cells)
        
        # Paste onto output grid
        new_obj.paste(output_grid)
        
        return output_grid

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        """Create train and test grids."""
        # Choose colors ensuring they're different
        object_color = random.randint(1, 9)
        fill_color = random.choice([c for c in range(1, 10) if c != object_color])
        
        # Create 3-5 train pairs
        num_train = random.randint(3, 5)
        train_pairs = []
        
        for _ in range(num_train):
            # Randomize size and letter for each train input
            size = random.randint(5, 20)
            letter = random.choice(['Z', 'S', 'T', 'U', 'O', 'L', 'I'])
            rotation = random.randint(0, 3)
            
            gridvars = {
                'size': size,
                'object_color': object_color,
                'fill_color': fill_color,
                'letter': letter,
                'rotation': rotation
            }
            
            input_grid = self.create_input(gridvars)
            output_grid = self.transform_input(input_grid, gridvars)
            train_pairs.append(GridPair({'input': input_grid, 'output': output_grid}))
        
        # Create test pair with new letter/rotation
        test_gridvars = {
            'size': random.randint(5, 20),
            'object_color': object_color,
            'fill_color': fill_color,
            'letter': random.choice(['Z', 'S', 'T', 'U', 'L', 'I']),
            'rotation': random.randint(0, 3)
        }
        test_input = self.create_input(test_gridvars)
        test_output = self.transform_input(test_input, test_gridvars)
        
        # Wrap test grids in a GridPair
        test_pair = GridPair({'input': test_input, 'output': test_output})
        
        # Prepare gridvars for the reasoning chain
        reasoning_gridvars = {
            'object_color': object_color,
            'fill_color': fill_color
        }
        
        return reasoning_gridvars, TrainTestData(train_pairs, [test_pair])

if __name__ == "__main__":
    generator = LetterMoveDownTaskGenerator()
    gridvars, train_test_data = generator.create_grids()
    ARCTaskGenerator.visualize_train_test_data(train_test_data)
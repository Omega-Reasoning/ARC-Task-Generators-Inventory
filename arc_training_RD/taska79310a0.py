from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from input_library import create_object, retry, Contiguity
from transformation_library import GridObject, GridObjects, find_connected_objects
import numpy as np
import random

class Taska79310a0Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are squares and can have different sizes.",
            "They contain only one object of {color(\"object_color\")} color and empty (0) cells.",
            "The object can be one of four types:",
            "1. Letter \"I\" (1x3 horizontal line) - [1,1,1]",
            "2. Letter \"T\" (3x2 shape with top bar) - [[1,1,1],[0,1,0]]",
            "3. Letter \"O\" (2x2 solid square) - [[1,1],[1,1]]",
            "4. Single cell (1x1) - [1]"
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid object by just moving the object position to one row below.",
            "The pushed object will have color {color(\"fill_color\")}"
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_letter_shape(self, letter: str) -> np.ndarray:
        """Create a letter shape based on the specified type."""
        if letter == 'I':
            return np.array([[1, 1, 1]])
        elif letter == 'T':
            return np.array([[1, 1, 1],
                           [0, 1, 0]])
        elif letter == 'O':
            return np.array([[1, 1],
                           [1, 1]])
        else:  # Single cell
            return np.array([[1]])

    def create_input(self, gridvars) -> np.ndarray:
        """Create an input grid with a single letter object."""
        size = gridvars.get('size', random.randint(5, 20))
        object_color = gridvars.get('object_color', random.randint(1, 9))
        letter = gridvars.get('letter', random.choice(['I', 'T', 'O', 'single']))
        rotation = gridvars.get('rotation', random.randint(0, 3))

        # Create base letter shape
        mat = self.create_letter_shape(letter)
        
        # Apply rotation
        mat = np.rot90(mat, k=rotation)

        # Create empty grid
        grid = np.zeros((size, size), dtype=int)
        
        # Find position where the object fits
        obj_height, obj_width = mat.shape
        max_row = size - obj_height - 1  # -1 to ensure we can move down
        max_col = size - obj_width
        
        if max_row < 0 or max_col < 0:
            raise ValueError(f"Grid too small for object: {size}x{size} vs {obj_height}x{obj_width}")
        
        # Random position ensuring object can move down
        start_row = random.randint(0, max_row)
        start_col = random.randint(0, max_col)
        
        # Place object in grid
        grid[start_row:start_row+obj_height, start_col:start_col+obj_width] = mat * object_color

        return grid

    def transform_input(self, input_grid: np.ndarray, gridvars) -> np.ndarray:
        """Transform input by moving object down one row."""
        objects = find_connected_objects(input_grid, background=0)
        if len(objects) != 1:
            raise ValueError(f"Expected exactly one object, found {len(objects)}")
        
        output_grid = np.zeros_like(input_grid)
        obj = objects[0]
        
        # Create new object one row below with new color
        new_cells = {(r + 1, c, gridvars['fill_color']) 
                    for r, c, _ in obj.cells}
        
        # Create and paste the moved object
        moved_obj = GridObject(new_cells)
        moved_obj.paste(output_grid)
        
        return output_grid

    def create_grids(self):
        """Create train and test grids."""
        # Initialize colors ensuring they are different
        object_color = random.randint(1, 9)
        fill_color = random.choice([c for c in range(1, 10) if c != object_color])
        
        reasoning_gridvars = {
            'object_color': object_color,
            'fill_color': fill_color
        }
        
        # Create train pairs
        num_train = random.randint(3, 5)
        train_pairs = []
        
        # Ensure all letter types appear in training
        letters = ['I', 'T', 'O', 'single']
        random.shuffle(letters)
        
        for i in range(num_train):
            letter = letters[i] if i < len(letters) else random.choice(letters)
            gridvars = {
                'size': random.randint(5, 20),
                'object_color': object_color,
                'fill_color': fill_color,
                'letter': letter,
                'rotation': random.randint(0, 3)
            }
            
            input_grid = self.create_input(gridvars)
            output_grid = self.transform_input(input_grid, gridvars)
            train_pairs.append({'input': input_grid, 'output': output_grid})
        
        # Create test pair with random letter and rotation
        test_gridvars = {
            'size': random.randint(5, 20),
            'object_color': object_color,
            'fill_color': fill_color,
            'letter': random.choice(letters),
            'rotation': random.randint(0, 3)
        }
        
        test_input = self.create_input(test_gridvars)
        test_output = self.transform_input(test_input, test_gridvars)
        test_pair = {'input': test_input, 'output': test_output}
        
        return reasoning_gridvars, TrainTestData({"train": train_pairs, "test": [test_pair]})
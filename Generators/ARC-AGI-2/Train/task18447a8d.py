from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.transformation_library import find_connected_objects, GridObject, GridObjects
from Framework.input_library import retry, create_object, Contiguity
import numpy as np
import random
from typing import Dict, List, Any, Tuple, Set

class Task18447a8d(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['rows']} × {vars['cols']}.",
            "Each input grid contains {((vars['rows']-1)//4)*2} objects formed by 4-way connected cells, while the remaining cells are filled with {color('background_color')}.",
            "The {((vars['rows']-1)//4)*2} objects are first created as {(vars['rows']-1)//4} rectangles, each occupying 3 rows in height and {(vars['cols']-1)//2} columns in width.",
            "Each rectangle is divided into two parts: the left half is made of {color('object_color')}, while the right half can be a different color that varies across objects.",
            "The objects do not need to be perfectly halved with straight vertical divisions; instead, they may have irregular, puzzle-like separations. When combined, the two halves should still form a complete rectangle.",
            "All objects must include the {color('object_color')} portion, which should have a different shape in each rectangle.",
            "Once the rectangles are created, they are separated into two colored parts. The first {color('object_color')} object is placed at position (1,0). Each subsequent {color('object_color')} object is placed below the previous one, leaving exactly one row of {color('background_color')} in between.",
            "In this way, all {color('object_color')} objects are aligned in the left half of the grid, with one {color('background_color')} row between them. The first and last rows of the grid are also filled with {color('background_color')}.",
            "The remaining {(vars['rows']-1)//4} halves of the rectangles are placed in the right half of the grid, starting from position (1, {vars['cols']-1}). Each subsequent half is placed below the previous one, with exactly one row of {color('background_color')} left in between. The first row, the last row, and the rows separating each object are also filled with {color('background_color')}.",
            "After the rectangles are split into two halves, the placement must ensure that each {color('object_color')} half is aligned with (or parallel to) a half of a different color.",
            "However, a {color('object_color')} half must never be aligned with the other half from the same original rectangle."
        ]
        
        transformation_reasoning_chain = [
            "The output grids are constructed by copying the input grids and identifying {((vars['rows']-1)//4) * 2} objects, which are then paired to form complete rectangles.",
            "The process involves connecting the puzzle-like pieces so that they form rectangles. However, the same halves of rectangles must not be placed parallel to each other, so all possible pairings must be checked to find valid connections.",
            "Each left-half is connected with one right-half to complete a rectangle.",
            "The {color('object_color')} halves must remain fixed in their original positions. Only the other halves (non-{color('object_color')}) may be moved to align and complete the rectangles."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        rows, cols = taskvars['rows'], taskvars['cols']
        grid = np.full((rows, cols), taskvars['background_color'], dtype=int)
        
        # Number of rectangles to create
        num_rectangles = (rows - 1) // 4
        rect_height = 3
        rect_width = (cols - 1) // 2
        
        # Generate puzzle piece pairs for each rectangle
        puzzle_pairs = []
        available_colors = [c for c in range(1, 10) if c not in [taskvars['object_color'], taskvars['background_color']]]
        
        for i in range(num_rectangles):
            # Create a rectangle and split it into puzzle pieces
            right_color = random.choice(available_colors)
            left_piece, right_piece = self._create_puzzle_pair(rect_height, rect_width, 
                                                             taskvars['object_color'], 
                                                             right_color)
            puzzle_pairs.append((left_piece, right_piece))
            # Remove used color to ensure uniqueness
            available_colors.remove(right_color)
        
        # Place left pieces (object_color) on the left side
        for i, (left_piece, _) in enumerate(puzzle_pairs):
            start_row = 1 + i * 4  # Leave one row between objects
            self._place_piece(grid, left_piece, start_row, 0)
        
        # Shuffle and place right pieces on the right side
        right_pieces = [pair[1] for pair in puzzle_pairs]
        random.shuffle(right_pieces)
        
        for i, right_piece in enumerate(right_pieces):
            start_row = 1 + i * 4  # Leave one row between objects
            start_col = cols - rect_width
            self._place_piece(grid, right_piece, start_row, start_col)
        
        return grid
    
    def _create_puzzle_pair(self, height: int, width: int, left_color: int, right_color: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create two puzzle pieces that fit together to form a rectangle"""
        # Create the base rectangle
        rectangle = np.zeros((height, width), dtype=int)
        
        # Create an irregular vertical division line
        division = []
        mid_col = width // 2
        
        for row in range(height):
            # Add some randomness to the division line
            offset = random.randint(-1, 1)
            div_col = max(0, min(width - 2, mid_col + offset))
            division.append(div_col)
        
        # Create left and right pieces
        left_piece = np.zeros((height, width), dtype=int)
        right_piece = np.zeros((height, width), dtype=int)
        
        for row in range(height):
            div_col = division[row]
            # Fill left side
            for col in range(div_col + 1):
                left_piece[row, col] = left_color
            # Fill right side  
            for col in range(div_col + 1, width):
                right_piece[row, col] = right_color
        
        return left_piece, right_piece
    
    def _place_piece(self, grid: np.ndarray, piece: np.ndarray, start_row: int, start_col: int):
        """Place a puzzle piece on the grid at the specified position"""
        piece_height, piece_width = piece.shape
        for r in range(piece_height):
            for c in range(piece_width):
                if piece[r, c] != 0:  # Only place non-background cells
                    if 0 <= start_row + r < grid.shape[0] and 0 <= start_col + c < grid.shape[1]:
                        grid[start_row + r, start_col + c] = piece[r, c]
    
    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        output_grid = grid.copy()
        
        # Calculate rectangle dimensions
        rect_height = 3
        rect_width = (taskvars['cols'] - 1) // 2
        
        # Find all objects in the grid
        objects = find_connected_objects(grid, diagonal_connectivity=False, 
                                       background=taskvars['background_color'], 
                                       monochromatic=True)
        
        # Separate object_color pieces (left side) and other pieces (right side)
        left_pieces = []
        right_pieces = []
        
        for obj in objects:
            if taskvars['object_color'] in obj.colors:
                left_pieces.append(obj)
            else:
                right_pieces.append(obj)
        
        # Sort pieces by their row positions
        left_pieces.sort(key=lambda obj: min(r for r, c, _ in obj.cells))
        right_pieces.sort(key=lambda obj: min(r for r, c, _ in obj.cells))
        
        # Clear right pieces from their current positions
        for obj in right_pieces:
            obj.cut(output_grid, taskvars['background_color'])
        
        # Keep track of used right pieces
        used_right_pieces = set()
        
        # For each left piece, find the matching right piece
        for left_piece in left_pieces:
            # Get the bounding rectangle for this left piece
            left_min_row = min(r for r, c, _ in left_piece.cells)
            left_min_col = min(c for r, c, _ in left_piece.cells)
            
            # Create the full 3 × rect_width rectangle that this piece should belong to
            target_rectangle = set()
            for r in range(left_min_row, left_min_row + rect_height):
                for c in range(left_min_col, left_min_col + rect_width):
                    target_rectangle.add((r, c))
            
            # Find what cells are occupied by the left piece
            left_occupied = set((r, c) for r, c, _ in left_piece.cells)
            
            # Find the missing cells (what the right piece should fill)
            missing_cells = target_rectangle - left_occupied
            
            # Try to match this missing pattern with each unused right piece
            best_match = None
            for i, right_piece in enumerate(right_pieces):
                if i in used_right_pieces:
                    continue
                
                # Get the right piece's current cells relative to its top-left corner
                right_cells = [(r, c) for r, c, _ in right_piece.cells]
                right_min_row = min(r for r, c in right_cells)
                right_min_col = min(c for r, c in right_cells)
                
                # Normalize right piece cells to start from (0,0)
                normalized_right = set((r - right_min_row, c - right_min_col) for r, c in right_cells)
                
                # Normalize missing cells to start from (0,0)
                missing_min_row = min(r for r, c in missing_cells) if missing_cells else 0
                missing_min_col = min(c for r, c in missing_cells) if missing_cells else 0
                normalized_missing = set((r - missing_min_row, c - missing_min_col) for r, c in missing_cells)
                
                # Check if the shapes match exactly
                if normalized_right == normalized_missing:
                    best_match = i
                    break
            
            # If we found a matching right piece, fill the missing cells with its color
            if best_match is not None:
                right_piece = right_pieces[best_match]
                right_color = list(right_piece.colors)[0]  # Get the color of the right piece
                
                # Fill the missing cells with the right piece's color
                for r, c in missing_cells:
                    if 0 <= r < output_grid.shape[0] and 0 <= c < output_grid.shape[1]:
                        output_grid[r, c] = right_color
                
                used_right_pieces.add(best_match)
        
        return output_grid
    
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # rows and cols as you had them
        rows_choices = [9, 13, 17, 21, 25, 29]
        cols_choices = [i for i in range(11, 31, 2)]  # odd 11..29
        
        # Choose object/background from 1..9, distinct
        object_color = random.randint(1, 9)
        background_candidates = [c for c in range(1, 10) if c != object_color]
        background_color = random.choice(background_candidates)
        
        taskvars = {
            'rows': random.choice(rows_choices),
            'cols': random.choice(cols_choices),
            'object_color': object_color,
            'background_color': background_color
        }
        
        # Ensure we have enough unique right-half colors
        num_rectangles = (taskvars['rows'] - 1) // 4
        available_colors = [c for c in range(1, 10)
                            if c not in (taskvars['object_color'], taskvars['background_color'])]
        if len(available_colors) < num_rectangles:
            # Reduce rows so num_rectangles fits the available unique colors (max 7)
            taskvars['rows'] = 9  # -> num_rectangles = 2
        
        # Create train/test
        num_train = random.randint(3, 5)
        train_test_data = self.create_grids_default(num_train, 1, taskvars)
        return taskvars, train_test_data



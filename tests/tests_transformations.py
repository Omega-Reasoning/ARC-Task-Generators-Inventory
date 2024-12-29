import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformation_library import GridObject, find_connected_objects
from utilities import visualize_matrix

class TestTransformations(unittest.TestCase):
    def assertMatrixEqual(self, actual, expected, msg=None):
        """Helper method to compare matrices with visual output on failure"""
        if not np.array_equal(actual, expected):
            message = f"\n{msg if msg else 'Matrices are not equal'}\n"
            message += "\nActual matrix:\n"
            message += visualize_matrix(actual)
            message += "\nExpected matrix:\n"
            message += visualize_matrix(expected)
            self.fail(message)

    def test_rotate_and_translate_quadrant(self):
        # build a 4x4 matrix with some colors in the top-left quadrant
        grid = np.zeros((4, 4), dtype=int)
        grid[0:2, 0:2] = [[1, 2],
                         [2, 3]]
        initial_grid = grid.copy()
        
        print("Initial grid:")
        print(visualize_matrix(initial_grid))

        # rotate quadrant and move it to the top-right
        top_left = GridObject.from_grid(grid, {(0,0), (0,1), (1,0), (1,1)})
        top_left.rotate(1).translate(0,2).paste(grid)
               
        print("Grid after rotation:")
        print(visualize_matrix(grid))

        # Assert
        # Check that original quadrant is unchanged
        self.assertMatrixEqual(
            grid[0:2, 0:2],
            initial_grid[0:2, 0:2],
            "Original top-left quadrant was modified"
        )
        
        # Check that rotated quadrant is correct
        expected_top_right = np.array([[2, 3],
                                     [1, 2]])
        self.assertMatrixEqual(
            grid[0:2, 2:4],
            expected_top_right,
            "Rotated pattern in top-right quadrant is incorrect"
        )
        
        # Check that rest of grid is empty
        self.assertMatrixEqual(
            grid[2:, :],
            np.zeros((2, 4)),
            "Bottom half of grid should be empty"
        )

    def test_cut_translate_paste(self):
        """Test cutting an object, translating it, and pasting it back."""
        # Arrange
        grid = np.zeros((4, 4), dtype=int)
        grid[0:2, 0:2] = [[0, 1], [2, 3]]
        
        # Create object from colored cells (excluding the 0 at position 0,0)
        obj = GridObject.from_grid(grid, {(0,1), (1,0), (1,1)})
        
        # Act
        obj.cut(grid).translate(1,2).paste(grid)
        
        # Assert
        expected_grid = np.zeros((4, 4), dtype=int)
        expected_grid[1:3, 2:4] = [[0, 1],
                                [2, 3]]
        
        self.assertMatrixEqual(
            grid,
            expected_grid,
            "Grid after cut-translate-paste sequence is incorrect"
        )
        
        # Verify object's final position
        expected_cells = {
            (1, 3, 1),  # Original (0,1) cell moved to (1,3)
            (2, 2, 2),  # Original (1,0) cell moved to (2,2)
            (2, 3, 3)   # Original (1,1) cell moved to (2,3)
        }
        
        self.assertEqual(
            obj.cells,
            expected_cells,
            "Object's cells are not in expected positions after translation"
        )

    def test_move_multiple_objects(self):
        """Test finding gray objects and moving them using chained operations."""
        # Arrange
        grid = np.zeros((6, 6), dtype=int)
        # Set up red shape (color 2)
        grid[0:2, 0:2] = 2
        # Set up two gray shapes (color 5)
        grid[1:3, 2:4] = 5  # First gray shape
        grid[4:6, 0:2] = 5  # Second gray shape
        initial_grid = grid.copy()

        # print("input:")
        # print(visualize_matrix(initial_grid))

        # Act
        # Find all objects, filter gray ones, cut them, translate right 2 spaces, and paste back
        objects = find_connected_objects(grid)
        result = objects.with_color(5).cut(grid).translate(0, 2).paste(grid)

        # Assert
        expected_grid = np.zeros((6, 6), dtype=int)
        # Red shape stays in place
        expected_grid[0:2, 0:2] = 2
        # Gray shapes moved 2 spaces right
        expected_grid[1:3, 4:6] = 5 # First gray shape
        expected_grid[4:6, 2:4] = 5 # Second gray shape

        self.assertMatrixEqual(
            grid,
            expected_grid,
            "Grid after moving gray objects is incorrect"
        )

        # Verify that red object was untouched
        red_area = grid[0:2, 0:2]
        self.assertTrue(
            np.array_equal(red_area, initial_grid[0:2, 0:2]),
            "Red object should not have been modified"
        )

        # Verify we got two objects back from the chain
        self.assertEqual(
            len(result), 2,
            "Should have found and moved two gray objects"
        )

if __name__ == '__main__':
    unittest.main()

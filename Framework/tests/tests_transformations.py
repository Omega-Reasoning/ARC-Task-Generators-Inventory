import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformation_library import GridObject, find_connected_objects, get_objects_from_raster
from utilities import visualize_grid, visualize_grid_object

class TestTransformations(unittest.TestCase):
    def assertMatrixEqual(self, actual, expected, msg=None):
        """Helper method to compare matrices with visual output on failure"""
        if not np.array_equal(actual, expected):
            message = f"\n{msg if msg else 'Matrices are not equal'}\n"
            message += "\nActual matrix:\n"
            message += visualize_grid(actual)
            message += "\nExpected matrix:\n"
            message += visualize_grid(expected)
            self.fail(message)

    def test_rotate_and_translate_quadrant(self):
        # build a 4x4 matrix with some colors in the top-left quadrant
        grid = np.zeros((4, 4), dtype=int)
        grid[0:2, 0:2] = [[1, 2],
                         [2, 3]]
        initial_grid = grid.copy()
        
        print("Initial grid:")
        print(visualize_grid(initial_grid))

        # rotate quadrant and move it to the top-right
        top_left = GridObject.from_grid(grid, {(0,0), (0,1), (1,0), (1,1)})
        top_left.rotate(1).translate(0,2).paste(grid)
               
        print("Grid after rotation:")
        print(visualize_grid(grid))

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

    def test_get_objects_from_raster(self):
        """Test extracting objects from a raster grid with delimiters."""
        # Create the example grid
        grid = np.array([
            [0,0,0,9,6,6,6,9,0,0,0,9,6,6,6],
            [0,0,0,9,6,6,6,9,0,0,0,9,6,6,6],
            [0,0,0,9,6,6,6,9,0,0,0,9,6,6,6],
            [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
            [6,6,6,9,0,0,0,9,6,6,6,9,0,0,0],
            [6,6,6,9,0,0,0,9,6,6,6,9,0,0,0],
            [6,6,6,9,0,0,0,9,6,6,6,9,0,0,0],
            [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
            [3,3,3,9,0,0,0,9,0,0,0,9,3,3,3],
            [3,3,3,9,0,0,0,9,0,0,0,9,3,3,3],
            [3,3,3,9,0,0,0,9,0,0,0,9,3,3,3]
        ])

        print("\nInput grid:")
        print(visualize_grid(grid))

        # Get the objects
        objects = get_objects_from_raster(grid, subgrid_rows=3, subgrid_cols=3, initial_rows=3, initial_cols=3)

        # Print all extracted objects
        print("\nExtracted objects by position:")
        for row_idx, row in enumerate(objects):
            print(f"\nRow {row_idx}:")
            for col_idx, obj in enumerate(row):
                print(f"\nPosition ({row_idx}, {col_idx}):")
                # Create a 3x3 grid from the object's cells
                subgrid = np.zeros((3, 3), dtype=int)
                if obj.cells:  # Only try to reconstruct if there are cells
                    # Get minimum coordinates from the cells
                    min_r = min(r for r, _, _ in obj.cells)
                    min_c = min(c for _, c, _ in obj.cells)
                    # Fill the subgrid relative to minimum coordinates
                    for r, c, color in obj.cells:
                        subgrid[r - min_r, c - min_c] = color
                print(visualize_grid(subgrid))

        # Test basic structure
        self.assertEqual(len(objects), 3, "Should have 3 rows of objects")
        self.assertEqual(len(objects[0]), 4, "Should have 4 columns of objects")

        # Test specific subgrids by reconstructing them
        # First row, second column (should be all 6's)
        subgrid = np.zeros((3, 3), dtype=int)
        obj = objects[0][1]
        if obj.cells:
            min_r = min(r for r, _, _ in obj.cells)
            min_c = min(c for _, c, _ in obj.cells)
            for r, c, color in obj.cells:
                subgrid[r - min_r, c - min_c] = color
        expected = np.full((3, 3), 6)
        self.assertMatrixEqual(subgrid, expected, "First row, second column should be all 6's")

        # Last row, first column (should be all 3's)
        subgrid = np.zeros((3, 3), dtype=int)
        obj = objects[2][0]
        if obj.cells:
            min_r = min(r for r, _, _ in obj.cells)
            min_c = min(c for _, c, _ in obj.cells)
            for r, c, color in obj.cells:
                subgrid[r - min_r, c - min_c] = color
        expected = np.full((3, 3), 3)
        self.assertMatrixEqual(subgrid, expected, "Last row, first column should be all 3's")

        # Middle row, second column (should be all 0's)
        subgrid = np.zeros((3, 3), dtype=int)
        obj = objects[1][1]
        if obj.cells:
            min_r = min(r for r, _, _ in obj.cells)
            min_c = min(c for _, c, _ in obj.cells)
            for r, c, color in obj.cells:
                subgrid[r - min_r, c - min_c] = color
        expected = np.zeros((3, 3))
        self.assertMatrixEqual(subgrid, expected, "Middle row, second column should be all 0's")

        # Verify all positions match expected colors
        expected_colors = [
            [0, 6, 0, 6],  # First row
            [6, 0, 6, 0],  # Second row
            [3, 0, 0, 3]   # Third row
        ]

        for row_idx, row in enumerate(objects):
            for col_idx, obj in enumerate(row):
                expected = expected_colors[row_idx][col_idx]
                colors = obj.colors - {0}
                if expected == 0:
                    self.assertEqual(len(colors), 0, 
                        f"Object at ({row_idx},{col_idx}) should be empty")
                else:
                    self.assertEqual(len(colors), 1, 
                        f"Object at ({row_idx},{col_idx}) should have exactly one non-zero color")
                    self.assertIn(expected, colors, 
                        f"Object at ({row_idx},{col_idx}) should have color {expected}")

    def test_get_objects_from_raster_with_offsets_and_delimiters(self):
        """Test extracting objects from a raster grid with offsets and delimiters inside cells."""
        grid = np.array([
            [0,8,0,0,8,0,0,8,0],
            [0,8,0,0,8,1,1,8,0],
            [8,8,8,8,8,8,8,8,8],
            [1,8,0,0,8,0,0,8,2],
            [1,8,0,8,8,0,0,8,5],
            [8,8,8,8,8,8,8,8,8]
        ])

        print("\nInput grid:")
        print(visualize_grid(grid))

        objects = get_objects_from_raster(grid, 
                                        subgrid_rows=2, 
                                        subgrid_cols=2,
                                        initial_rows=2, 
                                        initial_cols=1)

        print("\nExtracted objects by position:")
        for row_idx, row in enumerate(objects):
            print(f"\nRow {row_idx}:")
            for col_idx, obj in enumerate(row):
                print(f"\nPosition ({row_idx}, {col_idx}):")
                print(visualize_grid_object(obj))

        # Test basic structure
        self.assertEqual(len(objects), 2, "Should have 2 rows of objects")
        self.assertEqual(len(objects[0]), 4, "Should have 4 columns of objects")

        # Expected patterns for each position
        expected_cells = [
            # First row
            [set([(0,0,0), (1,0,0)]),                    # First column (1x2)
            set([(0,2,0), (0,3,0), (1,2,0), (1,3,0)]), # Second column (2x2)
            set([(0,5,0), (0,6,0), (1,5,1), (1,6,1)]), # Third column (2x2)
            set([(0,8,0), (1,8,0)])],                   # Fourth column (1x2)
            # Second row
            [set([(3,0,1), (4,0,1)]),                    # First column (1x2)
            set([(3,2,0), (3,3,0), (4,2,0), (4,3,8)]), # Second column (2x2)
            set([(3,5,0), (3,6,0), (4,5,0), (4,6,0)]), # Third column (2x2)
            set([(3,8,2), (4,8,5)])]                    # Fourth column (1x2)
        ]

        # Test each position against expected cells
        for row_idx, row in enumerate(objects):
            for col_idx, obj in enumerate(row):
                self.assertEqual(
                    obj.cells,
                    expected_cells[row_idx][col_idx],
                    f"Cells at position ({row_idx},{col_idx}) do not match expected pattern"
                )

    def test_get_objects_from_raster_with_initial_delimiters(self):
        """Test extracting objects from a raster grid where first row/column are delimiters."""
        grid = np.array([
            [8,8,8,8,8,8,8,8,8],  # First row is delimiter
            [8,0,0,8,0,0,8,0,0],
            [8,0,0,8,1,1,8,0,0],
            [8,8,8,8,8,8,8,8,8],
            [8,1,0,8,0,0,8,2,8],
            [8,1,8,8,0,0,8,5,1],
            [8,8,8,8,8,8,8,8,8]
        ])

        print("\nInput grid:")
        print(visualize_grid(grid))

        objects = get_objects_from_raster(grid, 
                                        subgrid_rows=2, 
                                        subgrid_cols=2,
                                        initial_rows=0,  # First row is delimiter
                                        initial_cols=0)  # First column is delimiter

        print("\nExtracted objects by position:")
        for row_idx, row in enumerate(objects):
            print(f"\nRow {row_idx}:")
            for col_idx, obj in enumerate(row):
                print(f"\nPosition ({row_idx}, {col_idx}):")
                print(visualize_grid_object(obj))

        # Test basic structure
        self.assertEqual(len(objects), 2, "Should have 2 rows of objects")
        self.assertEqual(len(objects[0]), 3, "Should have 3 columns of objects")

        # Expected patterns for each position
        expected_cells = [
            # First row
            [set([(1,1,0), (1,2,0), (2,1,0), (2,2,0)]),  # First column (2x2)
            set([(1,4,0), (1,5,0), (2,4,1), (2,5,1)]),  # Second column (2x2)
            set([(1,7,0), (1,8,0), (2,7,0), (2,8,0)])], # Third column (2x2)
            # Second row
            [set([(4,1,1), (4,2,0), (5,1,1), (5,2,8)]),  # First column (2x2)
            set([(4,4,0), (4,5,0), (5,4,0), (5,5,0)]),  # Second column (2x2)
            set([(4,7,2), (4,8,8), (5,7,5), (5,8,1)])]  # Third column (2x2)
        ]

        # Test each position against expected cells
        for row_idx, row in enumerate(objects):
            for col_idx, obj in enumerate(row):
                self.assertEqual(
                    obj.cells,
                    expected_cells[row_idx][col_idx],
                    f"Cells at position ({row_idx},{col_idx}) do not match expected pattern"
                )

if __name__ == '__main__':
    unittest.main()

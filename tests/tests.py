import numpy as np
import unittest
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformation_library import flood_fill
from input_library import create_object, Contiguity

class TestFloodFill(unittest.TestCase):
    def setUp(self):
        # Create sample test matrices
        self.simple_matrix = np.array([
            [1, 1, 1, 1, 1],
            [1, 1, 0, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 0, 1, 1],
            [1, 1, 1, 1, 1]
        ])
        
        self.empty_matrix = np.array([])
        
        self.single_value_matrix = np.ones((3, 3))

    def test_basic_flood_fill(self):
        result = flood_fill(self.simple_matrix.copy(), (2, 2), 2)
        expected = np.array([
            [1, 1, 1, 1, 1],
            [1, 1, 2, 1, 1],
            [1, 2, 2, 2, 1],
            [1, 1, 2, 1, 1],
            [1, 1, 1, 1, 1]
        ])
        np.testing.assert_array_equal(result, expected)

    def test_edge_flood_fill(self):
        result = flood_fill(self.simple_matrix.copy(), (0, 0), 2)
        expected = np.array([
            [2, 2, 2, 2, 2],
            [2, 2, 0, 2, 2],
            [2, 0, 0, 0, 2],
            [2, 2, 0, 2, 2],
            [2, 2, 2, 2, 2]
        ])
        np.testing.assert_array_equal(result, expected)

    def test_out_of_bounds(self):
        result = flood_fill(self.simple_matrix.copy(), (5, 5), 2)
        np.testing.assert_array_equal(result, self.simple_matrix)

    def test_same_value_fill(self):
        result = flood_fill(self.simple_matrix.copy(), (2, 2), 0)
        np.testing.assert_array_equal(result, self.simple_matrix)

    def test_single_value_matrix(self):
        result = flood_fill(self.single_value_matrix.copy(), (1, 1), 2)
        expected = np.full((3, 3), 2)
        np.testing.assert_array_equal(result, expected)

if __name__ == '__main__':
    unittest.main()
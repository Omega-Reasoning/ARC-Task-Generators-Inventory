import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformation_library import get_objects
from utilities import visualize_grid, visualize_objects

import unittest
import numpy as np

class TestGetObjects(unittest.TestCase):
    def assertObjectsEqual(self, matrix, actual_objects, expected_objects):
        """Helper method to compare objects with detailed error messages"""
        if len(actual_objects) != len(expected_objects) or not all(obj in expected_objects for obj in actual_objects):
            message = "\nMatrix:\n"
            message += visualize_grid(matrix)
            message += "\n\nFound objects:\n"
            message += visualize_objects(matrix, actual_objects)
            message += "\nExpected objects:\n"
            message += visualize_objects(matrix, expected_objects)
            self.fail(message)

    def test_basic(self):
        matrix = np.array([
            [1, 1, 0, 0],
            [1, 0, 0, 2],
            [0, 0, 2, 2]
        ])
        
        objects = get_objects(matrix)
        expected = [{(0,0), (0,1), (1,0)}, {(1,3), (2,2), (2,3)}]
        self.assertObjectsEqual(matrix, objects, expected)

    def test_diagonal_connectivity(self):
        matrix = np.array([
            [1, 0, 2],
            [0, 1, 0],
            [1, 0, 2]
        ])
        
        # Without diagonal connectivity
        objects = get_objects(matrix)
        expected = [{(0,0)}, {(1,1)}, {(2,0)}, {(0,2)}, {(2,2)}]
        self.assertObjectsEqual(matrix, objects, expected)
        
        # With diagonal connectivity
        objects_diagonal = get_objects(matrix, diagonal_connectivity=True)
        expected_diagonal = [{(0,0), (1,1), (2,0)}, {(0,2), (2,2)}]
        self.assertObjectsEqual(matrix, objects_diagonal, expected_diagonal)

    def test_specific_color(self):
        matrix = np.array([
            [1, 1, 2],
            [0, 1, 2],
            [2, 2, 2]
        ])
        
        objects = get_objects(matrix, color=2)
        expected = [{(0,2), (1,2), (2,0), (2,1), (2,2)}]
        self.assertObjectsEqual(matrix, objects, expected)

    def test_custom_background(self):
        matrix = np.array([
            [5, 1, 5, 2],
            [1, 1, 5, 5],
            [5, 5, 5, 2]
        ])
        
        objects = get_objects(matrix, background=5)
        expected = [{(0,1), (1,0), (1,1)}, {(0,3)}, {(2,3)}]
        self.assertObjectsEqual(matrix, objects, expected)

    def test_empty(self):
        matrix = np.zeros((3, 3))
        objects = get_objects(matrix)
        expected = []
        self.assertObjectsEqual(matrix, objects, expected)

    def test_single_cell(self):
        matrix = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ])
        
        objects = get_objects(matrix)
        expected = [{(1,1)}]
        self.assertObjectsEqual(matrix, objects, expected)

if __name__ == '__main__':
    unittest.main()

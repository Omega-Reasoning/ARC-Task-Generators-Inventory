import numpy as np
import unittest
import os
import sys
from scipy.ndimage import label

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from input_library import create_object, enforce_object_width, enforce_object_height, Contiguity

class TestCreateObject(unittest.TestCase):
    """Test cases for object creation functions in input_library."""

    def setUp(self):
        """Initialize test case with empty object."""
        self.object = None
    
    def format_object(self, matrix):
        """Format matrix for pretty printing.
        
        Args:
            matrix: The matrix to format, or None
            
        Returns:
            str: Formatted string representation of the matrix
        """
        if matrix is None:
            return "No object was created"
        return "\n" + "\n".join(
            "[" + " ".join(f"{cell:2d}" for cell in row) + "]" 
            for row in matrix
        ) + "\n"
    
    def tearDown(self):
        """Print failed test objects for debugging."""
        result = self._outcome.result
        test_name = self._testMethodName
        
        for failure in result.failures:
            if test_name in str(failure[0]):
                if self.object is not None:
                    sys.stderr.write("\n" + "="*50 + "\n")
                    sys.stderr.write(f"FAILED TEST OBJECT for {test_name}:\n")
                    sys.stderr.write(self.format_object(self.object))
                    sys.stderr.write("="*50 + "\n")
                    sys.stderr.flush()

    def test_basic_creation(self):
        """Test basic object creation with single color."""
        self.object = create_object(3, 3, color_palette=1)
        self.assertEqual(self.object.shape, (3, 3))
        self.assertTrue(np.all(np.isin(self.object, [0, 1])))

    def test_color_palette(self):
        """Test that only colors from palette are used."""
        self.object = create_object(4, 4, color_palette=[1, 2, 3])
        self.assertTrue(np.all(np.isin(self.object, [0, 1, 2, 3])))

    def test_background_color(self):
        """Test custom background color."""
        self.object = create_object(3, 3, color_palette=1, background=9)
        self.assertTrue(np.all(np.isin(self.object, [9, 1])))

    def test_enforce_object_width(self):
        """Test that each row has at least one colored cell when width is enforced."""
        base_generator = lambda: create_object(4, 4, color_palette=1)
        self.object = enforce_object_width(base_generator)
        self.assertTrue(all(np.any(row != 0) for row in self.object))

    def test_enforce_object_height(self):
        """Test that each column has at least one colored cell when height is enforced."""
        base_generator = lambda: create_object(4, 4, color_palette=1)
        self.object = enforce_object_height(base_generator)
        self.assertTrue(all(np.any(col != 0) for col in self.object.T))

    def test_enforce_both(self):
        """Test that both width and height constraints are satisfied."""
        # Create a base generator function
        base_generator = lambda: create_object(4, 4, color_palette=1)
        
        # Create a new generator that wraps the width-enforced result in a generator
        width_enforced_generator = lambda: enforce_object_width(base_generator)
        
        # Apply height enforcement
        self.object = enforce_object_height(width_enforced_generator)
        
        self.assertTrue(all(np.any(row != 0) for row in self.object))
        self.assertTrue(all(np.any(col != 0) for col in self.object.T))

    def test_small_matrix(self):
        """Test with minimal 2x2 matrix."""
        # Create a base generator function
        base_generator = lambda: create_object(2, 2, color_palette=1)
        
        # Create a new generator that wraps the width-enforced result in a generator
        width_enforced_generator = lambda: enforce_object_width(base_generator)
        
        # Apply height enforcement
        self.object = enforce_object_height(width_enforced_generator)
        
        self.assertEqual(self.object.shape, (2, 2))
        self.assertTrue(all(np.any(row != 0) for row in self.object))
        self.assertTrue(all(np.any(col != 0) for col in self.object.T))

    def test_contiguity_four(self):
        """Test 4-way contiguity."""
        self.object = create_object(5, 5, color_palette=1, contiguity=Contiguity.FOUR)
        structure = np.array([[0, 1, 0],
                            [1, 1, 1],
                            [0, 1, 0]])
        labeled, n_components = label(self.object != 0, structure=structure)
        self.assertLessEqual(n_components, 1)

    def test_contiguity_eight(self):
        """Test 8-way contiguity."""
        self.object = create_object(5, 5, color_palette=1, contiguity=Contiguity.EIGHT)
        structure = np.ones((3, 3))
        labeled, n_components = label(self.object != 0, structure=structure)
        self.assertLessEqual(n_components, 1)

    def test_small_matrix(self):
        """Test with minimal 2x2 matrix."""
        # Create a base generator function
        base_generator = lambda: create_object(2, 2, color_palette=1)
        
        # Create a new generator that includes width enforcement
        width_enforced_generator = lambda: enforce_object_width(base_generator)
        
        # Apply height enforcement to the width-enforced generator
        self.object = enforce_object_height(width_enforced_generator)
        
        self.assertEqual(self.object.shape, (2, 2))
        self.assertTrue(all(np.any(row != 0) for row in self.object))
        self.assertTrue(all(np.any(col != 0) for col in self.object.T))

    def test_complex_constraints(self):
        """Test combination of multiple constraints."""
        generator = lambda: create_object(4, 4, 
                                      color_palette=[1, 2],
                                      contiguity=Contiguity.FOUR)
        self.object = enforce_object_width(generator)
        
        # Check constraints
        self.assertTrue(all(np.any(row != 0) for row in self.object))
        
        # Check contiguity
        structure = np.array([[0, 1, 0],
                            [1, 1, 1],
                            [0, 1, 0]])
        labeled, n_components = label(self.object != 0, structure=structure)
        self.assertLessEqual(n_components, 1)

if __name__ == '__main__':
    unittest.main()

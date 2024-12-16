"""
Input Matrix Generation Library

This module provides a collection of utility functions for creating ARC matrices. 
They can be used in the create_input() of the ARCTaskGenerator class, but not in
the transform_input() method. In other words, those functions are not relevant
for solving ARC tasks at runtime.
"""

import numpy as np

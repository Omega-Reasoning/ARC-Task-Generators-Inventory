import numpy as np
from typing import Dict, List, Any, Tuple
from arc_task_generator import ARCTaskGenerator

class ARCTask00d62c1bGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input matrices can have different sizes.",
            "They only contain {color('object_color')} and empty cells.",
            "The {color('object_color')} cells sometimes form closed objects."
        ]
        transformation_reasoning_chain = [
            "The output matrix is constructed by copying the input matrix and filling the closed {color('object_color')} objects with {color('fill_color')} cells."
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars: Dict[str, Any], matrixvars: Dict[str, Any]) -> np.ndarray:
        height, width = matrixvars['height'], matrixvars['width']
        matrix = np.zeros((height, width), dtype=int)
        
        # Ensure minimum matrix size for both shapes
        if min(height, width) < 8:
            height = max(height, 8)
            width = max(width, 8)
            matrix = np.zeros((height, width), dtype=int)
        
        # Decide on shape type (rectangle or octagon)
        shape_type = np.random.choice(['rectangle', 'octagon'])
        
        if shape_type == 'rectangle':
            # Create a rectangle outline
            x1 = np.random.randint(1, width-5)
            y1 = np.random.randint(1, height-5)
            
            w = np.random.randint(4, min(width-x1-1, 8))
            h = np.random.randint(4, min(height-y1-1, 8))
            
            # Draw rectangle borders with green (3)
            matrix[y1:y1+h, x1] = taskvars['object_color']  # Left
            matrix[y1:y1+h, x1+w-1] = taskvars['object_color']   # Right
            matrix[y1, x1:x1+w] = taskvars['object_color']   # Top
            matrix[y1+h-1, x1:x1+w] = taskvars['object_color']   # Bottom
            
        else:  # octagon
            center_x = width // 2
            center_y = height // 2
            
            # Ensure valid radius range
            min_radius = 3
            max_radius = min(min(width, height) // 3, 5)
            if max_radius <= min_radius:
                max_radius = min_radius + 1
            
            radius = np.random.randint(min_radius, max_radius)
            
            # Create an octagon by defining 8 key points
            points = []
            for i in range(8):
                angle = i * np.pi / 4
                x = int(center_x + radius * np.cos(angle))
                y = int(center_y + radius * np.sin(angle))
                # Ensure points are within bounds
                x = max(0, min(x, width-1))
                y = max(0, min(y, height-1))
                points.append((x, y))
            
            # Connect the points with lines
            for i in range(8):
                x1, y1 = points[i]
                x2, y2 = points[(i + 1) % 8]
                
                # Draw line between points using Bresenham's algorithm
                dx = abs(x2 - x1)
                dy = abs(y2 - y1)
                sx = 1 if x1 < x2 else -1
                sy = 1 if y1 < y2 else -1
                err = dx - dy
                
                x, y = x1, y1
                while True:
                    if 0 <= x < width and 0 <= y < height:
                        matrix[y, x] = taskvars['object_color'] 
                    if x == x2 and y == y2:
                        break
                    e2 = 2 * err
                    if e2 > -dy:
                        err -= dy
                        x += sx
                    if e2 < dx:
                        err += dx
                        y += sy
        
        return matrix

    def transform_input(self, matrix: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        output = matrix.copy()
        height, width = matrix.shape
        
        # First fill all non-green cells with fill color
        for y in range(height):
            for x in range(width):
                if matrix[y, x] == 0:
                    output[y, x] = taskvars['fill_color']  
        
        # Then flood fill from edges with empty cells
        def flood_fill(y: int, x: int):
            if not (0 <= y < height and 0 <= x < width):  # Out of bounds
                return
            if output[y, x] != taskvars['fill_color']:  
                return
                
            output[y, x] = 0  # Set back to empty
            
            # Spread in four directions
            flood_fill(y+1, x)  # down
            flood_fill(y-1, x)  # up
            flood_fill(y, x+1)  # right
            flood_fill(y, x-1)  # left
        
        # Start flood fill from all edges
        for x in range(width):
            if output[0, x] == taskvars['fill_color']:
                flood_fill(0, x)
            if output[height-1, x] == taskvars['fill_color']:
                flood_fill(height-1, x)
        
        for y in range(height):
            if output[y, 0] == taskvars['fill_color']:
                flood_fill(y, 0)
            if output[y, width-1] == taskvars['fill_color']:
                flood_fill(y, width-1)
        
        return output

    def create_matrices(self) -> Tuple[Dict[str, Any], Dict[str, List[Dict[str, np.ndarray]]]]:
        # select a random object color and a fill color which is different from it
        object_color = np.random.randint(1, 9)
        available_colors = list(range(1, 9))
        available_colors.remove(object_color)
        fill_color = np.random.choice(available_colors)
        taskvars = {}
        taskvars['object_color'] = object_color
        taskvars['fill_color'] = fill_color
        num_train = np.random.randint(2, 6)
        min_size = 5
        max_size = 15
        
        train_examples = []
        for _ in range(num_train):
            height = np.random.randint(min_size, max_size)
            width = np.random.randint(min_size, max_size)
            
            matrixvars = {'height': height, 'width': width}
            input_matrix = self.create_input(taskvars, matrixvars)
            output_matrix = self.transform_input(input_matrix, taskvars)
            
            train_examples.append({
                'input': input_matrix,
                'output': output_matrix
            })
        
        # create test example (potentially larger)
        test_height = np.random.randint(10, 20)
        test_width = np.random.randint(10, 20)
        matrixvars = {'height': test_height, 'width': test_width}
        test_input = self.create_input(taskvars, matrixvars)
        test_output = self.transform_input(test_input, taskvars)
        
        test_examples = [{
            'input': test_input,
            'output': test_output
        }]
        
        return (taskvars,  {'train': train_examples, 'test': test_examples })


from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from typing import Dict, Any, Tuple, List, Set

class Task4612dd53Generator(ARCTaskGenerator):

    def __init__(self):
        # 1) Input reasoning chain
        input_reasoning_chain = [
            "Input grids have {vars['rows']} rows; the number of columns varies between examples.",
            "They contain {color('object_color')} and empty (0) cells.",
            "To construct the input grids, first create a one-cell-wide {color('object_color')} rectangular frame and sometimes add a horizontal or vertical line that extends from one edge to the other.",
            "The vertical or horizontal lines should always be separated from the rectangular frame by at least one empty column or row, respectively.",
            "Next, create holes by emptying (0) several {color('object_color')} cells from the structure, ensuring that it still maintains the appearance of a frame.",
            "The rectangular frame always covers more than three rows and more than three columns, and its position varies across examples."
        ]

        # 2) Transformation reasoning chain
        transformation_reasoning_chain = [
            "The output grids are constructed by copying the input grids and identifying the structure of the frame and the single horizontal or vertical line passing through the frame, if present.",
            "Once identified, fill all empty (0) cells that are part of the one-cell wide frame and the horizontal or vertical line with {color('fill_color')} color.",
            "This results in a one-cell-wide rectangular frame made of two colors, sometimes including a horizontal or vertical line inside the frame."
        ]

        # 3) Call super().__init__
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        """
        Create 3-5 training examples and 2 test examples with varying frame sizes.
        Rows are fixed across all examples for the task; columns vary per example.
        One test grid always has a vertical or horizontal line inside the frame.
        """
        # Basic task-wide random variables
        # Number of rows is fixed for all examples in this task; columns will vary per example
        rows = random.randint(8, 30)

        # object_color and fill_color must be different
        object_color = random.randint(1, 9)
        possible_fill_colors = [c for c in range(1, 10) if c != object_color]
        fill_color = random.choice(possible_fill_colors)

        taskvars = {
            'rows': rows,
            'object_color': object_color,
            'fill_color': fill_color,
        }

        # Make each example with its own unique frame
        def make_example(example_type):
            # Choose the number of columns for this specific example (columns vary across examples)
            cols_local = random.randint(8, 30)
            # Create frame dimensions for this specific example
            # Frame size constraints - ensure the frame always occupies more than three rows and more than three columns
            # (i.e., minimum size is 4). Ensure it fits within grid bounds (rows fixed, cols vary per-example).
            frame_height = random.randint(4, rows - 2)
            frame_width = random.randint(4, cols_local - 2)
            
            # For examples with lines, ensure frame is big enough
            if example_type in ["frame_hline", "frame_vline"]:
                # For horizontal lines, need at least 5 rows
                # For vertical lines, need at least 5 columns
                frame_height = max(frame_height, 5) if example_type == "frame_hline" else frame_height
                frame_width = max(frame_width, 5) if example_type == "frame_vline" else frame_width
                
                # Ensure we don't exceed grid boundaries
                frame_height = min(frame_height, rows - 2)
                frame_width = min(frame_width, cols_local - 2)
            
            # Calculate frame position (centered) within the per-example grid
            top = (rows - frame_height) // 2
            left = (cols_local - frame_width) // 2
            bottom = top + frame_height - 1
            right = left + frame_width - 1
            
            # Create gridvars for this example
            gridvars = {
                'example_type': example_type,
                'top': top,
                'left': left,
                'bottom': bottom,
                'right': right
            }
            # Pass per-example cols through a copy of taskvars so create_input/transform_input
            # see the correct grid shape for this example.
            taskvars_local = dict(taskvars)
            taskvars_local['cols'] = cols_local

            gridvars['cols'] = cols_local

            input_grid = self.create_input(taskvars_local, gridvars)
            output_grid = self.transform_input(input_grid, taskvars_local)
            return GridPair(input=input_grid, output=output_grid)

        # Required examples for training
        train_examples = [
            make_example("frame_only"),
            make_example("frame_hline"),
            make_example("frame_vline")
        ]
        
        # Add 0-2 more random examples
        num_extras = random.randint(0, 2)
        for _ in range(num_extras):
            train_examples.append(make_example(random.choice(["frame_only", "frame_hline", "frame_vline"])))

        # Two test examples: one with just a frame, one with a line
        test_examples = [
            make_example("frame_only"),
            make_example(random.choice(["frame_hline", "frame_vline"]))  # Guaranteed to have a line
        ]

        return taskvars, {'train': train_examples, 'test': test_examples}

    def create_input(self,
                     taskvars: Dict[str, Any],
                     gridvars: Dict[str, Any]) -> np.ndarray:
        """
        Create an input grid with a rectangular frame and optional line.
        Frame dimensions are specified in gridvars for each example.
        """
        rows = taskvars['rows']
        cols = taskvars['cols']
        object_color = taskvars['object_color']
        
        # Get frame dimensions from gridvars
        top = gridvars['top']
        left = gridvars['left']
        bottom = gridvars['bottom']
        right = gridvars['right']
        
        example_type = gridvars['example_type']
        
        # Create empty grid
        grid = np.zeros((rows, cols), dtype=int)
        
        # Define corner coordinates
        corners = {(top, left), (top, right), (bottom, left), (bottom, right)}
        
        # Create the rectangular frame
        frame_coords = set()
        
        # Top edge
        for c in range(left, right + 1):
            frame_coords.add((top, c))
        
        # Bottom edge
        for c in range(left, right + 1):
            frame_coords.add((bottom, c))
        
        # Left edge (excluding corners)
        for r in range(top + 1, bottom):
            frame_coords.add((r, left))
        
        # Right edge (excluding corners)
        for r in range(top + 1, bottom):
            frame_coords.add((r, right))
        
        # Add line if needed
        line_coords = set()
        if example_type == "frame_hline":
            # Ensure the frame is big enough for a proper line with buffers
            frame_height = bottom - top + 1
            if frame_height < 5:  # Need at least 5 rows for top, buffer, line, buffer, bottom
                # If not, make the frame bigger
                raise ValueError("Frame too small for horizontal line")
            else:
                # Choose a random row for horizontal line that's not adjacent to the frame edges
                line_r = random.randint(top + 2, bottom - 2)
                gridvars['line_r'] = line_r
                
                # Create horizontal line
                for c in range(left, right + 1):
                    line_coords.add((line_r, c))
                
        elif example_type == "frame_vline":
            # Ensure the frame is big enough for a proper line with buffers
            frame_width = right - left + 1
            if frame_width < 5:  # Need at least 5 columns for left, buffer, line, buffer, right
                # If not, make the frame bigger
                raise ValueError("Frame too small for vertical line")
            else:
                # Choose a random column for vertical line
                line_c = random.randint(left + 2, right - 2)
                gridvars['line_c'] = line_c
                
                # Create vertical line
                for r in range(top, bottom + 1):
                    line_coords.add((r, line_c))
        
        # Fill in frame and line cells
        all_structure_coords = frame_coords.union(line_coords)
        for r, c in all_structure_coords:
            grid[r, c] = object_color
            
        # Create separate list of segments
        segments = []
        
        # Top edge (exclude corners)
        if right - left > 2:  # Only if there's room for holes
            segments.append([(top, c) for c in range(left + 1, right)])
        
        # Bottom edge (exclude corners)
        if right - left > 2:
            segments.append([(bottom, c) for c in range(left + 1, right)])
        
        # Left edge (exclude corners)
        if bottom - top > 2:
            segments.append([(r, left) for r in range(top + 1, bottom)])
        
        # Right edge (exclude corners)
        if bottom - top > 2:
            segments.append([(r, right) for r in range(top + 1, bottom)])
        
        # Add line if present (special handling)
        if example_type == "frame_hline" and 'line_r' in gridvars:
            # Only include the part of the line that's inside the frame
            line_segment = [(gridvars['line_r'], c) for c in range(left + 1, right)]
            if line_segment:
                segments.append(line_segment)
                
        elif example_type == "frame_vline" and 'line_c' in gridvars:
            # Only include the part of the line that's inside the frame
            line_segment = [(r, gridvars['line_c']) for r in range(top + 1, bottom)]
            if line_segment:
                segments.append(line_segment)
        
        # Create holes using improved method
        for segment in segments:
            # Skip very short segments
            if len(segment) < 3:
                continue
                
            # Find valid hole positions (must not create adjacent holes)
            valid_positions = []
            for i in range(1, len(segment) - 1):  # Skip first and last
                r, c = segment[i]
                valid_positions.append((i, r, c))
            
            # If we have valid positions, make some holes
            if valid_positions:
                # Determine how many holes to add - scale with segment length
                # For smaller segments, use fewer holes
                num_holes = min(len(valid_positions) // 2, random.randint(1, 3))
                if num_holes == 0 and valid_positions:
                    num_holes = 1  # Ensure at least one hole if possible
                
                # Randomly choose positions
                if num_holes <= len(valid_positions):
                    selected = random.sample(valid_positions, num_holes)
                    for i, r, c in selected:
                        grid[r, c] = 0
        
        # Ensure at least one hole exists (but not in corners)
        if not np.any(grid == 0):
            valid_positions = [pos for pos in all_structure_coords if pos not in corners]
            if valid_positions:
                hole_pos = random.choice(valid_positions)
                grid[hole_pos] = 0
        
        return grid

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        """
        Fill all empty (0) cells that are part of the frame or line with fill_color.
        """
        output_grid = grid.copy()
        object_color = taskvars['object_color']
        fill_color = taskvars['fill_color']
        
        # First, we need to identify the frame in the grid
        # We'll look for a rectangular pattern of object_color cells
        
        # Find min/max coordinates of non-zero cells to determine frame bounds
        non_zero_cells = np.where(grid != 0)
        if len(non_zero_cells[0]) == 0:  # No non-zero cells
            return output_grid
            
        # Get top, left, bottom, right bounds of the object_color cells
        top = min(non_zero_cells[0])
        bottom = max(non_zero_cells[0])
        left = min(non_zero_cells[1])
        right = max(non_zero_cells[1])
        
        # STEP 1: Define the basic frame structure
        frame_structure = set()
        
        # Top edge
        for c in range(left, right + 1):
            frame_structure.add((top, c))
        
        # Bottom edge
        for c in range(left, right + 1):
            frame_structure.add((bottom, c))
        
        # Left edge (excluding corners)
        for r in range(top + 1, bottom):
            frame_structure.add((r, left))
        
        # Right edge (excluding corners)
        for r in range(top + 1, bottom):
            frame_structure.add((r, right))
        
        # STEP 2: Detect horizontal or vertical line
        # We'll look for the most promising line and only add one
        
        # First check for horizontal lines
        h_line_candidates = []
        for r in range(top + 1, bottom):
            line_cells = []
            
            # Count object_color and zero cells across this row
            valid_cells = 0
            for c in range(left + 1, right):
                if grid[r, c] == object_color or grid[r, c] == 0:
                    valid_cells += 1
                    if grid[r, c] == object_color:
                        line_cells.append((r, c))
            
            # If there are object_color cells and no invalid cells
            if len(line_cells) > 0 and valid_cells == (right - left - 1):
                h_line_candidates.append({
                    'row': r,
                    'line_cells': line_cells,
                    'count': len(line_cells),
                    'cells': [(r, c) for c in range(left, right + 1)]
                })
        
        # Check for vertical lines
        v_line_candidates = []
        for c in range(left + 1, right):
            line_cells = []
            
            # Count object_color and zero cells down this column
            valid_cells = 0
            for r in range(top + 1, bottom):
                if grid[r, c] == object_color or grid[r, c] == 0:
                    valid_cells += 1
                    if grid[r, c] == object_color:
                        line_cells.append((r, c))
            
            # If there are object_color cells and no invalid cells
            if len(line_cells) > 0 and valid_cells == (bottom - top - 1):
                v_line_candidates.append({
                    'col': c,
                    'line_cells': line_cells,
                    'count': len(line_cells),
                    'cells': [(r, c) for r in range(top, bottom + 1)]
                })
        
        # If we have candidates for both horizontal and vertical lines,
        # choose the one with more object_color cells
        line_coords = set()
        if h_line_candidates and v_line_candidates:
            best_h = max(h_line_candidates, key=lambda x: x['count'])
            best_v = max(v_line_candidates, key=lambda x: x['count'])
            
            if best_h['count'] > best_v['count']:
                line_coords = set(best_h['cells'])
            else:
                line_coords = set(best_v['cells'])
        elif h_line_candidates:
            best_h = max(h_line_candidates, key=lambda x: x['count'])
            line_coords = set(best_h['cells'])
        elif v_line_candidates:
            best_v = max(v_line_candidates, key=lambda x: x['count'])
            line_coords = set(best_v['cells'])
        
        # Combine frame structure with the detected line
        all_structure_coords = frame_structure.union(line_coords)
        
        # STEP 3: Fill empty cells in the structure with fill_color
        for r, c in all_structure_coords:
            if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
                if grid[r, c] == 0:  # Only fill empty cells
                    output_grid[r, c] = fill_color
        
        return output_grid



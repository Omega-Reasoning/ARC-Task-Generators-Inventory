from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from input_library import create_object, random_cell_coloring, retry
from transformation_library import find_connected_objects
import numpy as np
import random

class ARCTask1f642eb9Generator(ARCTaskGenerator):

    def __init__(self):
        input_reasoning_chain = [
            "The input grid has size {vars['rows']} X {vars['rows']}.",
            "There is exactly one subgrid of dimension {vars['subrows']} X {vars['subcols']} of color cyan(8).",
            "The cyan subgrid is placed with a minimum offset of 2 cells from any edge.",
            "Colored cells (1-7, 9) are placed at the edges of the input grid.",
            "Edge cells are only placed in positions that align with the cyan subgrid.",
            "Each edge gets 1-3 cells, with a total maximum based on the subgrid size.",
            "No two edge cells share the same row or column to avoid ambiguity.",
            "The remaining cells of the input grid are empty(0)."
        ]

        transformation_reasoning_chain = [
            "The output grid has the same dimensions as the input grid.",
            "Copy the input grid to the output grid.",
            "For each colored cell at the edges, find the closest boundary cell of the cyan(8) subgrid that is in the same row or column.",
            "Change the color of that boundary cell to match the color of the edge cell.",
            "Only one boundary cell is colored per edge cell - the closest one in the same row or column."
        ]

        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars, gridvars):
        rows = taskvars['rows']
        subrows = taskvars['subrows']
        subcols = taskvars['subcols']

        grid = np.zeros((rows, rows), dtype=int)

        # Place the subgrid with minimum offset of 2 from edges
        min_offset = 2
        offset_row = random.randint(min_offset, rows - subrows - min_offset)
        offset_col = random.randint(min_offset, rows - subcols - min_offset)
        grid[offset_row:offset_row + subrows, offset_col:offset_col + subcols] = 8

        # Place edge cells
        colors = list(range(1, 8)) + [9]  # Exclude color 8 (cyan) from edge cells
        random.shuffle(colors)

        # For each edge, only consider positions that align with the subgrid
        edges = [
            [(0, c) for c in range(offset_col, offset_col + subcols)],  # top edge
            [(rows-1, c) for c in range(offset_col, offset_col + subcols)],  # bottom edge
            [(r, 0) for r in range(offset_row, offset_row + subrows)],  # left edge
            [(r, rows-1) for r in range(offset_row, offset_row + subrows)]  # right edge
        ]

        # Track intersections of rows and columns to avoid ambiguity
        used_rows = set()
        used_cols = set()

        # Calculate maximum total cells based on subgrid size
        max_total_cells = min(subrows * subcols, 9)  # Cap at 9 since we have 9 colors
        total_cells = random.randint(4, max_total_cells)
        
        random.shuffle(edges)
        max_cells_per_edge = [1, 1, 1, 1]  # Start with 1 cell per edge
        
        # Distribute remaining cells randomly
        remaining_cells = total_cells - 4
        edge_idx = 0
        while remaining_cells > 0 and edge_idx < 4:
            if edge_idx == 0:
                extra = random.randint(0, min(2, remaining_cells))
            else:
                extra = random.randint(0, min(1, remaining_cells))
            max_cells_per_edge[edge_idx] += extra
            remaining_cells -= extra
            edge_idx += 1

        for edge_idx, (edge, max_cells) in enumerate(zip(edges, max_cells_per_edge)):
            valid_positions = []
            for r, c in edge:
                # For top/bottom edges, check if any cell in this column's row is already used
                if edge_idx < 2:  # top or bottom edge
                    if c not in used_cols and not any(r in used_rows for r in range(offset_row, offset_row + subrows)):
                        valid_positions.append((r, c))
                # For left/right edges, check if any cell in this row's column is already used
                else:  # left or right edge
                    if r not in used_rows and not any(c in used_cols for c in range(offset_col, offset_col + subcols)):
                        valid_positions.append((r, c))
            
            if valid_positions:
                positions = random.sample(valid_positions, k=min(max_cells, len(valid_positions)))
                for r, c in positions:
                    grid[r, c] = colors.pop() if colors else random.choice([1,2,3,4,5,6,7,9])
                    # Track used rows and columns
                    if edge_idx < 2:  # top or bottom edge
                        used_cols.add(c)
                    else:  # left or right edge
                        used_rows.add(r)

        return grid

    def transform_input(self, grid, taskvars):
        rows, cols = grid.shape
        output_grid = grid.copy()

        # Collect all edge cells with their colors
        edge_cells = []
        for r in range(rows):
            for c in range(cols):
                if r == 0 or r == rows - 1 or c == 0 or c == cols - 1:
                    if grid[r, c] != 0:
                        edge_cells.append((r, c, grid[r, c]))

        # Find the subgrid boundaries
        subgrid_cells = []
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] == 8:
                    # Check if it's a boundary cell (has a non-8 neighbor)
                    is_boundary = False
                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nr, nc = r + dr, c + dc
                        if (0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] != 8) or \
                           nr < 0 or nr >= rows or nc < 0 or nc >= cols:  # Edge of grid
                            is_boundary = True
                            break
                    if is_boundary:
                        subgrid_cells.append((r, c))

        # For each edge cell, find and color only the closest boundary cell in its row/column
        for er, ec, color in edge_cells:
            closest_distance = float('inf')
            closest_cell = None
            
            # Find closest boundary cell in same row or column
            for r, c in subgrid_cells:
                if r == er or c == ec:  # Same row or column
                    distance = abs(r - er) + abs(c - ec)
                    if distance < closest_distance:
                        closest_distance = distance
                        closest_cell = (r, c)
            
            # Color only the closest boundary cell
            if closest_cell:
                r, c = closest_cell
                output_grid[r, c] = color

        return output_grid

    def create_grids(self):
        # Ensure grid is large enough to accommodate largest possible subgrid + offsets
        rows = 10 #random.randint(12, 20)  # Increased minimum size
        subrows = 3 #random.randint(3, 5)  # Reduced maximum size
        subcols = 3 #random.randint(3, 5 )  # Reduced maximum size

        taskvars = {
            'rows': rows,
            'subrows': subrows,
            'subcols': subcols
        }

        train_pairs = []
        for _ in range(3):  # 3 training examples
            taskvars['subrows'] = random.randint(2, 4)
            taskvars['subcols'] = random.randint(2, 4)
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            train_pairs.append(GridPair({"input": input_grid, "output": output_grid}))

        test_pairs = []
        for _ in range(1):  # 1 test example
            taskvars['subrows'] = random.randint(3, 5)
            taskvars['subcols'] = random.randint(3, 5)
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            test_pairs.append(GridPair({"input": input_grid, "output": output_grid}))

        return taskvars, TrainTestData({"train": train_pairs, "test": test_pairs})


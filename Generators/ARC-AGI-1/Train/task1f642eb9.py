from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.input_library import create_object, random_cell_coloring, retry
from Framework.transformation_library import find_connected_objects
import numpy as np
import random

class Task1f642eb9Generator(ARCTaskGenerator):

    def __init__(self):
        input_reasoning_chain = [
            "Input grids are square and have size {vars['grid_size']} x {vars['grid_size']}.",
            "There is exactly one rectangular block of color {color('block_color') }.",
            "The {color('block_color')} block is placed at least 2 cells away from every edge.",
            "Other colored cells (not {color('block_color')}) are placed only on the edges of the input grid.",
            "Edge cells are placed only at positions that align with the cyan subgrid.",
            "Each edge contains 1–3 cells; the total number is limited by the subgrid size.",
            "No two edge cells share the same row or the same column to avoid ambiguity.",
            "All remaining cells in the input grid are empty (0)."
        ]

        transformation_reasoning_chain = [
            "The output grid has the same dimensions as the input and is initialized as a copy of the input grid.",
            "For each colored cell on the input border, locate the nearest uncolored boundary cell of the cyan {color('block_color')} block that lies in the same row or the same column.",
            "Color that boundary cell with the same color as the corresponding border cell.",
            "Each boundary cell is colored by at most one border cell; no two border cells color the same boundary cell."
        ]

        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars, gridvars):
        grid_size = taskvars['grid_size']
        subrows = taskvars['subrows']
        subcols = taskvars['subcols']

        grid = np.zeros((grid_size, grid_size), dtype=int)

        # Place the subgrid with minimum offset of 2 from edges
        min_offset = 2
        offset_row = random.randint(min_offset, grid_size - subrows - min_offset)
        offset_col = random.randint(min_offset, grid_size - subcols - min_offset)
        grid[offset_row:offset_row + subrows, offset_col:offset_col + subcols] = 8

        # Place edge cells
        colors = list(range(1, 8)) + [9]  # Exclude color 8 (cyan) from edge cells
        random.shuffle(colors)

        # For each edge, only consider positions that align with the subgrid (named edges so we keep track of type)
        edges = [
            ("top", [(0, c) for c in range(offset_col, offset_col + subcols)]),  # top edge
            ("bottom", [(grid_size - 1, c) for c in range(offset_col, offset_col + subcols)]),  # bottom edge
            ("left", [(r, 0) for r in range(offset_row, offset_row + subrows)]),  # left edge
            ("right", [(r, grid_size - 1) for r in range(offset_row, offset_row + subrows)])  # right edge
        ]

        # Track used rows/columns/positions so we don't create ambiguous placements
        used_rows = set()
        used_cols = set()
        used_positions = set()

        # Keep track of which subgrid boundary cells have already been targeted by edge cells
        reserved_targets = set()

        # Calculate maximum total cells based on subgrid area and available colors
        max_total_cells = min(subrows * subcols, 9)  # Cap at 9 since we have 9 colors
        total_cells = random.randint(4, max(4, max_total_cells))

        random.shuffle(edges)
        max_cells_per_edge = [1, 1, 1, 1]

        # Distribute remaining cells randomly across edges
        remaining_cells = total_cells - 4
        edge_idx = 0
        while remaining_cells > 0 and edge_idx < 4:
            extra = random.randint(0, min(2, remaining_cells)) if edge_idx == 0 else random.randint(0, min(1, remaining_cells))
            max_cells_per_edge[edge_idx] += extra
            remaining_cells -= extra
            edge_idx += 1

        for edge_idx, ((ename, edge), max_cells) in enumerate(zip(edges, max_cells_per_edge)):
            valid_positions = []
            for r, c in edge:
                if (r, c) in used_positions:
                    continue

                # Determine the boundary target cell (in the subgrid) that this edge position would map to
                if ename == "top":
                    target = (offset_row, c)
                elif ename == "bottom":
                    target = (offset_row + subrows - 1, c)
                elif ename == "left":
                    target = (r, offset_col)
                else:  # right
                    target = (r, offset_col + subcols - 1)

                # Skip if this target is already reserved (would create ambiguity)
                if target in reserved_targets:
                    continue

                # For top/bottom edges avoid re-using a column; for left/right avoid re-using a row
                if ename in ("top", "bottom"):
                    if c in used_cols:
                        continue
                else:
                    if r in used_rows:
                        continue

                valid_positions.append((r, c, target))

            if valid_positions:
                selected = random.sample(valid_positions, k=min(max_cells, len(valid_positions)))
                for r, c, target in selected:
                    grid[r, c] = colors.pop() if colors else random.choice([1,2,3,4,5,6,7,9])
                    used_positions.add((r, c))
                    reserved_targets.add(target)
                    if ename in ("top", "bottom"):
                        used_cols.add(c)
                    else:
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

        # For each edge cell, find and color only the closest unassigned boundary cell in its row/column
        assigned_boundary = set()
        for er, ec, color in edge_cells:
            # Collect candidate boundary cells in same row or column and sort by distance
            candidates = [(abs(r - er) + abs(c - ec), (r, c)) for r, c in subgrid_cells if r == er or c == ec]
            candidates.sort(key=lambda x: x[0])

            # Pick the closest candidate that hasn't been assigned yet
            chosen = None
            for _, (r, c) in candidates:
                if (r, c) not in assigned_boundary:
                    chosen = (r, c)
                    break

            if chosen:
                r, c = chosen
                output_grid[r, c] = color
                assigned_boundary.add((r, c))

        return output_grid

    def create_grids(self):
        # Choose a single grid size for this task (fixed across all examples for this task).
        # Ensure grid is large enough to place a subgrid at least 2 cells from each edge.
        min_offset = 2
        grid_size = random.randint(max(6, 5), 30)  # minimum 6 to allow subgrid >=2 and offset 2

        taskvars = {
            'grid_size': grid_size,
            'block_color': 8,  # cyan block color
            'subrows': None,
            'subcols': None
        }

        train_pairs = []
        for _ in range(3):  # 3 training examples
            max_sub = max(2, min(4, grid_size - 2 * min_offset))
            taskvars['subrows'] = random.randint(2, max_sub)
            taskvars['subcols'] = random.randint(2, max_sub)
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            train_pairs.append(GridPair({"input": input_grid, "output": output_grid}))

        test_pairs = []
        for _ in range(1):  # 1 test example
            max_sub = max(2, min(5, grid_size - 2 * min_offset))
            taskvars['subrows'] = random.randint(3, max_sub) if max_sub >= 3 else 2
            taskvars['subcols'] = random.randint(3, max_sub) if max_sub >= 3 else 2
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            test_pairs.append(GridPair({"input": input_grid, "output": output_grid}))

        return taskvars, TrainTestData({"train": train_pairs, "test": test_pairs})


from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from input_library import retry, random_cell_coloring
from transformation_library import find_connected_objects, GridObjects
import numpy as np
import random
from typing import Dict, Any, Tuple

class Task8731374eGenerator(ARCTaskGenerator):
    
    def __init__(self):
        input_reasoning_chain = [
            "The input grids are of size {vars['rows']} × {vars['cols']}.",
            "Each input grid has a completely filled background of multi-colored cells (0–9), with only a few empty cells; most of the cells are colored.",
            "On top of this multi-colored background lies a rectangular block of a single color (call it a), at least 3×3 in size, containing several cells of a different color (call it b).",
            "These color-b cells are positioned in different rows and columns within the rectangle, with no two color-b cells sharing the same row or column.",
            "The color and size of the rectangular object vary across grids."
        ]
        
        transformation_reasoning_chain = [
            "The output grids are constructed by copying the input grids and identifying the single colored rectangular object on the multi-colored background.",
            "Once the rectangular object is identified, locate the differently colored cells (color b) within it.",
            "Then, color the entire rows and columns containing these color-b cells within the rectangular object.",
            "Once this is done, remove all background cells and crop the grid so that it contains only the rectangular object, including the rows and columns with color-b cells."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Only rows/cols are task-level (fixed across examples for the description).
        taskvars = {
            'rows': random.randint(10, 30),
            'cols': random.randint(10, 30),
        }
        
        num_train = random.randint(3, 5)
        train_grids = []
        test_grids = []
        
        for _ in range(num_train):
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            train_grids.append({'input': input_grid, 'output': output_grid})
        
        input_grid = self.create_input(taskvars, {})
        output_grid = self.transform_input(input_grid, taskvars)
        test_grids.append({'input': input_grid, 'output': output_grid})
        
        train_test_data = {
            'train': train_grids,
            'test': test_grids
        }
        
        return taskvars, train_test_data

    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        rows = taskvars['rows']
        cols = taskvars['cols']
        
        # Choose per-example rectangle & marker colors (a and b), distinct.
        rect_color = random.randint(1, 9)
        marker_color = random.randint(1, 9)
        while marker_color == rect_color:
            marker_color = random.randint(1, 9)
        
        # Create background (mostly filled with colors not including a/b)
        grid = np.zeros((rows, cols), dtype=int)
        available_bg = [i for i in range(1, 10) if i not in (rect_color, marker_color)]
        for r in range(rows):
            for c in range(cols):
                if random.random() < 0.85:
                    grid[r, c] = random.choice(available_bg)
        
        # Create rectangular object with minimum size 3x3
        rect_height = random.randint(3, min(8, rows - 2))
        rect_width  = random.randint(3, min(8, cols - 2))
        rect_top    = random.randint(0, rows - rect_height)
        rect_left   = random.randint(0, cols - rect_width)
        
        grid[rect_top:rect_top + rect_height, rect_left:rect_left + rect_width] = rect_color
        
        # Place marker_color cells (b) with unique rows/cols inside the rectangle
        num_markers = max(2, rect_height // 3)  # at least 2 markers for clarity
        num_markers = min(num_markers, rect_height, rect_width)
        marker_rows = random.sample(range(rect_top, rect_top + rect_height), num_markers)
        marker_cols = random.sample(range(rect_left, rect_left + rect_width), num_markers)
        random.shuffle(marker_cols)
        for r, c in zip(marker_rows, marker_cols):
            grid[r, c] = marker_color
        
        return grid

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        # Detect rectangle by looking for a compact region dominated by a single color
        unique_colors, counts = np.unique(grid[grid != 0], return_counts=True)
        sorted_indices = np.argsort(counts)[::-1]
        
        rect_color = None
        marker_color = None
        min_row = min_col = max_row = max_col = None
        
        for idx in sorted_indices:
            candidate = unique_colors[idx]
            positions = np.where(grid == candidate)
            if len(positions[0]) < 9:
                continue
            rmin, rmax = positions[0].min(), positions[0].max()
            cmin, cmax = positions[1].min(), positions[1].max()
            sub = grid[rmin:rmax+1, cmin:cmax+1]
            region_colors, region_counts = np.unique(sub, return_counts=True)
            
            # Expect two colors: rectangle (a) and markers/background inside (b),
            # but background should have been overwritten within the rectangle,
            # so we look for exactly two colors and a dominant one.
            if len(region_colors) == 2:
                # identify rect vs marker by dominance
                major_idx = np.argmax(region_counts)
                minor_idx = 1 - major_idx
                rect_cand = region_colors[major_idx]
                marker_cand = region_colors[minor_idx]
                # Ensure rect_cand equals the candidate we’re evaluating
                if rect_cand == candidate and rect_cand != 0 and marker_cand != 0:
                    rect_color = rect_cand
                    marker_color = marker_cand
                    min_row, max_row, min_col, max_col = rmin, rmax, cmin, cmax
                    break
        
        # Fallbacks if needed
        if rect_color is None or marker_color is None:
            if len(unique_colors) >= 2:
                rect_color = unique_colors[sorted_indices[0]]
                marker_color = unique_colors[sorted_indices[1]]
                rows, cols = np.where((grid == rect_color) | (grid == marker_color))
                min_row, max_row = rows.min(), rows.max()
                min_col, max_col = cols.min(), cols.max()
            else:
                rect_color = unique_colors[0] if len(unique_colors) > 0 else 1
                marker_color = 2 if rect_color != 2 else 3
                rows, cols = np.where((grid == rect_color) | (grid == marker_color))
                min_row, max_row = rows.min(), rows.max()
                min_col, max_col = cols.min(), cols.max()
        
        rect_region = grid[min_row:max_row+1, min_col:max_col+1].copy()
        
        # Fill rows/cols containing markers within the rectangle
        marker_positions = np.where(rect_region == marker_color)
        for r in marker_positions[0]:
            rect_region[r, :] = marker_color
        for c in marker_positions[1]:
            rect_region[:, c] = marker_color
        
        # Crop already done by extracting rect_region
        return rect_region



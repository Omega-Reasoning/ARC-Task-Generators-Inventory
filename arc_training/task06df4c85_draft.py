import numpy as np
import random

from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from typing import Dict, List, Any, Tuple

class JailFillingGeneratorV2(ARCTaskGenerator):
    def __init__(self):
        """
        Two-step logic:
          1) The input reasoning chain describes how the base grid has a certain dimension,
             with some cells (representing subgrids) colored, and the rest empty.
          2) The transformation reasoning chain describes how pairs of identically colored cells
             in the same row or column cause the in-between cells to be filled,
             provided there is no different color blocking the way.
        """
        input_reasoning_chain = [
            "A smaller base grid of size (H, W) is constructed, with some of its cells colored. "
            "Here each cell will later be expanded to a {vars['subgrid_rows']}x{vars['subgrid_cols']} subgrid in the final matrix. "
            "Rows and columns are separated by jail bars of color {color('sep_color')}."
        ]
        transformation_reasoning_chain = [
            "If two cells in the base grid share the same row or column and have the same color, "
            "then all cells in between them are filled with that color as long as no different color blocks the way. "
            "The final grid is then created by expanding each base cell and adding the jail bar separator lines."
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        """
        Creates the task variables and the training/test pairs of grids.
        """
        # 1) Pick random subgrid scaling factors
        subgrid_rows = random.randint(1, 4)
        subgrid_cols = random.randint(1, 4)
        
        # 2) Pick a separator color (1..9)
        sep_color = random.randint(1, 9)
        
        # 3) Choose 1-3 object colors different from sep_color
        all_colors = list(range(1, 10))
        all_colors.remove(sep_color)
        random.shuffle(all_colors)
        obj_colors = all_colors[: random.randint(1, 3)]
        
        # 4) Store them in taskvars
        taskvars = {
            "subgrid_rows": subgrid_rows,
            "subgrid_cols": subgrid_cols,
            "sep_color": sep_color,
        }
        for i, c in enumerate(obj_colors, start=1):
            taskvars[f"obj_color_{i}"] = c
        
        # Generate 3 training + 1 test examples
        nr_train = 3
        nr_test = 1
        
        train_data = []
        for _ in range(nr_train):
            inp = self.create_input(taskvars, {})
            outp = self.transform_input(inp, taskvars)
            train_data.append(GridPair(input=inp, output=outp))
        
        test_data = []
        for _ in range(nr_test):
            inp = self.create_input(taskvars, {})
            outp = self.transform_input(inp, taskvars)
            test_data.append(GridPair(input=inp, output=outp))
        
        return taskvars, {"train": train_data, "test": test_data}

    def create_input(self,
                     taskvars: Dict[str, Any],
                     gridvars: Dict[str, Any]) -> np.ndarray:
        """
        1) Choose dimensions for the base grid, ensuring that when scaled up
           it won't exceed 30 in either dimension.
        2) Color some cells with the chosen object colors (others remain 0).
        3) Scale up the base grid and insert the jail bars (sep_color).
        4) Return the final (scaled) input grid.
        
        The actual “connecting” logic is done by transform_input().
        """
        subR = taskvars["subgrid_rows"]
        subC = taskvars["subgrid_cols"]
        sep_color = taskvars["sep_color"]
        
        # Collect the available object colors from taskvars
        obj_colors = []
        idx = 1
        while f"obj_color_{idx}" in taskvars:
            obj_colors.append(taskvars[f"obj_color_{idx}"])
            idx += 1
        
        # --- 1) Determine base grid shape (H_small, W_small) ---
        # We want base_grid to have at least 3*(subR+1) rows, up to the max allowed
        # so that after scaling, the final shape is <= 30 in both dims.
        # Final shape = (H_small * subR + (H_small+1)) x (W_small * subC + (W_small+1))
        # We choose H_small in [3*(subR+1) .. feasible_max], similarly W_small in [3*(subC+1) .. feasible_max].
        
        def feasible_maxH(subR):
            # Solve (H_small * subR + (H_small+1)) <= 30
            # => H_small * subR + H_small + 1 <= 30
            # => H_small*(subR+1) <= 29
            # => H_small <= floor(29 / (subR+1))
            return max(3*(subR+1), 1) if (subR+1) == 0 else min(30, 29 // (subR+1))
        
        def feasible_maxW(subC):
            # Same logic for columns
            return max(3*(subC+1), 1) if (subC+1) == 0 else min(30, 29 // (subC+1))
        
        # Actually we want a range for H_small:
        # Lower bound: 3*(subR+1)
        # Upper bound: feasible_maxH(subR)
        # But we must ensure the upper bound is >= that lower bound; if not, we clamp them.
        minH = 3*(subR+1)
        maxH = feasible_maxH(subR)
        if maxH < minH:
            minH = maxH  # clamp to avoid negative range
        H_small = random.randint(minH, maxH) if minH <= maxH else minH
        
        minW = 3*(subC+1)
        maxW = feasible_maxW(subC)
        if maxW < minW:
            minW = maxW
        W_small = random.randint(minW, maxW) if minW <= maxW else minW
        
        # Build the base grid
        base_grid = np.zeros((H_small, W_small), dtype=int)
        
        # --- 2) Color some cells in base_grid ---
        # Random fraction of cells. We do not do any "connecting" here; just place colors.
        # The transformation step will do the logic of whether they remain or get filled, etc.
        nr_cells = H_small * W_small
        fill_count = random.randint(1, nr_cells // 2)  # up to half the cells colored
        coords = [(r, c) for r in range(H_small) for c in range(W_small)]
        random.shuffle(coords)
        chosen = coords[:fill_count]
        for (r, c) in chosen:
            base_grid[r, c] = random.choice(obj_colors)
        
        # --- 3) Scale up + insert bars to form the final input grid ---
        # We'll expand each base cell into subR x subC, then separate each row/column with a bar of sep_color.
        # Final shape = (H_small * subR + H_small + 1) x (W_small * subC + W_small + 1).
        final_H = H_small * subR + (H_small + 1)
        final_W = W_small * subC + (W_small + 1)
        grid = np.zeros((final_H, final_W), dtype=int)
        
        def row_bar_index(r_base):
            return r_base * (subR + 1)
        
        def col_bar_index(c_base):
            return c_base * (subC + 1)
        
        # Put horizontal bars
        for rb in range(H_small+1):
            r_bar = row_bar_index(rb)
            grid[r_bar, :] = sep_color
        
        # Put vertical bars
        for cb in range(W_small+1):
            c_bar = col_bar_index(cb)
            grid[:, c_bar] = sep_color
        
        # Fill subR x subC blocks
        for rb in range(H_small):
            for cb in range(W_small):
                color_val = base_grid[rb, cb]
                if color_val != 0:
                    top = row_bar_index(rb) + 1
                    left = col_bar_index(cb) + 1
                    # fill the sub-block
                    grid[top : top+subR, left : left+subC] = color_val
        
        return grid

    def transform_input(self,
                        grid: np.ndarray,
                        taskvars: Dict[str, Any]) -> np.ndarray:
        """
        1) Reconstruct the base_grid from the scaled input.
        2) For each row in base_grid, whenever we find two cells of the same color,
           fill the intervening cells with that color unless blocked by a different color.
        3) Repeat for columns.
        4) Rescale the updated base_grid + separator lines to produce the output.
        """
        sep_color = taskvars["sep_color"]
        subR = taskvars["subgrid_rows"]
        subC = taskvars["subgrid_cols"]
        
        # ---- 1) Rebuild the base_grid from the scaled input ----
        # We find the horizontal bar rows and vertical bar columns.
        nrows, ncols = grid.shape
        
        # Locate all full sep_color rows
        horiz_bars = []
        for r in range(nrows):
            if np.all(grid[r, :] == sep_color):
                horiz_bars.append(r)
        # Locate all full sep_color cols
        vert_bars = []
        for c in range(ncols):
            if np.all(grid[:, c] == sep_color):
                vert_bars.append(c)
        
        # number of base_grid rows, columns
        H_small = len(horiz_bars) - 1
        W_small = len(vert_bars) - 1
        
        # Rebuild the base_grid by sampling the sub-block top-left
        base_grid = np.zeros((H_small, W_small), dtype=int)
        for rbase in range(H_small):
            for cbase in range(W_small):
                r0 = horiz_bars[rbase] + 1
                c0 = vert_bars[cbase] + 1
                block = grid[r0 : r0+subR, c0 : c0+subC]
                # If multiple colors in the block, pick the first non-zero color found
                colors_in_block = block[block != 0]
                colors_in_block = colors_in_block[colors_in_block != sep_color]
                if len(colors_in_block) > 0:
                    base_grid[rbase, cbase] = colors_in_block[0]
        
        # ---- 2) Fill horizontally in base_grid ----
        for r in range(H_small):
            c = 0
            while c < W_small:
                colorA = base_grid[r, c]
                if colorA == 0:
                    c += 1
                    continue
                # find next cell with same color
                blocked = False
                for c2 in range(c+1, W_small):
                    colorB = base_grid[r, c2]
                    if colorB == colorA:
                        # fill between c+1 and c2-1 if not blocked
                        if not blocked:
                            for fill_c in range(c+1, c2):
                                if base_grid[r, fill_c] == 0:
                                    base_grid[r, fill_c] = colorA
                                else:
                                    # There's another color -> that blocks further fill
                                    blocked = True
                                    break
                        break
                    elif colorB != 0:
                        # Different color => block
                        blocked = True
                c += 1
        
        # ---- 3) Fill vertically in base_grid ----
        for c in range(W_small):
            r = 0
            while r < H_small:
                colorA = base_grid[r, c]
                if colorA == 0:
                    r += 1
                    continue
                blocked = False
                for r2 in range(r+1, H_small):
                    colorB = base_grid[r2, c]
                    if colorB == colorA:
                        if not blocked:
                            for fill_r in range(r+1, r2):
                                if base_grid[fill_r, c] == 0:
                                    base_grid[fill_r, c] = colorA
                                else:
                                    # block
                                    blocked = True
                                    break
                        break
                    elif colorB != 0:
                        blocked = True
                r += 1
        
        # ---- 4) Rescale the updated base_grid to final output with separator lines ----
        final_H = H_small * subR + (H_small + 1)
        final_W = W_small * subC + (W_small + 1)
        output = np.zeros((final_H, final_W), dtype=int)
        
        # horizontal bars
        for rb in range(H_small+1):
            output[rb*(subR+1), :] = sep_color
        # vertical bars
        for cb in range(W_small+1):
            output[:, cb*(subC+1)] = sep_color
        
        # fill blocks
        for rb in range(H_small):
            for cb in range(W_small):
                color_val = base_grid[rb, cb]
                if color_val != 0:
                    top = rb * (subR+1) + 1
                    left = cb * (subC+1) + 1
                    output[top : top+subR, left : left+subC] = color_val
        
        return output

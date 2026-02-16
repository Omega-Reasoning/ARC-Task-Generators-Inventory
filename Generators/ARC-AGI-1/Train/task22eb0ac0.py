from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

class Task22eb0ac0Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
                "The input grid is a square of size {vars['rows']} x {vars['rows']}.",
                "All cells are empty (0) except for some rows where two endpoint cells are placed in the first and last column.",
                "These endpoint rows are chosen from alternate rows starting from the 2nd row (row index 1) down to the last row.",
                "At most {vars['rows']//2} such rows contain endpoint cells.",
                "For each chosen row, the left endpoint (first column) is assigned a random color from 1–9, and the right endpoint (last column) is also assigned a random color from 1–9.",
                "In some chosen rows, both endpoints have the same color; in other chosen rows, the two endpoint colors are different."
            ]

        transformation_reasoning_chain = [
                "The output grid has the same size as the input grid.",
                "Copy the input grid to the output grid.",
                "For each row, check the colors of the first and last cell in that row.",
                "If both endpoint cells are non-zero and have the same color, fill the entire row with that color.",
                "Otherwise, leave the row unchanged."
            ]


        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    # -----------------------------
    # NEW: create input with an exact number of matching rows
    # -----------------------------
    def create_input_with_match_count(self, taskvars, match_count: int):
        rows = taskvars['rows']
        grid = np.zeros((rows, rows), dtype=int)

        max_cells = rows // 2
        available_rows = list(range(1, rows, 2))  # 2nd row to last, step 2

        # choose how many rows will contain endpoint cells (must be >= 1)
        num_cells = random.randint(max(1, match_count), max_cells)

        selected_rows = random.sample(available_rows, num_cells)

        # pick exactly match_count rows to have same colors on both ends
        match_rows = set(random.sample(selected_rows, match_count)) if match_count > 0 else set()

        for r in selected_rows:
            if r in match_rows:
                c = random.randint(1, 9)
                grid[r, 0] = c
                grid[r, rows - 1] = c
            else:
                left = random.randint(1, 9)
                right = random.randint(1, 9)
                while right == left:
                    right = random.randint(1, 9)
                grid[r, 0] = left
                grid[r, rows - 1] = right

        return grid

    # (kept for compatibility if you still want it elsewhere, but not used by create_grids now)
    def create_input(self, taskvars, gridvars):
        # default behavior: at least 1 matching row
        return self.create_input_with_match_count(taskvars, match_count=1)

    def transform_input(self, grid, taskvars):
        rows = taskvars['rows']
        output_grid = grid.copy()

        for i in range(rows):
            if grid[i, 0] == grid[i, rows - 1] and grid[i, 0] != 0:
                output_grid[i, :] = grid[i, 0]

        return output_grid

    def _count_matching_rows(self, grid):
        rows = grid.shape[0]
        cnt = 0
        for i in range(rows):
            if grid[i, 0] != 0 and grid[i, 0] == grid[i, rows - 1]:
                cnt += 1
        return cnt

    def create_grids(self):
        rows = random.randint(8, 30)
        taskvars = {'rows': rows}
        max_cells = rows // 2

        n_train = random.randint(3, 4)

        # -----------------------------
        # TRAIN: ensure at least one grid has 0 matching rows
        # -----------------------------
        train_match_counts = []
        train_match_counts.append(0)

        # make the remaining train match-counts (prefer unique & non-zero)
        used = {0}
        while len(train_match_counts) < n_train:
            mc = random.randint(1, max_cells)  # must be >=1 for non-zero train examples
            if mc not in used:
                used.add(mc)
                train_match_counts.append(mc)

        random.shuffle(train_match_counts)

        train_pairs = []
        train_counts_set = set()

        for mc in train_match_counts:
            input_grid = self.create_input_with_match_count(taskvars, match_count=mc)
            output_grid = self.transform_input(input_grid, taskvars)
            train_pairs.append(GridPair(input=input_grid, output=output_grid))
            train_counts_set.add(mc)

        # -----------------------------
        # TEST: matching-row count must differ from ALL train counts
        # Also: because one train has 0, test cannot be 0 -> force >=1
        # -----------------------------
        possible_test_counts = [k for k in range(1, max_cells + 1) if k not in train_counts_set]
        if not possible_test_counts:
            # very rare fallback: if train used all 1..max_cells, re-sample train setup
            return self.create_grids()

        test_mc = random.choice(possible_test_counts)
        test_input = self.create_input_with_match_count(taskvars, match_count=test_mc)
        test_output = self.transform_input(test_input, taskvars)

        # (optional sanity checks; safe to remove)
        assert self._count_matching_rows(test_input) == test_mc
        assert test_mc not in train_counts_set
        assert 0 in train_counts_set

        return taskvars, TrainTestData(train=train_pairs, test=[GridPair(input=test_input, output=test_output)])

import random
from typing import Dict, Any, Tuple, List
import numpy as np

from Framework.arc_task_generator import ARCTaskGenerator, TrainTestData, GridPair
from Framework.transformation_library import find_connected_objects, BorderBehavior
from Framework.input_library import create_object, Contiguity


class Task20270e3bGenerator(ARCTaskGenerator):
    """ARC‑AGI task generator for the vertical puzzle‑piece connection pattern.

    In each generated task the two objects' connector rows are separated by at
    least three rows and by an arbitrary horizontal offset. The solver must
    translate the lower object both upward and sideways so its connector cells
    sit one row above (and column‑aligned with) the upper connector.
    """

    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of different sizes.",
             "Each grid has a completely filled background of {color('background_color')} color, along with exactly two {color('object_color')} objects.",
            "The {color('object_color')} objects consist of 4‑way connected cells and are shaped so that in the output they can be connected like puzzle pieces without overlap or interruption.",
            "Each {color('object_color')} object contains either one single {color('cell_color')} cell or one horizontal strip made of three {color('cell_color')} cells. The {color('cell_color')} cell or strip is 4‑way connected to its respective object.",
            "The {color('cell_color')} cells act as connecting points, where the two objects connect by overlapping at the {color('cell_color')} cells in the output.",
            "If one object has a strip, the other object also has a strip; otherwise, both objects have single cells.",
            "The object positioned above has its {color('cell_color')} connector directly below the main shape, and the object positioned below has its {color('cell_color')} connector directly above the main shape."
        ]

        transformation_reasoning_chain = [
            "The output grid is created by copying the input grid and identifying the two {color('object_color')} puzzle‑like objects.",
            "The lower object is translated so that its {color('cell_color')} connector sits exactly one row above the upper connector and shares the same columns.",
            "All {color('cell_color')} connector cells are recolored to {color('object_color')}, forming one continuous shape.",
            "No rotation or flipping is applied; only translation.",
            "Finally, rows and columns containing no {color('object_color')} cells are trimmed."
        ]

        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    @staticmethod
    def _generate_shape(max_h: int, max_w: int, *, require_bottom_cell: bool = False, require_top_cell: bool = False) -> np.ndarray:
        """Return a 4‑way connected boolean mask satisfying optional row‑presence constraints."""
        while True:
            h = random.randint(2, max_h)
            w = random.randint(2, max_w)
            mask = create_object(h, w, color_palette=1, contiguity=Contiguity.FOUR) != 0
            if require_bottom_cell and not mask[-1, :].any():
                continue
            if require_top_cell and not mask[0, :].any():
                continue
            return mask

    @staticmethod
    def _paste_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int):
        h, w = mask.shape
        grid[top: top + h, left: left + w][mask] = color

    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        bg, obj_col, conn_col = (
            taskvars["background_color"],
            taskvars["object_color"],
            taskvars["cell_color"]
        )
        conn_type = gridvars.get("connection_type", random.choice(["single", "strip"]))
        conn_len = 1 if conn_type == "single" else 3

        height, width = random.randint(16, 24), random.randint(16, 24)
        grid = np.full((height, width), bg, dtype=int)

        # ------------------------ TOP OBJECT -------------------------------------
        while True:
            top_body = self._generate_shape(6, 7, require_bottom_cell=True)
            h1, w1 = top_body.shape
            row_conn_top = random.randint(h1 + 1, height // 2)
            row_start_top = row_conn_top - h1
            col_start_top = random.randint(1, width - w1 - 2)

            bottom_idxs = list(np.where(top_body[-1, :] != 0)[0])
            if conn_len == 1:
                centres = bottom_idxs
            else:
                centres = [i for i in bottom_idxs if i - 1 in bottom_idxs and i + 1 in bottom_idxs]
            if len(centres) == 0:
                continue

            centre = random.choice(centres)
            rel_cols_top = [centre] if conn_len == 1 else [centre - 1, centre, centre + 1]
            abs_cols_top = [col_start_top + c for c in rel_cols_top]
            break

        self._paste_mask(grid, top_body, row_start_top, col_start_top, obj_col)
        grid[row_conn_top, abs_cols_top] = conn_col

        # ------------------------ BOTTOM OBJECT ----------------------------------
        gap = random.randint(3, 7)
        row_conn_bot = row_conn_top + gap

        while True:
            bottom_body = self._generate_shape(6, 7, require_top_cell=True)
            h2, w2 = bottom_body.shape
            top_row_idxs = list(np.where(bottom_body[0, :] != 0)[0])
            if conn_len == 1:
                rel_cols_bot = [random.choice(top_row_idxs)]
            else:
                candidates = [i for i in top_row_idxs if i - 1 in top_row_idxs and i + 1 in top_row_idxs]
                if not candidates:
                    continue
                centre = random.choice(candidates)
                rel_cols_bot = [centre - 1, centre, centre + 1]

            min_offset, max_offset = 1, width - w2 - 2
            feasible_offsets: List[int] = []
            for offset in range(min_offset, max_offset + 1):
                abs_cols_bot = [offset + c for c in rel_cols_bot]
                if abs_cols_bot == abs_cols_top:
                    continue
                delta_col = abs_cols_top[0] - abs_cols_bot[0]
                new_left = offset + delta_col
                new_right = offset + w2 - 1 + delta_col
                if 0 <= new_left and new_right < width:
                    feasible_offsets.append(offset)
            if not feasible_offsets:
                continue
            col_start_bot = random.choice(feasible_offsets)
            abs_cols_bot = [col_start_bot + c for c in rel_cols_bot]
            break

        row_start_bot = row_conn_bot + 1
        needed_h = row_start_bot + h2 + 1
        if needed_h > height:
            pad = needed_h - height
            grid = np.pad(grid, ((0, pad), (0, 0)), constant_values=bg)

        self._paste_mask(grid, bottom_body, row_start_bot, col_start_bot, obj_col)
        grid[row_conn_bot, abs_cols_bot] = conn_col

        return grid

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        # Extract variables immediately to avoid NameError in partial evaluation
        bg = taskvars["background_color"]
        obj_col = taskvars["object_color"]
        conn_col = taskvars["cell_color"]

        out = grid.copy()
        objects = find_connected_objects(out, diagonal_connectivity=False, background=bg, monochromatic=False)
        objects = sorted(objects, key=lambda o: o.bounding_box[0].start)
        if len(objects) != 2:
            raise ValueError("Expected exactly two objects in grid.")
        top_obj, bottom_obj = objects

        conn_rows, conn_cols = np.where(out == conn_col)
        row_top_conn = int(conn_rows.min())
        row_bottom_conn = int(conn_rows.max())
        cols_top = sorted({c for r, c in zip(conn_rows, conn_cols) if r == row_top_conn})
        cols_bot = sorted({c for r, c in zip(conn_rows, conn_cols) if r == row_bottom_conn})
        if len(cols_top) != len(cols_bot):
            raise ValueError("Connector lengths differ between objects.")

        delta_row = (row_top_conn - 1) - row_bottom_conn
        delta_col = cols_top[0] - cols_bot[0]

        bottom_obj.cut(out, background=bg)
        bottom_obj.translate(delta_row, delta_col, border_behavior=BorderBehavior.CLIP, grid_shape=out.shape)
        bottom_obj.paste(out, overwrite=True, background=bg)

        out[out == conn_col] = obj_col

        mask = out == obj_col
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        out = out[rows.min():rows.max() + 1, cols.min():cols.max() + 1]

        return out

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        bg_col, obj_col, conn_col = random.sample(range(1, 10), 3)
        taskvars = {"background_color": bg_col, "object_color": obj_col, "cell_color": conn_col}

        n_train = random.randint(4, 6)
        styles = ["single", "strip"] + [random.choice(["single", "strip"]) for _ in range(n_train - 2)]
        random.shuffle(styles)

        train: List[GridPair] = []
        for st in styles:
            inp = self.create_input(taskvars, {"connection_type": st})
            out = self.transform_input(inp, taskvars)  # direct call avoids scope issues
            train.append({"input": inp, "output": out})

        test_inp = self.create_input(taskvars, {"connection_type": random.choice(["single", "strip"])})
        test_out = self.transform_input(test_inp, taskvars)
        test_pair = {"input": test_inp, "output": test_out}

        return taskvars, {"train": train, "test": [test_pair]}


if __name__ == "__main__":
    gen = PuzzlePieceTaskGenerator()
    _, demo = gen.create_grids()
    ARCTaskGenerator.visualize_train_test_data(demo)

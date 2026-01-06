from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from input_library import create_object, Contiguity, retry
from transformation_library import find_connected_objects, GridObject, GridObjects


class Task7468f01aGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids have a fixed number of columns {vars['cols']}; the number of rows varies across examples.",
            "They contain a colored rectangular object, with all remaining cells being empty (0).",
            "The rectangular object is located within the interior of the grid and never touches the grid border.",
            "The rectangular object contains one or more small colored objects inside it.",
            "These inner objects can be shaped differently and are significantly smaller than the rectangular object.",
            "All objects inside the rectangular object must be of the same color, which is different from the color of the rectangular object.",
            "The colors and size of both the rectangular object and the inner objects vary across examples."
        ]

        transformation_reasoning_chain = [
            "Output grids are constructed by identifying the large rectangular object that contains smaller objects inside it.",
            "The size of the output grid is exactly the same as the size of the rectangular object.",
            "Once the object has been identified, reflect it horizontally and paste it into the output grid.",
            "The reflection results in a mirrored version of the objects inside the rectangular object."
        ]

        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self) -> tuple[dict[str, any], TrainTestData]:
        cols = random.randint(10, 30)
        taskvars = {'cols': cols}

        train_examples = []

        for i in range(3):
            if i == 1:
                num_inners = random.randint(2, 4)
            else:
                num_inners = random.randint(1, 3)

            rows_i = random.randint(10, 30)
            gridvars = {'rows': rows_i, 'num_inner_objects': num_inners}

            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)

            train_examples.append({'input': input_grid, 'output': output_grid})

        def _count_inner_objects(grid: np.ndarray) -> int:
            objs = find_connected_objects(grid, diagonal_connectivity=False, background=0)
            if len(objs) == 0:
                return 0
            rectangle = objs.sort_by_size(reverse=True)[0]
            box = rectangle.bounding_box
            rect_region = grid[box[0], box[1]]
            vals, counts = np.unique(rect_region, return_counts=True)
            nz = [(v, c) for v, c in zip(vals, counts) if v != 0]
            if not nz:
                return 0
            outer_color = max(nz, key=lambda x: x[1])[0]
            mask = (rect_region != outer_color).astype(int)
            inner_objs = find_connected_objects(mask, diagonal_connectivity=False, background=0)
            return len(inner_objs)

        actual_training_counts = set(_count_inner_objects(ex['input']) for ex in train_examples)

        possible_nums = [1, 2, 3, 4]
        test_num = next((n for n in possible_nums if n not in actual_training_counts), None)
        if test_num is None:
            test_num = max(possible_nums) + 1

        test_rows = random.randint(10, 30)
        test_gridvars = {'rows': test_rows, 'num_inner_objects': test_num}

        max_test_attempts = 30
        test_input = None
        for attempt in range(max_test_attempts):
            candidate = self.create_input(taskvars, test_gridvars)
            actual = _count_inner_objects(candidate)
            if actual == test_num:
                test_input = candidate
                break

            if attempt == max_test_attempts - 1:
                remaining = [n for n in possible_nums if n not in actual_training_counts]
                if remaining:
                    test_num = remaining[0]
                    test_gridvars = {'rows': test_rows, 'num_inner_objects': test_num}

        if test_input is None:
            test_input = self.create_input(taskvars, test_gridvars)

        test_output = self.transform_input(test_input, taskvars)

        return taskvars, {
            'train': train_examples,
            'test': [{'input': test_input, 'output': test_output}]
        }

    # ---------- helpers for inner object shapes ----------

    def _shape_library(self) -> list[list[tuple[int, int]]]:
        """
        Each shape is a list of (dr, dc) offsets.
        All shapes have >= 2 cells, and several are clearly non-rectangular to make flips obvious.
        """
        return [
            # 2-cells
            [(0, 0), (0, 1)],  # horizontal domino
            [(0, 0), (1, 0)],  # vertical domino

            # 3-cells L
            [(0, 0), (1, 0), (1, 1)],
            [(0, 1), (1, 1), (1, 0)],

            # 4-cells L
            [(0, 0), (1, 0), (2, 0), (2, 1)],
            [(0, 1), (1, 1), (2, 1), (2, 0)],

            # 4-cells T (not symmetric under horizontal flip if positioned/combined with others, still good)
            [(0, 0), (0, 1), (0, 2), (1, 1)],

            # 4-cells zigzag
            [(0, 0), (0, 1), (1, 1), (1, 2)],
            [(0, 1), (0, 2), (1, 0), (1, 1)],

            # 5-cells "P" like (asymmetric)
            [(0, 0), (0, 1), (1, 0), (2, 0), (1, 1)],
        ]

    def _shape_bbox(self, offsets: list[tuple[int, int]]) -> tuple[int, int]:
        rs = [dr for dr, _ in offsets]
        cs = [dc for _, dc in offsets]
        h = max(rs) - min(rs) + 1
        w = max(cs) - min(cs) + 1
        return h, w

    def _normalize_shape(self, offsets: list[tuple[int, int]]) -> list[tuple[int, int]]:
        min_r = min(dr for dr, _ in offsets)
        min_c = min(dc for _, dc in offsets)
        return sorted([(dr - min_r, dc - min_c) for dr, dc in offsets])

    def _hflip_shape(self, offsets: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Horizontal flip within the shape's own bounding box."""
        norm = self._normalize_shape(offsets)
        _, w = self._shape_bbox(norm)
        flipped = [(dr, (w - 1) - dc) for dr, dc in norm]
        return sorted(self._normalize_shape(flipped))

    def _is_self_horiz_symmetric(self, offsets: list[tuple[int, int]]) -> bool:
        norm = self._normalize_shape(offsets)
        return norm == self._hflip_shape(norm)

    def create_input(self, taskvars: dict[str, any], gridvars: dict[str, any]) -> np.ndarray:
        rows = gridvars.get('rows', random.randint(10, 30))
        cols = taskvars['cols']

        grid = np.zeros((rows, cols), dtype=int)

        outer_color = random.randint(1, 9)
        inner_color = random.choice([c for c in range(1, 10) if c != outer_color])

        rect_height = random.randint(6, min(rows - 4, 15))  # slightly larger to support richer shapes
        rect_width = random.randint(6, min(cols - 4, 15))

        row_start = random.randint(1, rows - rect_height - 1)
        col_start = random.randint(1, cols - rect_width - 1)

        grid[row_start:row_start + rect_height, col_start:col_start + rect_width] = outer_color

        num_inner_objects = gridvars.get('num_inner_objects', random.randint(1, 3))

        # We enforce: at least one inner object is clearly "flip-obvious"
        # (either non-rect / non-self-symmetric), AND the whole rectangle content is not horizontally symmetric.
        max_global_attempts = 50
        shapes = self._shape_library()

        for _global_try in range(max_global_attempts):
            # reset the rectangle interior to outer_color before placing inner objects
            grid[row_start:row_start + rect_height, col_start:col_start + rect_width] = outer_color

            inner_positions = []
            placed = 0
            placed_has_flip_obvious_shape = False

            for k in range(num_inner_objects):
                # bias towards non-rect / more complex shapes
                # and ensure at least one flip-obvious shape exists
                if (not placed_has_flip_obvious_shape) and (k == num_inner_objects - 1):
                    # force a shape that is NOT self horizontally symmetric
                    candidates = [s for s in shapes if not self._is_self_horiz_symmetric(s)]
                    shape = random.choice(candidates) if candidates else random.choice(shapes)
                else:
                    # usually pick from full library, but slightly prefer non-symmetric ones
                    if random.random() < 0.7:
                        candidates = [s for s in shapes if not self._is_self_horiz_symmetric(s)]
                        shape = random.choice(candidates) if candidates else random.choice(shapes)
                    else:
                        shape = random.choice(shapes)

                shape = self._normalize_shape(shape)
                sh_h, sh_w = self._shape_bbox(shape)

                # place within rectangle with 1-cell padding from rectangle border
                max_attempts = 250
                success = False
                for _ in range(max_attempts):
                    inner_row = random.randint(row_start + 1, row_start + rect_height - sh_h - 1)
                    inner_col = random.randint(col_start + 1, col_start + rect_width - sh_w - 1)

                    # compute candidate bbox
                    cand_top = inner_row
                    cand_left = inner_col
                    cand_bottom = inner_row + sh_h - 1
                    cand_right = inner_col + sh_w - 1

                    # check separation from existing inner objects (1-cell expanded bbox)
                    too_close = False
                    for (pr, pc, ph, pw) in inner_positions:
                        exp_top = pr - 1
                        exp_left = pc - 1
                        exp_bottom = pr + ph
                        exp_right = pc + pw
                        if not (cand_bottom < exp_top or cand_top > exp_bottom or cand_right < exp_left or cand_left > exp_right):
                            too_close = True
                            break

                    if too_close:
                        continue

                    # place shape cells
                    for dr, dc in shape:
                        grid[inner_row + dr, inner_col + dc] = inner_color

                    inner_positions.append((inner_row, inner_col, sh_h, sh_w))
                    placed += 1
                    if not self._is_self_horiz_symmetric(shape):
                        placed_has_flip_obvious_shape = True
                    success = True
                    break

                if not success:
                    # couldn't place this one; continue trying others (the counting logic later handles mismatch)
                    continue

            # If we failed to place at least one inner object, try again globally (rare, but possible)
            if placed == 0:
                continue

            # Ensure the rectangle region is NOT horizontally symmetric (so the flip changes something)
            rect_region = grid[row_start:row_start + rect_height, col_start:col_start + rect_width]
            if np.array_equal(rect_region, np.fliplr(rect_region)):
                continue

            # Ensure at least one inner object is >=2 cells AND "flip-obvious" (non-self-symmetric)
            if not placed_has_flip_obvious_shape:
                continue

            return grid

        # Fallback: return whatever we have (should be extremely rare)
        return grid

    def transform_input(self, grid: np.ndarray, taskvars: dict[str, any]) -> np.ndarray:
        objects = find_connected_objects(grid, diagonal_connectivity=False, background=0)
        if len(objects) == 0:
            return np.zeros((1, 1), dtype=int)

        rectangle = objects.sort_by_size(reverse=True)[0]
        box = rectangle.bounding_box
        rect_region = grid[box[0], box[1]].copy()

        return np.fliplr(rect_region)

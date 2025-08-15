from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from input_library import create_object, retry, random_cell_coloring
from transformation_library import find_connected_objects
import numpy as np
import random
import math


class Task17cae0c1(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size 3x{vars['cols']}.",
            "Each grid can be completely divided into 3x3 subgrids, starting from the top-left corner and continuing to the bottom-right.",
            "Each 3x3 subgrid contains a specifically shaped object.",
            "There are five possible shapes; A 3x3 block with the center cell empty. A single filled cell at position (1,1) of the 3x3 grid. The entire first row of the 3x3 grid is filled. The entire last row of the 3x3 grid is filled. The main diagonal (from top-left to bottom-right) is filled.",
            "All filled cells use the color {color('object_color')}.",
            "Across all training examples, all five shapes must appear at least once.",
            "A single training example does not need to include all five shapes, and shapes may repeat across or within examples."
        ]

        transformation_reasoning_chain = [
            "The output grid is constructed by starting with a zero-filled grid same-sized as input and observing all objects in the input, which are divided into 3x3 subgrids.",
            "There are five possible shapes: Shape 1 is a 3x3 block with the center cell empty. Shape 2 is a single filled cell at position (1,1) of the 3x3 grid. Shape 3 is the entire first row of the 3x3 grid filled. Shape 4 is the entire last row of the 3x3 grid filled. Shape 5 is the main diagonal (from top-left to bottom-right) filled.",
            "Based on the shape of each 3x3 subgrid in the input, the corresponding 3x3 subgrid in the output is filled with a specific color.",
            "The shape-to-color mapping is as follows: Shape 1-> {color('fill_color1')}, Shape 2-> {color('fill_color2')}, Shape 3-> {color('fill_color3')}, Shape 4-> {color('fill_color4')}, Shape 5->{color('fill_color5')}."
        ]

        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def _create_shape(self, shape_type: int, color: int) -> np.ndarray:
        """Create a 3x3 shape of the specified type with the given color."""
        shape = np.zeros((3, 3), dtype=int)

        if shape_type == 1:  # 3x3 block with center empty
            shape.fill(color)
            shape[1, 1] = 0
        elif shape_type == 2:  # Single cell at (1,1)
            shape[1, 1] = color
        elif shape_type == 3:  # First row filled
            shape[0, :] = color
        elif shape_type == 4:  # Last row filled
            shape[2, :] = color
        elif shape_type == 5:  # Main diagonal filled
            shape[0, 0] = color
            shape[1, 1] = color
            shape[2, 2] = color

        return shape

    def create_input(self, taskvars, gridvars):
        """Create an input grid with 3 x cols dimensions containing 3x3 subgrids with shapes."""
        cols = taskvars['cols']
        object_color = taskvars['object_color']
        assert cols % 3 == 0, f"cols must be multiple of 3, got {cols}"

        num_subgrids = cols // 3

        # Decide shapes to place
        if 'shapes' in gridvars:
            shapes = list(gridvars['shapes'])
            # normalize length
            if len(shapes) > num_subgrids:
                shapes = shapes[:num_subgrids]
            elif len(shapes) < num_subgrids:
                shapes += [random.randint(1, 5) for _ in range(num_subgrids - len(shapes))]
        else:
            shapes = [random.randint(1, 5) for _ in range(num_subgrids)]

        # Build grid
        grid = np.zeros((3, cols), dtype=int)
        for i, shape_type in enumerate(shapes):
            start_col = i * 3
            end_col = start_col + 3
            if end_col > cols:
                break
            shape = self._create_shape(shape_type, object_color)
            grid[:, start_col:end_col] = shape

        return grid

    def transform_input(self, grid: np.ndarray, taskvars) -> np.ndarray:
        """Transform input grid by mapping shapes to their corresponding colors."""
        fill_colors = [
            taskvars['fill_color1'],  # Shape 1
            taskvars['fill_color2'],  # Shape 2
            taskvars['fill_color3'],  # Shape 3
            taskvars['fill_color4'],  # Shape 4
            taskvars['fill_color5']   # Shape 5
        ]

        output_grid = np.zeros_like(grid)
        cols = grid.shape[1]
        num_subgrids = cols // 3

        for i in range(num_subgrids):
            start_col = i * 3
            end_col = start_col + 3
            subgrid = grid[:, start_col:end_col]
            shape_type = self._identify_shape(subgrid)
            if shape_type > 0:
                output_grid[:, start_col:end_col] = fill_colors[shape_type - 1]

        return output_grid

    def _identify_shape(self, subgrid: np.ndarray) -> int:
        """Identify which of the 5 shapes a 3x3 subgrid represents."""
        non_zero_positions = {(r, c) for r in range(3) for c in range(3) if subgrid[r, c] != 0}
        if not non_zero_positions:
            return 0

        shape1_positions = {(0, 0), (0, 1), (0, 2),
                            (1, 0),           (1, 2),
                            (2, 0), (2, 1), (2, 2)}
        if non_zero_positions == shape1_positions:
            return 1
        if non_zero_positions == {(1, 1)}:
            return 2
        if non_zero_positions == {(0, 0), (0, 1), (0, 2)}:
            return 3
        if non_zero_positions == {(2, 0), (2, 1), (2, 2)}:
            return 4
        if non_zero_positions == {(0, 0), (1, 1), (2, 2)}:
            return 5

        return 0

    def _batch_required_shapes(self, num_subgrids: int):
        """
        Partition the 5 required shapes into batches of size num_subgrids.
        Each batch becomes one deterministic training example to ensure coverage.
        """
        required = [1, 2, 3, 4, 5]
        batches = []
        # Split list into chunks of length num_subgrids
        for i in range(0, len(required), num_subgrids):
            batches.append(required[i:i + num_subgrids])
        return batches  # e.g., if num_subgrids=2 -> [[1,2],[3,4],[5]]

    def create_grids(self):
        """Create training and test grids ensuring all 5 shapes appear at least once across training."""
        cols = random.choice([6, 9, 12, 15, 18, 21])
        assert cols % 3 == 0, f"cols must be multiple of 3, got {cols}"
        num_subgrids = cols // 3

        # Colors
        all_colors = list(range(1, 10))
        random.shuffle(all_colors)
        taskvars = {
            'cols': cols,
            'object_color': all_colors[0],
            'fill_color1': all_colors[1],
            'fill_color2': all_colors[2],
            'fill_color3': all_colors[3],
            'fill_color4': all_colors[4],
            'fill_color5': all_colors[5]
        }

        train_pairs = []

        # 1) Deterministic coverage batches to guarantee all 5 shapes appear
        coverage_batches = self._batch_required_shapes(num_subgrids)
        for batch in coverage_batches:
            # pad batch to full width with random shapes if needed
            shapes = list(batch)
            while len(shapes) < num_subgrids:
                shapes.append(random.randint(1, 5))
            random.shuffle(shapes)

            gridvars = {'shapes': shapes}
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            train_pairs.append({'input': input_grid, 'output': output_grid})

        # 2) Add 0–3 more random training examples (so total 3–6, like before)
        extra_needed_min = max(0, 3 - len(train_pairs))
        extra_allowed = max(0, 6 - len(train_pairs))
        extra_count = random.randint(extra_needed_min, extra_allowed)

        for _ in range(extra_count):
            shapes = [random.randint(1, 5) for _ in range(num_subgrids)]
            gridvars = {'shapes': shapes}
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            train_pairs.append({'input': input_grid, 'output': output_grid})

        # 3) Validate that training covers all 5 shapes
        observed = set()
        for pair in train_pairs:
            grid = pair['input']
            for i in range(num_subgrids):
                sub = grid[:, i*3:(i+1)*3]
                st = self._identify_shape(sub)
                if st > 0:
                    observed.add(st)

        assert observed.issuperset({1, 2, 3, 4, 5}), (
            f"Training coverage insufficient: observed {sorted(observed)}"
        )

        # Test example
        test_shapes = [random.randint(1, 5) for _ in range(num_subgrids)]
        test_gridvars = {'shapes': test_shapes}
        test_input = self.create_input(taskvars, test_gridvars)
        test_output = self.transform_input(test_input, taskvars)
        test_pairs = [{'input': test_input, 'output': test_output}]

        # Final sanity checks (these help avoid vague upstream errors)
        assert len(train_pairs) >= 3, "Need at least 3 training pairs"
        assert len(test_pairs) == 1, "Expect exactly one test pair"
        assert train_pairs[0]['input'].shape == train_pairs[0]['output'].shape, "Train I/O size mismatch"
        assert test_pairs[0]['input'].shape == test_pairs[0]['output'].shape, "Test I/O size mismatch"

        return taskvars, {'train': train_pairs, 'test': test_pairs}

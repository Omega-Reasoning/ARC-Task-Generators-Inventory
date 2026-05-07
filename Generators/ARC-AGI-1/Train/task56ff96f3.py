from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from typing import Dict, List, Any, Tuple


class Task56ff96f3Generator(ARCTaskGenerator):

    def __init__(self):
        input_reasoning_chain = [
            "The input grids are of size {vars['rows']}x{vars['cols']}.",
            "The grid contains 1 or 2 pairs of isolated single-cell dots, each pair sharing a unique color.",
            "Each pair of dots is placed on a background of 0 (empty) cells.",
            "The two dots of each color are separated by some distance, forming potential opposite corners of a rectangle.",
        ]

        transformation_reasoning_chain = [
            "The output grid is the same size as the input grid.",
            "For each pair of same-colored dots, identify the bounding rectangle defined by their two positions.",
            "The rectangle spans from the minimum row to the maximum row and minimum column to the maximum column of the two dots.",
            "Fill the entire bounding rectangle (all cells within) with the color of the dots.",
            "Repeat for every color pair present in the input.",
            "All other cells remain as background (0).",
        ]

        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        rows = taskvars['rows']
        cols = taskvars['cols']
        
        # Get colors from taskvars or gridvars
        colors = gridvars.get('colors')
        if colors is None:
            available_colors = list(range(1, 10))
            colors = random.sample(available_colors, 1)
        
        num_pairs = len(colors)
        
        # Generate pairs with non-overlapping rectangles
        used_rects = []
        pairs = []
        
        for attempt in range(200):
            if len(pairs) == num_pairs:
                break
            
            color = colors[len(pairs)]
            
            # Try random dot placement
            for _ in range(100):
                r1, c1 = random.randint(0, rows - 1), random.randint(0, cols - 1)
                r2, c2 = random.randint(0, rows - 1), random.randint(0, cols - 1)
                
                # Ensure distinct positions and minimum rectangle size (at least 2x2)
                if r1 == r2 or c1 == c2:
                    continue
                if abs(r1 - r2) < 1 or abs(c1 - c2) < 1:
                    continue
                
                min_r, max_r = min(r1, r2), max(r1, r2)
                min_c, max_c = min(c1, c2), max(c1, c2)
                
                # Rectangle must have area >= 4
                if (max_r - min_r + 1) * (max_c - min_c + 1) < 4:
                    continue
                
                # Check no overlap with existing rectangles (including 1-cell border)
                overlap = False
                for (er, er2, ec, ec2) in used_rects:
                    # Expand existing rect by 1 to enforce separation
                    if not (max_r < er - 1 or min_r > er2 + 1 or
                            max_c < ec - 1 or min_c > ec2 + 1):
                        overlap = True
                        break
                
                if not overlap:
                    used_rects.append((min_r, max_r, min_c, max_c))
                    pairs.append((color, (r1, c1), (r2, c2)))
                    break
        
        # Create grid with generated pairs
        grid = np.zeros((rows, cols), dtype=int)
        for color, (r1, c1), (r2, c2) in pairs:
            grid[r1, c1] = color
            grid[r2, c2] = color
        return grid

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        output = np.zeros_like(grid)
        rows, cols = grid.shape

        # Find all non-zero cells and group by color
        color_positions = {}
        for r in range(rows):
            for c in range(cols):
                v = grid[r, c]
                if v != 0:
                    color_positions.setdefault(v, []).append((r, c))

        # For each color with exactly 2 cells, fill the bounding rectangle
        for color, positions in color_positions.items():
            if len(positions) == 2:
                (r1, c1), (r2, c2) = positions
                min_r, max_r = min(r1, r2), max(r1, r2)
                min_c, max_c = min(c1, c2), max(c1, c2)
                output[min_r:max_r + 1, min_c:max_c + 1] = color

        return output

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Choose grid size
        rows = random.randint(6, 30)
        cols = random.randint(6, 30)

        def make_example(num_pairs_this_example):
            for _ in range(50):
                # Pick distinct non-zero colors
                available_colors = list(range(1, 10))
                chosen_colors = random.sample(available_colors, num_pairs_this_example)

                example_taskvars = {
                    'rows': rows,
                    'cols': cols,
                }

                gridvars = {'colors': chosen_colors}

                input_grid = self.create_input(example_taskvars, gridvars)
                output_grid = self.transform_input(input_grid, example_taskvars)

                return {
                    'input': input_grid,
                    'output': output_grid,
                    'taskvars': example_taskvars
                }
            return None

        # ----------------------------
        # TRAIN MUST COVER BOTH CASES:
        # 1 pair example(s)
        # 2 pair example(s)
        # ----------------------------
        nr_train = random.randint(3, 5)

        train_examples = []
        taskvars = None

        # Force one example with 1 pair
        ex1 = make_example(1)

        # Force one example with 2 pairs
        ex2 = make_example(2)

        required_examples = [ex1, ex2]

        for ex in required_examples:
            if ex is not None:
                if taskvars is None:
                    taskvars = ex['taskvars']
                train_examples.append({
                    'input': ex['input'],
                    'output': ex['output']
                })

        # Fill remaining train examples randomly
        while len(train_examples) < nr_train:
            num_pairs_this_example = random.randint(1, 2)
            ex = make_example(num_pairs_this_example)
            if ex is not None:
                train_examples.append({
                    'input': ex['input'],
                    'output': ex['output']
                })

        # Test example can be either case
        test_num_pairs = random.randint(1, 2)
        test_example = make_example(test_num_pairs)

        test_examples = []
        if test_example is not None:
            test_examples.append({
                'input': test_example['input'],
                'output': test_example['output']
            })

        train_test_data = {
            'train': train_examples,
            'test': test_examples,
        }

        return taskvars, train_test_data
from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import GridObject, GridObjects, find_connected_objects
from input_library import create_object, retry, Contiguity
import numpy as np
import random
from typing import Dict, Any, List, Tuple, Set

class Task0e206a2eGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of different sizes.",
            "Each grid contains 1 or 2 objects, where each object is made of 4-way connected cells using four colors, along with some single-colored cells, and the remaining cells are empty (0).",
            "The grid is constructed by first creating 1 or 2 objects, each made using one {color('color1')} cell, one {color('color2')} cell, one randomly chosen color (unique per grid), and the remaining cells filled with a base color that is consistent within the grid but varies across grids.",
            "The objects are constructed by 4-way connecting the colored cells in such a way that the three colored cells—one {color('color1')} cell, one {color('color2')} cell, and one randomly chosen color—are never directly connected to each other and are always separated by cells of the base color.",
            "Once the objects are created, each is copied and pasted into an empty space in the grid—either directly or in a rotated version. All base color cells are removed in the copied version.",
            "If there are two objects, ensure that the positions of the {color('color1')}, {color('color2')}, and the randomly colored cells are different from each other."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid and identifying all colored objects, which are made of 4-way connected cells using exactly 4 different colors; one {color('color1')} cell, one {color('color2')} cell, one randomly chosen color, and the remaining cells filled with a base color.",
            "After identifying the colored objects, locate the group or groups of single colored cells.",
            "In the case of 2 objects, there will be 2 groups of single colored cells; in the case of 1 object, there will be 1 group.",
            "Note that the single colored cells may appear in a rotated position and may not directly match the original object, so always compare based on relative positioning.",
            "Once the single colored cells are found, match their relative positions to the positions of the colored cells within the objects.",
            "Then, use the single colored cells as reference points and add base color cells around them to match the shape of one of the objects in the grid.",
            "Once the new objects are formed, remove the original objects from the grid."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_object_with_constraints(self, color1: int, color2: int, color3: int, base_color: int, max_size: int = 5) -> np.ndarray:
        """Create a smaller object where color1, color2, and color3 are not directly connected."""
        def create_candidate():
            # Use smaller grid size
            size = random.randint(4, max_size)
            obj = np.zeros((size, size), dtype=int)
            
            # Place the three special colors ensuring they're separated
            # Start with color1 at a random position
            r1, c1 = random.randint(1, size-2), random.randint(1, size-2)
            obj[r1, c1] = color1
            
            # Place color2 at least 2 cells away
            attempts = 0
            while attempts < 50:
                r2, c2 = random.randint(0, size-1), random.randint(0, size-1)
                if abs(r2-r1) + abs(c2-c1) >= 2:
                    obj[r2, c2] = color2
                    break
                attempts += 1
            
            # Place color3 at least 2 cells away from both
            attempts = 0
            while attempts < 50:
                r3, c3 = random.randint(0, size-1), random.randint(0, size-1)
                if abs(r3-r1) + abs(c3-c1) >= 2 and abs(r3-r2) + abs(c3-c2) >= 2:
                    obj[r3, c3] = color3
                    break
                attempts += 1
            
            # Connect them with base color cells
            # Add base color cells around and between the special colors
            positions = [(r1, c1), (r2, c2), (r3, c3)]
            
            # Add base color cells adjacent to special colors
            for r, c in positions:
                for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < size and 0 <= nc < size and obj[nr, nc] == 0:
                        obj[nr, nc] = base_color
            
            # Add fewer base color cells to keep object small
            for _ in range(random.randint(2, 6)):  # Even fewer base cells
                # Find cells adjacent to existing non-zero cells
                non_zero = [(r, c) for r in range(size) for c in range(size) if obj[r, c] != 0]
                if non_zero:
                    r, c = random.choice(non_zero)
                    for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < size and 0 <= nc < size and obj[nr, nc] == 0:
                            obj[nr, nc] = base_color
                            break
            
            # Extract minimal bounding box
            non_zero = np.argwhere(obj != 0)
            if len(non_zero) == 0:
                return None
            
            min_r, min_c = non_zero.min(axis=0)
            max_r, max_c = non_zero.max(axis=0)
            
            result = obj[min_r:max_r+1, min_c:max_c+1]
            
            # Ensure object isn't too large
            if result.shape[0] > max_size or result.shape[1] > max_size:
                return None
                
            return result
        
        def validate_object(obj):
            if obj is None:
                return False
            
            # Check size constraints
            if obj.shape[0] > max_size or obj.shape[1] > max_size:
                return False
            
            # Check that we have exactly one of each special color
            if (np.sum(obj == color1) != 1 or 
                np.sum(obj == color2) != 1 or 
                np.sum(obj == color3) != 1):
                return False
            
            # Check that special colors are not adjacent
            dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            
            # Get positions
            pos1 = tuple(np.argwhere(obj == color1)[0])
            pos2 = tuple(np.argwhere(obj == color2)[0])  
            pos3 = tuple(np.argwhere(obj == color3)[0])
            
            # Check adjacencies
            for dr, dc in dirs:
                # Check color1 neighbors
                nr, nc = pos1[0] + dr, pos1[1] + dc
                if 0 <= nr < obj.shape[0] and 0 <= nc < obj.shape[1]:
                    if obj[nr, nc] in [color2, color3]:
                        return False
                
                # Check color2 neighbors
                nr, nc = pos2[0] + dr, pos2[1] + dc
                if 0 <= nr < obj.shape[0] and 0 <= nc < obj.shape[1]:
                    if obj[nr, nc] == color3:
                        return False
            
            # Check 4-way connectivity
            objects = find_connected_objects(obj, diagonal_connectivity=False, background=0, monochromatic=False)
            if len(objects) != 1:
                return False
            
            # Ensure object has at least 6 cells total (3 special + some base)
            if np.sum(obj != 0) < 6:
                return False
            
            return True
        
        return retry(create_candidate, validate_object, max_attempts=100)
    
    def mark_occupied_area(self, occupied: np.ndarray, obj: np.ndarray, r: int, c: int, buffer: int = 2):
        """Mark an area as occupied including larger buffer zone."""
        for dr in range(-buffer, obj.shape[0] + buffer):
            for dc in range(-buffer, obj.shape[1] + buffer):
                gr, gc = r + dr, c + dc
                if 0 <= gr < occupied.shape[0] and 0 <= gc < occupied.shape[1]:
                    occupied[gr, gc] = 1
    
    def mark_occupied_sparse(self, occupied: np.ndarray, sparse_obj: np.ndarray, r: int, c: int, buffer: int = 2):
        """Mark area occupied by sparse object including larger buffer zone."""
        non_zero_positions = np.argwhere(sparse_obj != 0)
        if len(non_zero_positions) == 0:
            return
            
        min_r, min_c = non_zero_positions.min(axis=0)
        max_r, max_c = non_zero_positions.max(axis=0)
        
        for dr in range(min_r - buffer, max_r + 1 + buffer):
            for dc in range(min_c - buffer, max_c + 1 + buffer):
                gr = r + dr - min_r
                gc = c + dc - min_c
                if 0 <= gr < occupied.shape[0] and 0 <= gc < occupied.shape[1]:
                    occupied[gr, gc] = 1
    
    def can_place_object(self, occupied: np.ndarray, obj: np.ndarray, r: int, c: int, buffer: int = 2) -> bool:
        """Check if object can be placed at position with larger buffer."""
        # First check if object fits within grid
        if r + obj.shape[0] > occupied.shape[0] or c + obj.shape[1] > occupied.shape[1]:
            return False
            
        for dr in range(-buffer, obj.shape[0] + buffer):
            for dc in range(-buffer, obj.shape[1] + buffer):
                gr, gc = r + dr, c + dc
                if 0 <= gr < occupied.shape[0] and 0 <= gc < occupied.shape[1]:
                    if occupied[gr, gc] == 1:
                        return False
        return True
    
    def can_place_sparse_with_expansion(self, occupied: np.ndarray, full_obj: np.ndarray, sparse_obj: np.ndarray, 
                                       r: int, c: int, grid_shape: Tuple[int, int], buffer: int = 2) -> bool:
        """Check if sparse object can be placed considering its full expansion will fit in output."""
        # Get the positions of non-zero cells in sparse object
        sparse_positions = np.argwhere(sparse_obj != 0)
        if len(sparse_positions) == 0:
            return False
        
        # Get bounding box of sparse object
        min_r_sparse, min_c_sparse = sparse_positions.min(axis=0)
        max_r_sparse, max_c_sparse = sparse_positions.max(axis=0)
        
        # Get the full object dimensions
        full_h, full_w = full_obj.shape
        
        # Check if the full reconstructed object would fit in the grid
        # We need to ensure that when we place the sparse object at (r, c),
        # the full reconstruction will fit
        for pr in range(full_h):
            for pc in range(full_w):
                gr = r + pr - min_r_sparse
                gc = c + pc - min_c_sparse
                # Check if full reconstruction would go out of bounds
                if gr < 0 or gr >= grid_shape[0] or gc < 0 or gc >= grid_shape[1]:
                    return False
        
        # Also check the standard occupied constraints with buffer
        for dr in range(min_r_sparse - buffer, max_r_sparse + 1 + buffer):
            for dc in range(min_c_sparse - buffer, max_c_sparse + 1 + buffer):
                gr = r + dr - min_r_sparse
                gc = c + dc - min_c_sparse
                if 0 <= gr < occupied.shape[0] and 0 <= gc < occupied.shape[1]:
                    if occupied[gr, gc] == 1:
                        return False
        
        return True
    
    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        """Create input grid with objects and their copies without base color."""
        grid_size = gridvars['grid_size']
        grid = np.zeros((grid_size, grid_size), dtype=int)
        occupied = np.zeros((grid_size, grid_size), dtype=int)  # Track occupied areas
        
        num_objects = gridvars['num_objects']
        color3 = gridvars['color3']
        base_color = gridvars['base_color']
        
        # Calculate max object size based on grid size and number of objects
        max_object_size = 5  # Keep objects small
        
        objects_created = []
        
        for i in range(num_objects):
            if i == 1:
                # For second object, ensure different configuration
                attempts = 0
                while attempts < 50:
                    obj = self.create_object_with_constraints(
                        taskvars['color1'], taskvars['color2'], color3, base_color, max_object_size
                    )
                    
                    # Check if configuration is different from first object
                    if objects_created:
                        first_obj = objects_created[0]
                        
                        # Get relative positions
                        def get_relative_positions(o):
                            p1 = np.argwhere(o == taskvars['color1'])[0]
                            p2 = np.argwhere(o == taskvars['color2'])[0]
                            p3 = np.argwhere(o == color3)[0]
                            return ((p2[0]-p1[0], p2[1]-p1[1]), (p3[0]-p1[0], p3[1]-p1[1]))
                        
                        if get_relative_positions(obj) != get_relative_positions(first_obj):
                            break
                    
                    attempts += 1
            else:
                obj = self.create_object_with_constraints(
                    taskvars['color1'], taskvars['color2'], color3, base_color, max_object_size
                )
            
            objects_created.append(obj)
            
            # Place original object with separation
            placed = False
            for _ in range(300):
                r = random.randint(0, max(0, grid_size - obj.shape[0]))
                c = random.randint(0, max(0, grid_size - obj.shape[1]))
                
                if self.can_place_object(occupied, obj, r, c):
                    grid[r:r+obj.shape[0], c:c+obj.shape[1]] = obj
                    self.mark_occupied_area(occupied, obj, r, c)
                    placed = True
                    break
            
            if not placed:
                raise ValueError(f"Could not place object {i+1} with separation")
            
            # Create copy without base color
            obj_copy = obj.copy()
            obj_copy[obj_copy == base_color] = 0
            
            # Store the original full object for size checking
            full_obj = obj.copy()
            
            # Optionally rotate (but ensure it still fits)
            rotation_applied = 0
            if random.choice([True, False]):
                rotation_applied = random.choice([1, 2, 3])
                obj_copy = np.rot90(obj_copy, rotation_applied)
                full_obj = np.rot90(full_obj, rotation_applied)
            
            # Place copy with separation, ensuring full reconstruction will fit
            placed = False
            for _ in range(300):
                non_zero_positions = np.argwhere(obj_copy != 0)
                if len(non_zero_positions) == 0:
                    break
                    
                min_r, min_c = non_zero_positions.min(axis=0)
                max_r, max_c = non_zero_positions.max(axis=0)
                
                # We need more conservative placement to ensure reconstruction fits
                # Consider the full object size, not just the sparse version
                full_h, full_w = full_obj.shape
                
                # Ensure placement leaves room for full reconstruction
                max_r_pos = max(0, grid_size - full_h - 1)
                max_c_pos = max(0, grid_size - full_w - 1)
                
                if max_r_pos <= 0 or max_c_pos <= 0:
                    break
                
                r = random.randint(0, max_r_pos)
                c = random.randint(0, max_c_pos)
                
                if self.can_place_sparse_with_expansion(occupied, full_obj, obj_copy, r, c, (grid_size, grid_size)):
                    # Place the sparse object
                    for pr, pc in non_zero_positions:
                        gr, gc = r + pr - min_r, c + pc - min_c
                        grid[gr, gc] = obj_copy[pr, pc]
                    
                    # Mark as occupied (use full object size for buffer)
                    self.mark_occupied_sparse(occupied, obj_copy, r, c)
                    placed = True
                    break
            
            if not placed:
                raise ValueError(f"Could not place copy {i+1} with separation")
        
        return grid
    
    def simple_cluster(self, cells: List[Tuple[int, int, int]], num_clusters: int) -> List[List[Tuple[int, int, int]]]:
        """Simple clustering algorithm to group cells based on spatial proximity."""
        if len(cells) < num_clusters * 3:
            return []
        
        # Initialize clusters with farthest points
        clusters = []
        remaining = cells.copy()
        
        # Start first cluster with a random cell
        if remaining:
            first_cell = remaining.pop(random.randint(0, len(remaining) - 1))
            clusters.append([first_cell])
        
        # For each additional cluster, pick the cell farthest from existing clusters
        for _ in range(1, num_clusters):
            if not remaining:
                break
                
            max_dist = -1
            farthest_idx = 0
            
            for i, cell in enumerate(remaining):
                min_dist_to_clusters = float('inf')
                for cluster in clusters:
                    for cluster_cell in cluster:
                        dist = abs(cell[0] - cluster_cell[0]) + abs(cell[1] - cluster_cell[1])
                        min_dist_to_clusters = min(min_dist_to_clusters, dist)
                
                if min_dist_to_clusters > max_dist:
                    max_dist = min_dist_to_clusters
                    farthest_idx = i
            
            clusters.append([remaining.pop(farthest_idx)])
        
        # Assign remaining cells to nearest cluster
        for cell in remaining:
            min_dist = float('inf')
            best_cluster = 0
            
            for i, cluster in enumerate(clusters):
                for cluster_cell in cluster:
                    dist = abs(cell[0] - cluster_cell[0]) + abs(cell[1] - cluster_cell[1])
                    if dist < min_dist:
                        min_dist = dist
                        best_cluster = i
            
            clusters[best_cluster].append(cell)
        
        # Filter clusters that have exactly 3 cells
        return [cluster for cluster in clusters if len(cluster) == 3]
    
    def find_partial_object_groups(self, grid: np.ndarray, color1: int, color2: int, num_objects: int) -> List[Tuple[List[Tuple[int, int, int]], int]]:
        """Find groups of single colored cells that form partial objects and their base color."""
        # Get all non-zero cells
        all_cells = [(r, c, grid[r, c]) for r in range(grid.shape[0]) for c in range(grid.shape[1]) if grid[r, c] != 0]
        
        # Get all complete objects
        complete_objects = []
        all_objects = find_connected_objects(grid, diagonal_connectivity=False, background=0, monochromatic=False)
        for obj in all_objects:
            if len(obj.colors) >= 4:
                complete_objects.append(obj)
        
        # Get base colors from complete objects
        base_colors = set()
        for complete in complete_objects:
            # The base color is the most frequent color
            color_counts = {}
            for _, _, color in complete.cells:
                color_counts[color] = color_counts.get(color, 0) + 1
            # Base color appears most frequently
            base_color = max(color_counts, key=color_counts.get)
            base_colors.add(base_color)
        
        # Find cells that are not part of complete objects
        complete_cells = set()
        for obj in complete_objects:
            complete_cells.update([(r, c) for r, c, _ in obj.cells])
        
        single_cells = [cell for cell in all_cells if (cell[0], cell[1]) not in complete_cells]
        
        # Group single cells by expected number of groups
        groups = []
        
        if num_objects == 1:
            # All single cells form one group
            if len(single_cells) == 3:  # Should have exactly 3 cells
                colors = {cell[2] for cell in single_cells}
                if color1 in colors and color2 in colors:
                    for base in base_colors:
                        groups.append((single_cells, base))
                        break
        else:
            # Use simple clustering
            clusters = self.simple_cluster(single_cells, num_objects)
            
            for i, cluster_cells in enumerate(clusters):
                colors = {cell[2] for cell in cluster_cells}
                if color1 in colors and color2 in colors:
                    # Assign base color
                    base_color = list(base_colors)[i % len(base_colors)] if base_colors else 1
                    groups.append((cluster_cells, base_color))
        
        return groups
    
    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        """Transform the input grid according to the transformation chain."""
        output = grid.copy()
        
        # Count number of complete objects to know how many partial groups to expect
        all_objects = find_connected_objects(output, diagonal_connectivity=False, background=0, monochromatic=False)
        num_complete_objects = sum(1 for obj in all_objects if len(obj.colors) >= 4)
        
        # Find all complete objects
        complete_objects = []
        for obj in all_objects:
            colors = obj.colors
            if len(colors) >= 4 and taskvars['color1'] in colors and taskvars['color2'] in colors:
                complete_objects.append(obj)
        
        # Find partial object groups with their base colors
        partial_groups = self.find_partial_object_groups(output, taskvars['color1'], taskvars['color2'], num_complete_objects)
        
        # For each partial group, reconstruct the complete object
        for partial_cells, base_color in partial_groups:
            # Try to match with each complete object
            matched = False
            
            for complete in complete_objects:
                if matched:
                    break
                
                # Check if this complete object uses the same base color
                if base_color not in complete.colors:
                    continue
                
                # Get the pattern from the complete object
                complete_array = complete.to_array()
                
                # Try all possible transformations
                for flip_h in [False, True]:
                    for flip_v in [False, True]:
                        for rot in range(4):
                            if matched:
                                break
                            
                            # Apply transformations
                            transformed = complete_array.copy()
                            if flip_h:
                                transformed = np.fliplr(transformed)
                            if flip_v:
                                transformed = np.flipud(transformed)
                            transformed = np.rot90(transformed, rot)
                            
                            # Get positions of special colors
                            special_positions = {}
                            for r in range(transformed.shape[0]):
                                for c in range(transformed.shape[1]):
                                    if transformed[r, c] != base_color and transformed[r, c] != 0:
                                        special_positions[transformed[r, c]] = (r, c)
                            
                            # Get positions from partial cells
                            partial_positions = {cell[2]: (cell[0], cell[1]) for cell in partial_cells}
                            
                            # Check if we can match all partial positions
                            can_match = True
                            offset_r, offset_c = None, None
                            
                            for color, (pr, pc) in partial_positions.items():
                                if color not in special_positions:
                                    can_match = False
                                    break
                                
                                sr, sc = special_positions[color]
                                if offset_r is None:
                                    offset_r = pr - sr
                                    offset_c = pc - sc
                                else:
                                    # Check consistency
                                    if pr - sr != offset_r or pc - sc != offset_c:
                                        can_match = False
                                        break
                            
                            if can_match and offset_r is not None:
                                # Add base color cells
                                for r in range(transformed.shape[0]):
                                    for c in range(transformed.shape[1]):
                                        if transformed[r, c] == base_color:
                                            gr = r + offset_r
                                            gc = c + offset_c
                                            if 0 <= gr < output.shape[0] and 0 <= gc < output.shape[1]:
                                                if output[gr, gc] == 0:
                                                    output[gr, gc] = base_color
                                
                                matched = True
                                break
        
        # Remove original complete objects
        for obj in complete_objects:
            obj.cut(output)
        
        return output
    
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        """Create train and test grids."""
        # Initialize task variables
        taskvars = {
            'color1': random.randint(1, 9),
            'color2': random.randint(1, 9)
        }
        
        while taskvars['color2'] == taskvars['color1']:
            taskvars['color2'] = random.randint(1, 9)
        
        # Generate examples
        num_train = random.randint(3, 6)
        train_grids = []
        
        # Ensure we have both 1-object and 2-object examples in training
        num_objects_list = [1, 2] + [random.choice([1, 2]) for _ in range(num_train - 2)]
        random.shuffle(num_objects_list)
        
        for i in range(num_train):
            # Adjust grid size based on number of objects - need more space
            base_size = 20 if num_objects_list[i] == 1 else 25
            gridvars = {
                'grid_size': random.randint(base_size, base_size + 5),
                'num_objects': num_objects_list[i],
                'color3': random.randint(1, 9),
                'base_color': random.randint(1, 9)
            }
            
            # Ensure all colors are different
            all_colors = {taskvars['color1'], taskvars['color2']}
            while gridvars['color3'] in all_colors:
                gridvars['color3'] = random.randint(1, 9)
            all_colors.add(gridvars['color3'])
            
            while gridvars['base_color'] in all_colors:
                gridvars['base_color'] = random.randint(1, 9)
            
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            train_grids.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Generate test example
        num_test_objects = random.choice([1, 2])
        base_size = 20 if num_test_objects == 1 else 25
        test_gridvars = {
            'grid_size': random.randint(base_size, base_size + 5),
            'num_objects': num_test_objects,
            'color3': random.randint(1, 9),
            'base_color': random.randint(1, 9)
        }
        
        all_colors = {taskvars['color1'], taskvars['color2']}
        while test_gridvars['color3'] in all_colors:
            test_gridvars['color3'] = random.randint(1, 9)
        all_colors.add(test_gridvars['color3'])
        
        while test_gridvars['base_color'] in all_colors:
            test_gridvars['base_color'] = random.randint(1, 9)
        
        test_input = self.create_input(taskvars, test_gridvars)
        test_output = self.transform_input(test_input, taskvars)
        
        return taskvars, {
            'train': train_grids,
            'test': [{'input': test_input, 'output': test_output}]
        }


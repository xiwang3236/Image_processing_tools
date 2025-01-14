import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass

@dataclass
class Point:
    """Represents a 2D point."""
    x: float
    y: float

class VertexConverter:
    """Handles conversion between 3D and 2D vertices."""
    
    @staticmethod
    def convert_3d_to_2d(vertices_3d: List[Tuple[float, float, float]]) -> List[Tuple[float, float]]:
        return [(v[0], v[1]) for v in vertices_3d if isinstance(v, tuple)]

class LineIntersector:
    """Handles line intersection calculations."""
    
    @staticmethod
    def find_intersection(line1_start: np.ndarray, line1_end: np.ndarray,
                         line2_start: np.ndarray, line2_end: np.ndarray) -> Optional[np.ndarray]:
        """
        Find intersection point between two line segments.
        
        Args:
            line1_start, line1_end: Start and end points of first line
            line2_start, line2_end: Start and end points of second line
            
        Returns:
            Intersection point if it exists, None otherwise
        """
        v1 = line1_end - line1_start
        v2 = line2_end - line2_start
        
        cross_v1_v2 = np.cross(v1, v2)
        if abs(cross_v1_v2) < 1e-10:
            return None
            
        t = np.cross(line2_start - line1_start, v2) / cross_v1_v2
        if 0 <= t <= 1:
            u = np.cross(line2_start - line1_start, v1) / cross_v1_v2
            if 0 <= u <= 1:
                return line1_start + t * v1
        return None

class TileAnalyzer:
    """Analyzes tile positions and intersections."""
    
    def __init__(self, tile_size: int = 1024):
        self.tile_size = tile_size
    
    def get_all_xy_coordinates(self, vertices: List[Tuple[float, float]]) -> Tuple[List[float], List[float], List[Dict]]:
        """
        Get all X and Y coordinates from vertices and their intersections.
        
        Args:
            vertices: List of 2D vertex coordinates
            
        Returns:
            Tuple of (sorted x-coordinates, sorted y-coordinates, intersection points)
        """
        x_coordinates: Set[float] = set()
        y_coordinates: Set[float] = set()
        intersections = []
        
        for i, vertex in enumerate(vertices):
            x_start, y_start = vertex
            x_end = x_start + self.tile_size
            y_end = y_start + self.tile_size
            
            x_coordinates.update([x_start, x_end])
            y_coordinates.update([y_start, y_end])
            edges1 = self._create_rectangle_edges(x_start, y_start)
            
            for j in range(i + 1, len(vertices)):
                edges2 = self._create_rectangle_edges(*vertices[j])
                self._find_edge_intersections(edges1, edges2, i, j, x_coordinates, y_coordinates, intersections)
        
        return sorted(list(x_coordinates)), sorted(list(y_coordinates)), intersections

    def _find_edge_intersections(self, edges1, edges2, i, j, x_coordinates, y_coordinates, intersections):
        """
        Find intersections between edges and update coordinates and intersections.
        Now includes both x and y coordinates.
        """
        for edge1 in edges1:
            for edge2 in edges2:
                intersection = LineIntersector.find_intersection(
                    np.array(edge1[0]), np.array(edge1[1]),
                    np.array(edge2[0]), np.array(edge2[1])
                )
                if intersection is not None:
                    x_coordinates.add(intersection[0])
                    y_coordinates.add(intersection[1])
                    intersections.append({
                        'x': intersection[0],
                        'y': intersection[1],
                        'tiles': (i+1, j+1)
                    })

    def _create_rectangle_edges(self, x_start: float, y_start: float) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """Create edges for a rectangle given its start coordinates."""
        x_end = x_start + self.tile_size
        y_end = y_start + self.tile_size
        return [
            ((x_start, y_start), (x_end, y_start)),
            ((x_end, y_start), (x_end, y_end)),
            ((x_end, y_end), (x_start, y_end)),
            ((x_start, y_end), (x_start, y_start))
        ]
    


class TilePlotter:
    """Handles visualization of tiles and intersections."""
    
    @staticmethod
    def plot_top_view(vertices: List[Tuple[float, float]], tile_size: int = 1024):
        """Create top view plot of tiles with intersections."""
        fig, ax = plt.subplots(figsize=(12, 12))
        colors = ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan']
        
        for i, vertex in enumerate(vertices):
            TilePlotter._plot_single_tile(ax, vertex, tile_size, colors[i % len(colors)], i+1)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Top View of Tiles with Intersection Points')
        ax.legend()
        ax.set_aspect('equal')
        plt.grid(True)
        plt.show()
    
    @staticmethod
    def _plot_single_tile(ax, vertex, tile_size, color, tile_number):
        """Plot a single tile with its rectangle and annotations."""
        x, y = vertex
        rectangle = plt.Rectangle((x, y), tile_size, tile_size, 
                                alpha=0.2, color=color, label=f'Tile {tile_number}')
        ax.add_patch(rectangle)
        ax.plot([x, x+tile_size, x+tile_size, x, x], 
                [y, y, y+tile_size, y+tile_size, y], 'k-', linewidth=1)

def calculate_target_point(size: int, overlap_percent: float) -> tuple[float, float, float]:

    # Input validation
    if size < 0:
        raise ValueError("Size must be non-negative")
    if not 0 <= overlap_percent <= 100:
        raise ValueError("Overlap percentage must be between 0 and 100")
    
    # Calculate common factor to avoid repetition
    overlap_factor = (100 - (overlap_percent * 0.5)) / 100
    
    # Calculate target points
    target_x = size * overlap_factor
    target_y1 = size * overlap_factor
    target_y2 = target_y1 * 2
    
    return target_x, target_y1, target_y2

# def find_AB_points(vertices: List[Tuple[float, float]], target_point: float) -> Dict:
#     """Find points A and B closest to the target point."""
#     analyzer = TileAnalyzer()
#     x_coordinates, intersections = analyzer.get_all_x_coordinates(vertices)
    
#     left_points = [p for p in x_coordinates if p < target_point]
#     right_points = [p for p in x_coordinates if p > target_point]
    
#     point_A = max(left_points) if left_points else None
#     point_B = min(right_points) if right_points else None
#     middle_point = (point_A + point_B) / 2 if point_A and point_B else None
    
#     return {
#         'point_A': point_A,
#         'point_B': point_B,
#         'middle_point': middle_point,
#         'intersections': intersections
#     }

def find_AB_points(vertices: List[Tuple[float, float]], target_point: Tuple[float, float, float]) -> Dict:
    """Find points A and B closest to each component of target_point.
    
    Args:
        vertices: List of vertex coordinates as (x, y) tuples
        target_point: Tuple of (target_x, target_y1, target_y2)
        
    Returns:
        Dict containing closest points and middle points for each target
    """
    target_x, target_y1, target_y2 = target_point
    
    # Get coordinates using analyzer
    analyzer = TileAnalyzer()
    x_coords, y_coords, intersections = analyzer.get_all_xy_coordinates(vertices)
    
    # Find points for target_x
    left_x = [p for p in x_coords if p < target_x]
    right_x = [p for p in x_coords if p > target_x]
    point_A_x = max(left_x) if left_x else None
    point_B_x = min(right_x) if right_x else None
    middle_x = (point_A_x + point_B_x) / 2 if point_A_x and point_B_x else None
    
    # Find points for target_y1
    left_y1 = [p for p in y_coords if p < target_y1]
    right_y1 = [p for p in y_coords if p > target_y1]
    point_A_y1 = max(left_y1) if left_y1 else None
    point_B_y1 = min(right_y1) if right_y1 else None
    middle_y1 = (point_A_y1 + point_B_y1) / 2 if point_A_y1 and point_B_y1 else None
    
    # Find points for target_y2
    left_y2 = [p for p in y_coords if p < target_y2]
    right_y2 = [p for p in y_coords if p > target_y2]
    point_A_y2 = max(left_y2) if left_y2 else None
    point_B_y2 = min(right_y2) if right_y2 else None
    middle_y2 = (point_A_y2 + point_B_y2) / 2 if point_A_y2 and point_B_y2 else None
    
    return {
        'x': {
            'point_A': point_A_x,
            'point_B': point_B_x,
            'middle_point': middle_x
        },
        'y1': {
            'point_A': point_A_y1,
            'point_B': point_B_y1,
            'middle_point': middle_y1
        },
        'y2': {
            'point_A': point_A_y2,
            'point_B': point_B_y2,
            'middle_point': middle_y2
        }
    }
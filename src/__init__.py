"""
Image processing package for cell segmentation and analysis.

This package provides tools for:
1. Cell segmentation using Cellpose
2. Tile analysis and intersection detection
3. Coordinate system transformations
"""

from .segmentation import CellposeSegmenter
from .reference_line_analysis import (
    VertexConverter,
    LineIntersector,
    TileAnalyzer,
    TilePlotter,
    find_AB_points,
    calculate_target_point,
)
from .map_coordinates import (
    # Data loading and processing
    read_csv_files,
    
    # Plotting functions
    plot_3d_points,
    
    # Coordinate transformation and analysis
    apply_offset_to_tile,
    create_sub_maps_for6,
    create_sub_maps_for4,
    find_lowest_negative
)

# Version info
__version__ = '1.0.0'
__author__ = 'Alexis'

# Define what can be imported with 'from src import *'
__all__ = [
    # Segmentation
    'CellposeSegmenter',
    
    # Analysis
    'VertexConverter',
    'LineIntersector',
    'TileAnalyzer',
    'TilePlotter',
    'find_AB_points',
    'calculate_target_point',
    
    # Data loading and processing
    'read_csv_files',
    
    # Plotting and visualization
    'plot_3d_points',
    
    # Coordinate operations
    'apply_offset_to_tile',
    'create_sub_maps_for6',
    'create_sub_maps_for4',
    'find_lowest_negative'
]


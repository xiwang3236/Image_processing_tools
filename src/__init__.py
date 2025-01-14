"""
Image processing package for cell segmentation and analysis.
"""

# Import only what we currently have
from .segmentation import CellposeSegmenter

# Define what can be imported with 'from src import *'
__all__ = [
    'CellposeSegmenter'
]
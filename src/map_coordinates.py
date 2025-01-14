import os
import pandas as pd
import glob
import matplotlib.pyplot as plt
from typing import List, Tuple  

def read_csv_files(folder_path):
    # Get list of all CSV files in the folder
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
    
    # Dictionary to store dataframes
    dataframes = {}
    
    # Read each CSV file and create a dataframe
    for file_path in csv_files:
        # Extract the tile number from the filename
        # Assuming the pattern is always *_tile{number}_*.csv
        filename = os.path.basename(file_path)
        try:
            tile_num = filename.split('tile')[1].split('_')[0]
            # Create dataframe name
            df_name = f'pd_tile{tile_num}'
            
            # Read the CSV file into a dataframe
            df = pd.read_csv(file_path)
            
            # Store the dataframe in the dictionary
            dataframes[df_name] = df
            
            print(f"Successfully read: {filename} -> {df_name}")
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
    
    return dataframes

def plot_3d_points(df):
    """
    Create a 3D scatter plot of points from a DataFrame
    
    Parameters:
    df (pandas.DataFrame): DataFrame containing columns 'centroid-0', 'centroid-1', 'centroid-2'
    """
    # Create a new 3D figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract coordinates
    z = df['centroid-0']
    y = df['centroid-1']
    x = df['centroid-2']
    
    # Create scatter plot
    scatter = ax.scatter(x, y, z, 
                        # c='purple',
                        c=df['area'],  # Color points by area
                        cmap='viridis',
                        s=1,         # Point size
                        alpha=0.8)     # Transparency
    
    # Add labels and title
    ax.set_xlabel('Z (centroid-0)')
    ax.set_ylabel('Y (centroid-1)')
    ax.set_zlabel('X (centroid-2)')
    ax.set_title('3D Visualization of Centroids')
    
    # Add colorbar
    plt.colorbar(scatter, label='Area')
    
    # Adjust the viewing angle for better visualization
    ax.view_init(elev=20, azim=45)
    
    # Add grid
    ax.grid(True)
    
    return fig

def apply_offset_to_tile(df, tile_number, vertices):
    """
    Apply offset to centroids based on tile number
    df: pandas DataFrame with centroid-0 (z), centroid-1 (x), centroid-2 (y)
    tile_number: number of the tile (1-6)
    """
    # Get the vertex offset for this tile (subtract 1 from tile_number for 0-based indexing)
    offset_x, offset_y, offset_z= vertices[tile_number - 1]
    
    # Create a copy of the DataFrame to avoid modifying the original
    df_offset = df.copy()
    
    # Apply offsets according to the mapping:
    # centroid-0 (z) += offset_z
    # centroid-1 (x) += offset_x
    # centroid-2 (y) += offset_y
    df_offset['centroid-0'] = df_offset['centroid-0'] + offset_z
    df_offset['centroid-1'] = df_offset['centroid-1'] + offset_y
    df_offset['centroid-2'] = df_offset['centroid-2'] + offset_x

    print(f"Offset applied to tile {tile_number}: x add {offset_x}, y add {offset_y}, z add {offset_z}")
    return df_offset

def create_sub_maps_for6(map_tiles: list, x_middle: float, y1_middle: float, y2_middle: float) -> dict:
    """Create sub-maps for 6 tiles based on middle points."""
    if len(map_tiles) != 6:
        raise ValueError("Must provide exactly 6 tile DataFrames")
        
    sub_mapped_dfs = {}
    
    # Create global variables for each sub-map
    global sub_map_tile1, sub_map_tile2, sub_map_tile3, sub_map_tile4, sub_map_tile5, sub_map_tile6
    
    # Filter and create global variables for each tile
    sub_map_tile1 = map_tiles[0][  # Changed from 1 to 0
        (map_tiles[0]['centroid-2'] < x_middle) & 
        (map_tiles[0]['centroid-1'] < y1_middle)
    ]
    
    sub_map_tile2 = map_tiles[1][  # Changed from 2 to 1
        (map_tiles[1]['centroid-2'] > x_middle) & 
        (map_tiles[1]['centroid-1'] < y1_middle)
    ]
    
    sub_map_tile3 = map_tiles[2][  # Changed from 3 to 2
        (map_tiles[2]['centroid-2'] > x_middle) & 
        (map_tiles[2]['centroid-1'] > y1_middle) & 
        (map_tiles[2]['centroid-1'] < y2_middle)
    ]
    
    sub_map_tile4 = map_tiles[3][  # Changed from 4 to 3
        (map_tiles[3]['centroid-2'] < x_middle) & 
        (map_tiles[3]['centroid-1'] > y1_middle) & 
        (map_tiles[3]['centroid-1'] < y2_middle)
    ]
    
    sub_map_tile5 = map_tiles[4][  # Changed from 5 to 4
        (map_tiles[4]['centroid-2'] < x_middle) & 
        (map_tiles[4]['centroid-1'] > y2_middle)
    ]
    
    sub_map_tile6 = map_tiles[5][  # Changed from 6 to 5
        (map_tiles[5]['centroid-2'] > x_middle) & 
        (map_tiles[5]['centroid-1'] > y2_middle)
    ]
    
    # Store in dictionary
    sub_mapped_dfs = {
        'sub_map_tile1': sub_map_tile1,
        'sub_map_tile2': sub_map_tile2,
        'sub_map_tile3': sub_map_tile3,
        'sub_map_tile4': sub_map_tile4,
        'sub_map_tile5': sub_map_tile5,
        'sub_map_tile6': sub_map_tile6
    }
    
    # Print summary
    for df_name, df in sub_mapped_dfs.items():
        print(f"{df_name}: {len(df)} points")
    
    return sub_mapped_dfs

def create_sub_maps_for4(map_tiles: list, x_middle: float, y1_middle: float) -> dict:
    """Create sub-maps for 4 tiles based on middle points."""
    if len(map_tiles) != 4:
        raise ValueError("Must provide exactly 4 tile DataFrames")
        
    sub_mapped_dfs = {}
    
    # Create global variables for each sub-map
    global sub_map_tile1, sub_map_tile2, sub_map_tile3, sub_map_tile4
    
    # Filter and create global variables for each tile
    sub_map_tile1 = map_tiles[0][  # Changed from 1 to 0
        (map_tiles[0]['centroid-2'] < x_middle) & 
        (map_tiles[0]['centroid-1'] < y1_middle)
    ]
    
    sub_map_tile2 = map_tiles[1][  # Changed from 2 to 1
        (map_tiles[1]['centroid-2'] > x_middle) & 
        (map_tiles[1]['centroid-1'] < y1_middle)
    ]
    
    sub_map_tile3 = map_tiles[2][  # Changed from 3 to 2
        (map_tiles[2]['centroid-2'] > x_middle) & 
        (map_tiles[2]['centroid-1'] > y1_middle) 
    ]
    
    sub_map_tile4 = map_tiles[3][  # Changed from 4 to 3
        (map_tiles[3]['centroid-2'] < x_middle) & 
        (map_tiles[3]['centroid-1'] > y1_middle) 
    ]
    

    # Store in dictionary
    sub_mapped_dfs = {
        'sub_map_tile1': sub_map_tile1,
        'sub_map_tile2': sub_map_tile2,
        'sub_map_tile3': sub_map_tile3,
        'sub_map_tile4': sub_map_tile4,
    }
    
    # Print summary
    for df_name, df in sub_mapped_dfs.items():
        print(f"{df_name}: {len(df)} points")
    
    return sub_mapped_dfs

# To concatenate:
def concatenate_maps(sub_mapped_dfs: dict) -> pd.DataFrame:
    """Concatenate all sub-mapped DataFrames into one."""
    all_points = pd.concat(list(sub_mapped_dfs.values()), ignore_index=True)
    print(f"Total number of points after concatenation: {len(all_points)}")
    return all_points


def find_lowest_negative(vertices_3d: List[Tuple[float, float, float]]) -> List[float]:
   """Find the lowest negative value for each axis, if no negative values return 0.
   
   Args:
       vertices_3d: List of 3D coordinates (x, y, z) as tuples
       
   Returns:
       List of minimum negative values [x_min, y_min, z_min] converted to positive,
       or 0 if no negative values exist for that axis
   """
   # Initialize lists to store values for each axis
   x_vals = [x for x, _, _ in vertices_3d]
   y_vals = [y for _, y, _ in vertices_3d]
   z_vals = [z for _, _, z in vertices_3d]
   
   # Find minimum negative value for each axis (if exists)
   x_min = min(x_vals) if any(x < 0 for x in x_vals) else 0
   y_min = min(y_vals) if any(y < 0 for y in y_vals) else 0
   z_min = min(z_vals) if any(z < 0 for z in z_vals) else 0
   
   # Convert negatives to positive values
   return [abs(x_min), abs(y_min), abs(z_min)]
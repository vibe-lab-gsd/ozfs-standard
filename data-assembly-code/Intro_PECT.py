# %% [markdown]
# # The Code Intro for Parcel Edge Classification
# Houpu Li  
# 2025-6-11

# %% [markdown]
# ### Start the time  
# Now, we create a timestamp to record the coding start time.

# %%
import time

# record the start time
start_time = time.time()

# %% [markdown]
# # Step 1: Input the Packages

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import geopandas as gpd
import math

from shapely.geometry import LineString, MultiLineString, GeometryCollection
from shapely.geometry import box
from shapely.geometry import Point
from shapely.ops import unary_union
from shapely.ops import substring
from shapely.ops import linemerge
from shapely.ops import nearest_points
from shapely.ops import split
from shapely import wkt

from scipy.spatial import cKDTree
from rtree import index

from fuzzywuzzy import fuzz

import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)

# %% [markdown]
# # Step 2: Input the Dataset(YOU NEED TO CHANGE THIS PART)

# %%
# you should define the county name by yourself, such as 'Dallas_County', 'Tarrant_County', etc.
County_name = 'Dallas_County'

# %%
# you should define the file path
parcel_path = r'Dallas_County/ParcelView_Dallas.zip'
road_path = r'Dallas_County/tl_2023_48113_roads_Dallas.zip'

# standlize the column names
parcel_cols = {'OBJECTID': 'parcel_id', 'OBJECTID_1': 'parcel_id', 'SITUS_ADDR': 'parcel_addr', 'STAT_LAND_': 'landuse_spec'}
road_cols = {'LINEARID': 'road_id', 'FULLNAME': 'road_addr'}

# read the data
parcel = gpd.read_file(parcel_path)
road = gpd.read_file(road_path)

# %% [markdown]
# # Step 3: Data Preprocessing

# %% [markdown]
# ## Parcels and Roads Cleaning

# %%
# rename the columns
parcel.rename(columns=lambda x: parcel_cols.get(x, x), inplace=True)
road.rename(columns=lambda x: road_cols.get(x, x), inplace=True)

'''parcel address clean and preprocessing'''
# Define a function to extract only the road name part (before the first comma)
def optimize_road_name(situs_addr):
    if pd.isna(situs_addr) or situs_addr.strip() == ', ,':
        return None
    else:
        return situs_addr.split(',')[0].strip()
# Apply the function to the 'SITUS_ADDR' column
parcel['parcel_addr'] = parcel['parcel_addr'].apply(optimize_road_name)
parcel['parcel_addr'] = parcel['parcel_addr'].replace(r'^\s*$', None, regex=True)

'''extract residential area based on specifical landuse'''
parcel['landuse'] = parcel['landuse_spec'].apply(lambda x: 'R' if isinstance(x, str) and x[0] in ['A', 'B'] else None)

'''read the parcel data and data cleanning'''
# Step 1: Transfer the CRS to 4326
parcel = parcel.to_crs(4326)
# Step 2: Create a column to indicate whether 'parcel_addr' or 'landuse' has a value (True/False)
parcel['has_info'] = (~parcel['parcel_addr'].isna()) | (~parcel['landuse'].isna())
# Step 3: Sort the rows by 'has_info' in descending order to prioritize rows with parcel_addr or landuse values
parcel = parcel.sort_values(by='has_info', ascending=False)
# Step 4: Drop duplicates based on geometry, keeping the first occurrence (which now has priority rows at the top)
parcel = parcel.drop_duplicates(subset='geometry')
# Step 5: Drop the 'has_info' column as it's no longer needed
parcel = parcel.drop(columns=['has_info'])
# Step 6: Initialize 'parcel_labeled' column with None values
parcel['parcel_label'] = None
parcel.loc[parcel['parcel_addr'].isna(), 'parcel_label'] = 'parcel without address'
parcel = parcel.reset_index(drop=True)

'''read the road data and data cleanning'''
# Step 1: Transfer the CRS to 4326
road = road.to_crs(4326)
# Step 2: Create a column to indicate whether 'road_addr' has a value (True/False)
road['has_info'] = ~road['road_addr'].isna()
# Step 3: Sort the rows by 'has_info' in descending order to prioritize rows with Situs_Addr or RP values
road = road.sort_values(by='has_info', ascending=False)
# Step 4: Drop duplicates based on geometry, keeping the first occurrence (which now has priority rows at the top)
road = road.drop_duplicates(subset='geometry')
# Step 5: Drop the 'has_info' column as it's no longer needed
road = road.drop(columns=['has_info'])
road = road.reset_index(drop=True)

# %% [markdown]
# ## Attribute Management & Geometry Simplification

# %%
# extract useful columns
parcel = parcel[['Prop_ID','GEO_ID','parcel_id','parcel_addr','landuse','landuse_spec','parcel_label','geometry']]
road = road[['road_id','road_addr','geometry']]

# %%
# explode the MultiPolygons/Multilinestring to Polygons/lintestrings  
parcel = parcel.explode(index_parts=False).reset_index(drop=True)
road = road.explode(index_parts=False).reset_index(drop=True)

# %%
# Group the duplicate parcel_id values and add a suffix.
parcel.loc[parcel['parcel_id'].duplicated(keep=False), 'parcel_id'] = (
    parcel.loc[parcel['parcel_id'].duplicated(keep=False), 'parcel_id'].astype(str) + '_' +
    parcel.loc[parcel['parcel_id'].duplicated(keep=False)].groupby('parcel_id').cumcount().add(1).astype(str)
)

# %% [markdown]
# ## Geometry Extraction

# %%
'''Explode the road into segments'''
# Initialize lists to store line segments and corresponding addresses
line_strings = []
addrs = []
linear_ids = []

# Iterate over rows in road
for idx, row in road.iterrows():
    line = row['geometry']  # Assume this is a LineString geometry
    addr = row['road_addr']
    linear_id = row['road_id']
    
    if line.is_valid and isinstance(line, LineString):
        for i in range(len(line.coords) - 1):
            current_line = LineString([line.coords[i], line.coords[i + 1]])
            line_strings.append(current_line)
            addrs.append(addr)
            linear_ids.append(linear_id)
    else:
        print(f"Invalid or non-LineString geometry detected: {line}")
        
# Create GeoDataFrame for the split road segments
road_seg = gpd.GeoDataFrame({'geometry': line_strings, 'road_addr': addrs, 'road_id': linear_ids}, crs=road.crs)
road_seg = road_seg.to_crs(3857)

# %% [markdown]
# ## Address Matching

# %% [markdown]
# ### 01.Coordinate Transformation

# %%
# transfer the crs to projected crs
parcel = parcel.to_crs(3857)
road = road.to_crs(parcel.crs)

# %% [markdown]
# ### 02.Spatial Filtering Using cKDtree

# %%
# Find the centroid coordinate for each road and parcel
road_seg['x'] = road_seg.geometry.apply(lambda geom: geom.centroid.x)
road_seg['y'] = road_seg.geometry.apply(lambda geom: geom.centroid.y)

parcel['x'] = parcel.geometry.apply(lambda geom: geom.centroid.x)
parcel['y'] = parcel.geometry.apply(lambda geom: geom.centroid.y)

# find the nearest road by using cKDTree
n = 50
tree = cKDTree(road_seg[['x', 'y']])
distances, indices = tree.query(parcel[['x', 'y']], k=n)  # find the nearest n roads

# Create a temporary DataFrame to store the nearest road names
nearest_road_names = pd.DataFrame({
    f'Nearest_Road_{i+1}_Address': road_seg.iloc[indices[:, i]].road_addr.values
    for i in range(n)
})

# Concatenate the new columns with the original DataFrame
parcel = pd.concat([parcel, nearest_road_names], axis=1)
# Drop the x, y column
parcel = parcel.drop(columns=['x', 'y'])

# %% [markdown]
# ### 03.Field Similarity Matching Using Fuzzywuzzy

# %%
def check_and_extract_match_info(row):
    # Remove spaces from parcel_addr
    parcel_addr = row['parcel_addr'].replace(' ', '').lower()
    
    # Dynamically generate a list of the nearest n road names, check if they are not NaN
    road_names = [row[f'Nearest_Road_{i+1}_Address'].replace(' ', '').lower() 
                  if pd.notna(row[f'Nearest_Road_{i+1}_Address']) else '' 
                  for i in range(n)]
    
    # Define a similarity threshold (e.g., 50%)
    threshold = 50
    best_match = None
    best_similarity = 0
    
    # Check each road name and record match information
    for road in road_names:
        if road:  # Only proceed if the road name is not empty
            # Calculate the similarity score using fuzz.partial_ratio
            similarity = fuzz.partial_ratio(parcel_addr, road)
        
            # Keep track of the best match
            if similarity > best_similarity and similarity >= threshold:
                best_similarity = similarity
                best_match = road
    
    if best_match:
        match_segment = best_match  # Matched road segment
        original_road = row[f'Nearest_Road_{road_names.index(best_match) + 1}_Address']  # Original road name with spaces
        return pd.Series([True, match_segment, original_road])
    
    return pd.Series([False, None, None])  # Return False and None if no match found

# Step 1: Ensure 'parcel_addr' has no NaN values before applying the function
parcel_clean = parcel.loc[parcel['parcel_addr'].notna()].copy()
# Step 2: Apply the check_and_extract_match_info function to add new columns
parcel_clean[['Found_Match', 'match_segment', 'match_road_address']] = parcel_clean.apply(check_and_extract_match_info, axis=1)
# Step 3: Merge the newly created columns back into the original 'parcel' DataFrame
parcel = parcel.merge(parcel_clean[['Found_Match', 'match_segment', 'match_road_address']], 
                                          left_index=True, right_index=True, 
                                          how='left')
parcel.loc[parcel['Found_Match'] == False, 'parcel_label'] = 'no_match_address'
# Step 4: Count how many rows have 'Found_Match' == False
len(parcel[parcel['Found_Match'] == False])

# %% [markdown]
# ### 04.Extracted Useful Columns

# %%
parcel = parcel[['Prop_ID','GEO_ID','parcel_id','parcel_addr','landuse','landuse_spec','parcel_label','geometry','Found_Match','match_road_address']]

# %% [markdown]
# #  Step 4: Reshape Geometry  

# %%
# Function to explode Polygons into individual boundary line segments
def explode_to_lines(gdf):
    # Create a list to store new rows
    line_list = []

    for index, row in gdf.iterrows():
        # Get the exterior boundary of the polygon
        exterior = row['geometry'].exterior
        # Convert the boundary into LineString segments
        lines = [LineString([exterior.coords[i], exterior.coords[i + 1]]) 
                 for i in range(len(exterior.coords) - 1)]
        
        # Create new rows for each line segment, retaining the original attributes
        for line in lines:
            new_row = row.copy()
            new_row['geometry'] = line
            line_list.append(new_row)
    
    # Use pd.concat to generate the final GeoDataFrame
    line_gdf = pd.concat(line_list, axis=1).T
    line_gdf = gpd.GeoDataFrame(line_gdf, geometry='geometry', crs=gdf.crs)
    
    return line_gdf

# Call the function to explode the line segments
parcel_seg = explode_to_lines(parcel)

# Reset the index by group
parcel_seg['new_index'] = parcel_seg.groupby('parcel_id').cumcount()
parcel_seg.set_index('new_index', inplace=True)
parcel_seg.index.name = None

# %% [markdown]
# ## Geometry Correction

# %% [markdown]
# ### 01.Calculate the Angle Between Two Lines

# %%
# Function to calculate the bearing of a geometry
def fun_bearing_ra(geom):
    coords = np.array(geom.coords)
    # Use the first and last coordinates to calculate the bearing
    x1, y1 = coords[0]
    x2, y2 = coords[-1]
    
    # Calculate the bearing using atan2
    bearing = math.atan2(y2 - y1, x2 - x1)
    
    return bearing

def calculate_angle_difference(line1, line2):
    bearing1 = fun_bearing_ra(line1)
    bearing2 = fun_bearing_ra(line2)
    # Calculate the absolute angle difference and ensure it is <= 180 degrees
    delta_theta = bearing2 - bearing1
    
    # Ensure the angle is between -π and π
    if delta_theta > math.pi:
        delta_theta -= 2 * math.pi
    elif delta_theta < -math.pi:
        delta_theta += 2 * math.pi
    
    # Convert the angle to degrees
    angle_between_degrees = math.degrees(abs(delta_theta))
    
    # Return the smaller angle difference (angle or its supplement)
    return min(angle_between_degrees, 180 - angle_between_degrees)

# %% [markdown]
# ### 02.Check if Two Segments are Connected

# %%
# Check if two segments share a common point (i.e., their start or end point is the same)
def are_segments_connected(line1, line2):
    coords1 = np.array(line1.coords)
    coords2 = np.array(line2.coords)
    
    # Check if the start or end points of the segments are the same
    if np.all(coords1[0] == coords2[0]) or np.all(coords1[0] == coords2[-1]) or \
       np.all(coords1[-1] == coords2[0]) or np.all(coords1[-1] == coords2[-1]):
        return True
    return False

# %% [markdown]
# ### 03.Re-generate Segment Index Based on Turning Points

# %%
# Function to reorder segments based on the turning point
def reorder_segments_by_turning_point(segments, turning_point_index):
    # Reorder segments starting from the identified turning point
    reordered_segments = segments[turning_point_index:] + segments[:turning_point_index]
    return reordered_segments

# Main function: Process each parcel_id group and return a new GeoDataFrame
def process_parcel_segments(parcel_seg):
    merged_segments = []  # List to store the reordered segments

    # Group the parcel segments by parcel_id and process each group
    for object_id, group in parcel_seg.groupby('parcel_id'):
        segments = group['geometry'].tolist()  # Get the list of line segments for the current group
        original_indices = group.index.tolist()  # Preserve the original indices
        turning_points = []

        # Loop through all adjacent segments to calculate angle differences
        for i in range(1, len(segments)):
            if are_segments_connected(segments[i-1], segments[i]):
                angle_diff = calculate_angle_difference(segments[i-1], segments[i])
                if angle_diff > 15:  # If angle difference is greater than 15 degrees, mark it as a turning point
                    turning_points.append(i)

        # If there are turning points, reorder the segments starting from the first turning point
        if turning_points:
            turning_point_index = turning_points[0]
            reordered_segments = reorder_segments_by_turning_point(segments, turning_point_index)
            reordered_original_indices = reorder_segments_by_turning_point(original_indices, turning_point_index)
        else:
            # If no turning points, retain the original order
            reordered_segments = segments
            reordered_original_indices = original_indices

        # Store the reordered segments and their attributes
        for j, (line, original_index) in enumerate(zip(reordered_segments, reordered_original_indices)):
            row = group.iloc[0].copy()  # Copy the first row's attributes
            row['geometry'] = line
            row['original_index'] = original_index  # Preserve the original index
            row['new_index'] = j  # Assign the new index based on the reordered list
            merged_segments.append(row)

    # Create a new GeoDataFrame for the reordered segments
    updated_gdf = gpd.GeoDataFrame(merged_segments, columns=parcel_seg.columns.tolist() + ['original_index', 'new_index'])
    updated_gdf = updated_gdf.reset_index(drop=True)

    return updated_gdf

# Run the main function and get the new GeoDataFrame
updated_parcel_seg = process_parcel_segments(parcel_seg)
parcel_seg = updated_parcel_seg

# %% [markdown]
# ### 04.Iterative Merging of Connected Segments

# %%
# Group parcel_seg by parcel_id and process each group
merged_segments = []

for object_id, group in parcel_seg.groupby('parcel_id'):
    # Get the list of geometries in the current group
    segments = group.geometry.tolist()
    # Start with the first segment
    merged_lines = [segments[0]]  # Start with the first segment
    
    for i in range(1, len(segments)):
        connected = False
        
        # Always compare the current segment with the previous one
        if are_segments_connected(segments[i-1], segments[i]):
            # Calculate the angle difference between the current segment and the previous one
            angle_diff = calculate_angle_difference(segments[i-1], segments[i])
            
            # If the angle difference is less than 15 degrees, merge the adjacent line segments
            if angle_diff < 15:
                # Merge the current and previous segments
                merged_result = linemerge([merged_lines[-1], segments[i]])
                
                # Check if the result is a MultiLineString, if so, skip the merge
                if isinstance(merged_result, LineString):
                    merged_lines[-1] = merged_result
                    connected = True
                else:
                    # Skip the merge if it's a MultiLineString
                    continue
        
        # If no connected segment is found or the angle difference is too large, add the current segment as a new one
        if not connected:
            merged_lines.append(segments[i])
    
    # Keep the merged results and add other attributes
    for line in merged_lines:
        row = group.iloc[0].copy()  # Copy the first attribute row from the group
        row['geometry'] = line
        merged_segments.append(row)

# Create a new GeoDataFrame from the merged line segments
parcel_seg = gpd.GeoDataFrame(merged_segments, columns=parcel_seg.columns)

# Check for MultiLineString geometries and explode them into LineString
exploded_segments = []

for index, row in parcel_seg.iterrows():
    geom = row['geometry']
    
    if isinstance(geom, MultiLineString):
        # Explode the MultiLineString into individual LineStrings
        for line in geom:
            new_row = row.copy()
            new_row['geometry'] = line
            exploded_segments.append(new_row)
    else:
        # Keep the original LineString geometries
        exploded_segments.append(row)

# Create a new GeoDataFrame from the exploded segments
parcel_seg = gpd.GeoDataFrame(exploded_segments, columns=parcel_seg.columns)

# extract useful columns
parcel_seg.drop(columns=['original_index', 'new_index'], inplace=True)
# Reset the index of the final GeoDataFrame
parcel_seg = parcel_seg.reset_index(drop=True)

# %% [markdown]
# ### 05.divided the curve into two segments

# %%
edge_counts = parcel_seg.groupby('parcel_id').size()
parcel_seg['edge_num'] = parcel_seg['parcel_id'].map(edge_counts)

# %%
# Function to create tangent lines at both ends of a line segment
def create_tangents(line):
    coords = list(line.coords)
    if len(coords) < 2:
        return None, None  # Skip invalid geometries
    
    # Create tangents at the start and end of the line segment
    start_tangent = LineString([coords[0], coords[1]])
    end_tangent = LineString([coords[-2], coords[-1]])
    
    return start_tangent, end_tangent

# Function to filter curve segments based on angle difference of tangents > 30 degrees
def filter_curve_segments(parcel_seg, angle_threshold=30):
    filtered_segments = []
    non_filtered_segments = []
    
    for idx, row in parcel_seg.iterrows():
        line = row['geometry']
        start_tangent, end_tangent = create_tangents(line)
        
        if start_tangent and end_tangent:
            angle_diff = calculate_angle_difference(start_tangent, end_tangent)
            row_dict = row.to_dict()  # Convert the entire row to a dictionary
            row_dict['index'] = idx  # Preserve the original index
            
            if angle_diff > angle_threshold:
                # Add the entire row to the filtered list
                filtered_segments.append(row_dict)
            else:
                # Add the entire row to the non-filtered list
                non_filtered_segments.append(row_dict)
    
    # Create DataFrames with the filtered and non-filtered results if data exists
    if filtered_segments:
        filtered_df = pd.DataFrame(filtered_segments).set_index('index')
        filtered_gdf = gpd.GeoDataFrame(filtered_df, crs=parcel_seg.crs, geometry=filtered_df['geometry'])
    else:
        # Initialize an empty GeoDataFrame with the same structure if no data
        filtered_gdf = gpd.GeoDataFrame(columns=parcel_seg.columns, crs=parcel_seg.crs)
    
    if non_filtered_segments:
        non_filtered_df = pd.DataFrame(non_filtered_segments).set_index('index')
        non_filtered_gdf = gpd.GeoDataFrame(non_filtered_df, crs=parcel_seg.crs, geometry=non_filtered_df['geometry'])
    else:
        # Initialize an empty GeoDataFrame with the same structure if no data
        non_filtered_gdf = gpd.GeoDataFrame(columns=parcel_seg.columns, crs=parcel_seg.crs)
    
    return filtered_gdf, non_filtered_gdf

# Call the function to filter curve segments and create two GeoDataFrames
filtered_parcel_seg, non_filtered_parcel_seg = filter_curve_segments(parcel_seg[parcel_seg['edge_num'] == 3])

# %%
# Function to create tangent lines and reverse the line if necessary
def create_tangents_with_reversal(line):
    coords = list(line.coords)
    if len(coords) < 2:
        return None, None  # Skip invalid geometries
    
    # Find the points with the smallest and largest y-coordinate (latitude)
    if coords[0][1] < coords[-1][1]:  # If the first point's y is smaller, it's the start point
        start_point = coords[0]
        end_point = coords[-1]
    else:  # Otherwise, the last point is the start point
        start_point = coords[-1]
        end_point = coords[0]

    # Reverse the line if start_point is not the same as coords[0]
    if start_point != coords[0]:
        coords.reverse()  # Reverse the order of coordinates
    
    # Now create tangents based on the (possibly reversed) coordinates
    start_tangent = LineString([coords[0], coords[1]])  # Tangent from the first to the second point
    end_tangent = LineString([coords[-2], coords[-1]])  # Tangent from the second last to the last point

    return start_tangent, end_tangent, LineString(coords)  # Return the tangents and the (possibly reversed) line

# Function to calculate the split point based on the 4/5 rule
def calculate_split_point(line, start_tangent, end_tangent, angle_diff, angle_fraction=0.5):
    coords = list(line.coords)

    # Iterate through the line and find the point where the angle difference is approximately 4/5
    for i in range(1, len(coords) - 1):
        intermediate_tangent = LineString([coords[i - 1], coords[i]])
        current_angle_diff = calculate_angle_difference(start_tangent, intermediate_tangent)
        
        if current_angle_diff >= angle_diff * angle_fraction:
            return coords[i]  # Return the split point

    return coords[-1]  # If no point found, return the endpoint

# Function to process each segment in filtered_parcel_seg
def process_filtered_parcel_seg(filtered_parcel_seg, angle_threshold=30, angle_fraction=0.5):
    new_data = []
    
    for idx, row in filtered_parcel_seg.iterrows():
        line = row['geometry']
        
        # Apply the tangent and reversal function
        start_tangent, end_tangent, adjusted_line = create_tangents_with_reversal(line)
        
        if start_tangent and end_tangent:
            angle_diff = calculate_angle_difference(start_tangent, end_tangent)
            
            if angle_diff > angle_threshold:
                # Calculate the split point based on the angle difference and fraction
                split_point = calculate_split_point(adjusted_line, start_tangent, end_tangent, angle_diff, angle_fraction)
                
                # Add split point to row's data
                row_dict = row.to_dict()
                row_dict['split_point'] = Point(split_point)  # Store the split point as geometry
                row_dict['index'] = idx  # Store the original index
                
                new_data.append(row_dict)
            else:
                # If no split needed, just keep the original row
                row_dict = row.to_dict()
                row_dict['split_point'] = None  # No split point, store None
                row_dict['index'] = idx  # Store the original index
                
                new_data.append(row_dict)

    # Convert the processed data back into a GeoDataFrame
    new_df = pd.DataFrame(new_data).set_index('index')  # Use original index
    new_gdf = gpd.GeoDataFrame(new_df, crs=parcel_seg.crs, geometry='split_point')
    
    return new_gdf

# Check if filtered_parcel_seg is non-empty before processing
if not filtered_parcel_seg.empty:
    # Call the function to process the filtered_parcel_seg
    processed_parcel_seg = process_filtered_parcel_seg(filtered_parcel_seg)
else:
    # Handle the case where filtered_parcel_seg is empty
    processed_parcel_seg = gpd.GeoDataFrame(columns=filtered_parcel_seg.columns, crs=parcel_seg.crs)

# %%
# Function to split filtered_parcel_seg using points from processed_parcel_seg
def split_lines_with_points(filtered_parcel_seg, processed_parcel_seg):
    split_segments = []

    for idx, row in filtered_parcel_seg.iterrows():
        line = row['geometry']
        split_point_geom = processed_parcel_seg.loc[idx, 'split_point']  # Get the corresponding point geometry from split_point column
        
        if isinstance(split_point_geom, Point):
            # Check if the split point is on the line
            if line.contains(split_point_geom):
                # If the point is on the line, use it directly for splitting
                split_lines = split(line, split_point_geom)
            else:
                # If the point is not on the line, find the closest point on the line
                projected_distance = line.project(split_point_geom)
                nearest_point = line.interpolate(projected_distance)
                split_lines = split(line, nearest_point)
            
            # Handle GeometryCollection by extracting valid LineString geometries
            if isinstance(split_lines, GeometryCollection):
                split_segments.extend([{
                    **row.to_dict(), 'geometry': geom
                } for geom in split_lines.geoms if isinstance(geom, LineString)])
                continue  # Skip to the next iteration

        # If no valid split point or GeometryCollection, add the original row
        split_segments.append(row.to_dict())
    
    # Convert split_segments to a GeoDataFrame and return
    split_gdf = gpd.GeoDataFrame(split_segments, crs=parcel_seg.crs, geometry='geometry')
    return split_gdf

# Check if both filtered_parcel_seg and processed_parcel_seg are non-empty before processing
if not filtered_parcel_seg.empty and not processed_parcel_seg.empty:
    # Call the function to split lines based on points
    split_parcel_seg = split_lines_with_points(filtered_parcel_seg, processed_parcel_seg)
else:
    # Handle the case where one or both GeoDataFrames are empty
    split_parcel_seg = gpd.GeoDataFrame(columns=filtered_parcel_seg.columns, crs=parcel_seg.crs)

# Function to combine split_parcel_seg and non_filtered_parcel_seg, ensuring parcel_id proximity
def combine_parcel_segs(split_parcel_seg, non_filtered_parcel_seg):
    # Ensure both datasets contain the 'parcel_id' column
    if 'parcel_id' not in split_parcel_seg.columns or 'parcel_id' not in non_filtered_parcel_seg.columns:
        raise ValueError("Both datasets must contain the 'parcel_id' column.")
    
    # Convert parcel_id to string to avoid type errors during sorting
    split_parcel_seg['parcel_id'] = split_parcel_seg['parcel_id'].astype(str)
    non_filtered_parcel_seg['parcel_id'] = non_filtered_parcel_seg['parcel_id'].astype(str)
    
    # Concatenate the two GeoDataFrames and ensure 'crs' and 'geometry' are set
    combined_parcel_seg = gpd.GeoDataFrame(
        pd.concat([split_parcel_seg, non_filtered_parcel_seg], ignore_index=True),
        crs=parcel_seg.crs,  # Use the crs from one of the input GeoDataFrames
        geometry='geometry'  # Ensure the geometry column is correctly set
    )
    
    # Sort by 'parcel_id' to ensure similar parcel_id are together
    combined_parcel_seg_sorted = combined_parcel_seg.sort_values(by='parcel_id')
    
    return combined_parcel_seg_sorted

# Check if both split_parcel_seg and non_filtered_parcel_seg are non-empty before processing
if not split_parcel_seg.empty and not non_filtered_parcel_seg.empty:
    # Call the function to combine the datasets
    reconstr_seg = combine_parcel_segs(split_parcel_seg, non_filtered_parcel_seg)
else:
    # Handle the case where one or both GeoDataFrames are empty
    reconstr_seg = gpd.GeoDataFrame(columns=split_parcel_seg.columns, crs=parcel_seg.crs)

# %%
# Check if reconstr_seg is non-empty before concatenating
if not reconstr_seg.empty:
    parcel_seg = pd.concat([parcel_seg[parcel_seg['edge_num'] != 3], reconstr_seg], ignore_index=True).reset_index(drop=True)

parcel_seg = parcel_seg.drop(columns=['edge_num'])

# %% [markdown]
# #  Step 5: Parcel Classification

# %% [markdown]
# ## Duplicated Address Parcels

# %%
# Identify duplicated parcel_addr values
duplicated_ids = parcel[parcel['parcel_addr'].notna() & parcel['parcel_addr'].duplicated(keep=False)]
# updated those duplicated parcel_addr rows and lable them in the 'parcel_label' column
parcel.loc[parcel['parcel_addr'].isin(duplicated_ids['parcel_addr']), 'parcel_label'] = 'duplicated address'

# %% [markdown]
# ## Jagged Parcels

# %%
# Perimeter-Area Ratio (Shape Index)
parcel['shape_index'] = parcel['geometry'].length / (2 * (3.14159 * parcel['geometry'].area)**0.5)

si_threshold = 0.50
column_name = f"{int(si_threshold * 100)}_threshold"
parcel[column_name] = parcel['shape_index'] > parcel['shape_index'].quantile(si_threshold)

# Ensure the geometry is a Polygon type and calculate the number of edges
edge_count = parcel_seg.groupby('parcel_id').size().reset_index(name='num_edges')
parcel = parcel.merge(edge_count, on='parcel_id', how='left')

parcel['parcel_label'] = parcel.apply(
    lambda row: 'jagged parcel' if pd.isna(row['parcel_label']) and row[column_name] and row['num_edges'] >= 6 else row['parcel_label'], 
    axis=1
)

# %% [markdown]
# ## Regular Inside and Corner Parcels

# %% [markdown]
# ### Part I: Identify Regular Parcels

# %% [markdown]
# #### Step 1: Matching Parcel Edge with Road Geometries

# %%
parcel_seg_filter = parcel_seg[parcel_seg['match_road_address'].notnull()]

# Initialize lists to store the matched road geometries and distances
matched_road_geometries = []
midpoint_distances = []

# Iterate over each row in parcel_seg_filter
for idx, parcel_row in parcel_seg_filter.iterrows():
    match_addr = parcel_row['match_road_address']
    
    # Calculate the midpoint (centroid) of the parcel geometry
    midpoint = parcel_row.geometry.centroid
    # Filter road_seg to get rows where road_addr matches match_road_address
    matching_road_segs = road_seg[road_seg['road_addr'] == match_addr]
    
    if not matching_road_segs.empty:
        # Calculate distances between the midpoint of the parcel and matching road_seg geometries
        distances = matching_road_segs.geometry.apply(lambda geom: midpoint.distance(geom))
        # Find the index of the nearest road geometry
        nearest_index = distances.idxmin()
        # Append the nearest road geometry to the list
        matched_road_geometries.append(matching_road_segs.loc[nearest_index].geometry)
        # Append the corresponding distance (from midpoint to nearest road) to the list
        midpoint_distances.append(distances[nearest_index])
    else:
        # If no match is found, append None for both geometry and distance
        matched_road_geometries.append(None)
        midpoint_distances.append(None)
        
# Add the matched road geometries and midpoint distances to parcel_seg_filter
parcel_seg_filter['road_geometry'] = matched_road_geometries
parcel_seg_filter['midpoint_distance_to_road'] = midpoint_distances

# %% [markdown]
# #### Step 2: Calculating Angle Between Each Parcel Edge and Road Segment

# %%
# Create a new column to store the angle differences between geometry and road_geometry
angle_differences = []

# Iterate over each row and calculate the angle difference between geometry and road_geometry
for idx, row in parcel_seg_filter.iterrows():
    parcel_geom = row['geometry']
    road_geom = row['road_geometry']
    
    # Check if road_geometry is not None
    if road_geom is not None:
        # Calculate the angle difference
        angle_diff = calculate_angle_difference(parcel_geom, road_geom)
        angle_differences.append(angle_diff)
    else:
        # If no road_geometry is found, append None or 0
        angle_differences.append(None)

# Add the angle differences to a new column in parcel_seg_filter
parcel_seg_filter['angle_difference'] = angle_differences

# %% [markdown]
# #### Step 3: Selecting the Most Representative Angle within Each Parcel Group

# %%
# selecting the most representative parcel edge for stored the geometry into parcel geodataframe
parcel_seg_filter = parcel_seg_filter.loc[parcel_seg_filter.groupby('parcel_id')['midpoint_distance_to_road'].idxmin()]
parcel = parcel.merge(parcel_seg_filter[['parcel_id', 'angle_difference']], on='parcel_id', how='left')

# %% [markdown]
# #### Step 4: Creating Tangent Lines for Each Parcel Edge Endpoints

# %%
# Function to create tangent lines at both ends of a line segment
def create_tangents(line):
    coords = list(line.coords)
    if len(coords) < 2:
        return None, None  # Skip invalid geometries
    
    # Create tangents at the start and end of the line segment
    start_tangent = LineString([coords[0], coords[1]])
    end_tangent = LineString([coords[-2], coords[-1]])
    
    return start_tangent, end_tangent

# %% [markdown]
# #### Step 5: Filtering Parcel Based on Tangent Angle Differences

# %%
# Filter parcel segments based on angle difference of tangents > 30 degrees
def filter_parcel_segments(parcel_seg, angle_threshold=30):
    filtered_segments = []
    
    for idx, row in parcel_seg.iterrows():
        line = row['geometry']
        start_tangent, end_tangent = create_tangents(line)
        
        if start_tangent and end_tangent:
            angle_diff = calculate_angle_difference(start_tangent, end_tangent)
            if angle_diff > angle_threshold:
                # Add the segment to the filtered list along with parcel_id and parcel_addr
                filtered_segments.append({
                    'parcel_id': row['parcel_id'],
                    'parcel_addr': row['parcel_addr'],
                    'geometry': line
                })
    
    # Create a new DataFrame with the filtered results
    filtered_df = pd.DataFrame(filtered_segments)
    return filtered_df

# Call the function to filter parcel segments based on angle difference of tangents
filtered_parcel_seg = filter_parcel_segments(parcel_seg)

# %% [markdown]
# #### Step 6: Labeling Regular Parcels Based on Angle and Edge Criteria

# %%
# Check if filtered_parcel_seg is non-empty
if not filtered_parcel_seg.empty:
    # Get unique parcel_id values if filtered_parcel_seg has data
    filtered_object_ids = filtered_parcel_seg['parcel_id'].unique()
    
    # Update 'parcel_label' column with the exclusion condition
    parcel.loc[
        (parcel['num_edges'] == 4) & 
        (parcel['angle_difference'].notnull()) & 
        (parcel['angle_difference'] < 15) & 
        (~parcel['parcel_id'].isin(filtered_object_ids)),  # Exclude matching parcel_id
        'parcel_label'
    ] = 'regular parcel'
else:
    # Directly update 'parcel_label' without exclusion condition if filtered_parcel_seg is empty
    parcel.loc[
        (parcel['num_edges'] == 4) & 
        (parcel['angle_difference'].notnull()) & 
        (parcel['angle_difference'] < 15), 
        'parcel_label'
    ] = 'regular parcel'

# %% [markdown]
# ### Part II: Classify into Inside and Corner Parcels

# %% [markdown]
# #### Step 7: Normalizing Line Segment Directions

# %%
def normalize_linestring(line):
    # Ensure the coordinates are in a consistent direction (smallest point first)
    if isinstance(line, LineString):
        coords = list(line.coords)
        if coords[0] > coords[-1]:
            coords.reverse()  # Reverse the order of coordinates to normalize the direction
        return LineString(coords)
    else:
        return line  # If it's not a LineString, keep it as is

# %% [markdown]
# #### Step 8: Detecting Shared Sides with Normalized Geometries

# %%
def check_shared_sides_normalized(parcel_seg, threshold=0.1, distance_threshold=100):
    """
    Check for shared sides in parcel_seg using cKDTree for faster neighbor searches.
    
    Parameters:
    - parcel_seg: GeoDataFrame containing parcel segments.
    - threshold: float, minimum proportion of line length overlap to consider as a shared side.
    - distance_threshold: float, maximum distance between line segment midpoints to be considered for comparison.
    
    Returns:
    - parcel_seg: GeoDataFrame with 'shared_side' column indicating whether a side is shared.
    """
    
    # Normalize all the geometry objects
    parcel_seg['normalized_geom'] = parcel_seg['geometry'].apply(normalize_linestring)
    # Extract the midpoints of each line segment to build the KDTree
    midpoints = np.array([line.interpolate(0.5, normalized=True).coords[0] for line in parcel_seg['normalized_geom']])
    # Build cKDTree with midpoints
    kdtree = cKDTree(midpoints)
    # Initialize the 'shared_side' column as False
    parcel_seg['shared_side'] = False
    
    # Loop over each line and find nearby lines using KDTree
    for i, line1 in parcel_seg.iterrows():
        # Query the KDTree for neighbors within the distance_threshold
        indices = kdtree.query_ball_point(midpoints[i], r=distance_threshold)
        
        for j in indices:
            if i != j:  # Avoid comparing the line with itself
                line2 = parcel_seg.iloc[j]
                intersection = line1['normalized_geom'].intersection(line2['normalized_geom'])
                if not intersection.is_empty:
                    # Calculate the proportion of overlap relative to the length of line1
                    overlap_ratio = intersection.length / line1['normalized_geom'].length
                    if overlap_ratio > threshold:
                        # If the overlap is greater than the threshold, mark as shared side
                        parcel_seg.at[i, 'shared_side'] = True
                        parcel_seg.at[j, 'shared_side'] = True

    # Remove the temporarily generated 'normalized_geom' column
    parcel_seg = parcel_seg.drop(columns=['normalized_geom'])
    return parcel_seg

parcel_seg = check_shared_sides_normalized(parcel_seg)

# %% [markdown]
# #### Step 9: Calculating the Number of Unique Edges in Each Parcel Group

# %%
parcel_seg = parcel_seg.drop(columns=['parcel_label'])
parcel_seg = parcel_seg.merge(parcel[parcel['parcel_label'] == 'regular parcel'][['parcel_id','parcel_label']], on='parcel_id',how='left')

# count the number of unique side in each regular parcel
false_counts = parcel_seg[parcel_seg['parcel_label'] == 'regular parcel'].groupby('parcel_id')['shared_side'].apply(lambda x: (x == False).sum())

# Create a DataFrame from the Series
false_counts_df = false_counts.reset_index()
false_counts_df.columns = ['parcel_id', 'unique_edge_count'] 

false_counts_df['unique_edges_bigger_2'] = false_counts_df['unique_edge_count'] >= 2
false_counts_df['is_unique'] = false_counts_df['unique_edges_bigger_2'] 

# Merge the counts back to the original GeoDataFrame
parcel_seg = parcel_seg.merge(false_counts_df[['parcel_id','unique_edge_count','unique_edges_bigger_2']], on='parcel_id', how='left')

# Step 1: Extract rows where unique_edge_count == 2
filtered_seg = parcel_seg[parcel_seg['unique_edge_count'] == 2]

# Step 2: Group by parcel_id
grouped = filtered_seg.groupby('parcel_id')

# Function to check if two lines intersect
def check_intersection_for_group(group):
    # Find the two edges where shared_side is False within the group
    shared_sides = group[group['shared_side'] == False]
    
    if len(shared_sides) == 2:
        # Get the geometry (line) for both shared sides
        line1 = shared_sides.iloc[0]['geometry']
        line2 = shared_sides.iloc[1]['geometry']
        
        # Check if the two lines intersect
        intersects = line1.intersects(line2)
        
        return intersects
    else:
        return None  # If there are not exactly two shared sides, skip calculation

# Step 3: Check intersection for each group (parcel_id)
results = []
for OBJECTID, group in grouped:
    intersects = check_intersection_for_group(group)
    if intersects is not None:
        results.append({'parcel_id': OBJECTID, 'lines_intersect': intersects})

# Convert results to a DataFrame
intersection_df = pd.DataFrame(results)

# Convert results to a DataFrame or handle as needed
intersection_df = intersection_df[intersection_df['lines_intersect'] == False]

# Update the false_counts_df based on the intersection results
for i, row in intersection_df.iterrows():
    false_counts_df.loc[false_counts_df['parcel_id'] == row['parcel_id'], 'is_unique'] = row['lines_intersect']

# %% [markdown]
# #### Step 10: Labeling Regular Parcels as Inside or Corner Parcels

# %%
# Step 1: Filter the parcel data
filtered_parcels = parcel.loc[
    (parcel['parcel_label'] == 'regular parcel') & 
    (parcel['num_edges'] == 4)
]

# Step 2: Merge the filtered parcels with false_counts_df based on parcel_id
merged_df = filtered_parcels.merge(false_counts_df[['parcel_id', 'is_unique']], on='parcel_id', how='left')

# # Step 3: Update the parcel_label column based on the value of is_unique
merged_df.loc[merged_df['is_unique'] == True, 'parcel_label'] = 'regular corner parcel'
merged_df.loc[merged_df['is_unique'] == False, 'parcel_label'] = 'regular inside parcel'

# Step 4: Update the original parcel DataFrame
for i, row in merged_df.iterrows():
    parcel.loc[parcel['parcel_id'] == row['parcel_id'], 'parcel_label'] = row['parcel_label']

# %% [markdown]
# ## Label the Cul-de-sac Parcels

# %% [markdown]
# #### Step 1: Identify End Points

# %%
def identify_end_points(road_seg):
    # Get the start and end points of each road segment
    road_seg['start_point'] = road_seg['geometry'].apply(lambda geom: Point(geom.coords[0]))
    road_seg['end_point'] = road_seg['geometry'].apply(lambda geom: Point(geom.coords[-1]))
    
    # Create GeoDataFrames for start and end points, including road_addr and road_id
    start_points = road_seg[['road_addr', 'road_id', 'start_point']].rename(columns={'start_point': 'point'})
    end_points = road_seg[['road_addr', 'road_id', 'end_point']].rename(columns={'end_point': 'point'})
    
    # Concatenate start and end points into a single GeoDataFrame
    all_points = gpd.GeoDataFrame(pd.concat([start_points, end_points]), geometry='point')
    
    # Count how many times each point appears (indicating road connections)
    point_counts = all_points.groupby('point').size().reset_index(name='count')
    
    # Filter points that appear only once (end points connected to a single road)
    end_points = point_counts[point_counts['count'] == 1]
    
    # Merge back road_addr and road_id to end points
    end_points = pd.merge(end_points, all_points[['road_addr', 'road_id', 'point']], on='point', how='left')
    
    # Convert to a GeoDataFrame, using 'point' as the geometry
    end_points = gpd.GeoDataFrame(end_points, geometry='point', crs=road_seg.crs)
    
    return end_points

# Use the function to get endpoints that connect to only one road, including road_addr and road_id
end_points = identify_end_points(road_seg)

# Create a buffer for the end road segments (35m)
end_points_buffer = gpd.GeoDataFrame(end_points.copy(), geometry=end_points.geometry.buffer(35), crs=end_points.crs)
# Drop the 'point' column if no longer needed
end_points_buffer.drop(columns=['point'], inplace=True)

# %% [markdown]
# #### Step 2: Match Potential Cul-de-sac Parcels

# %%
# Check if each parcel in 'parcel' intersects with 'road_seg_end_buffer'
intersections = gpd.sjoin(parcel, end_points_buffer, how="inner", predicate="intersects")
intersections = intersections.set_crs(crs=parcel.crs)
# Filter parcels where the 'match_road_address' matches the 'road_addr' in 'road_seg_end_buffer'
filtered_parcels_seg = intersections[intersections['match_road_address'] == intersections['road_addr']]

# %% [markdown]
# #### Step 3: Label Cul-de-sac Parcels

# %%
def label_end_road_parcels(parcel, filtered_parcels_seg, filtered_object_ids):
    # Create a mask to identify rows in parcel that match parcel_id and parcel_addr
    matching_rows = parcel.merge(
        filtered_parcels_seg[['parcel_id', 'parcel_addr']], 
        on=['parcel_id', 'parcel_addr'], 
        how='inner'
    )
    
    # Update the 'parcel_label' column to 'end_road_parcel' for matched rows
    # Only if num_edges > 4 or parcel_id is in filtered_object_ids
    parcel.loc[
        (parcel['parcel_id'].isin(matching_rows['parcel_id']) & 
         parcel['parcel_addr'].isin(matching_rows['parcel_addr'])) & 
        ((parcel['num_edges'] > 4) | 
         parcel['parcel_id'].isin(filtered_object_ids)), 
        'parcel_label'
    ] = 'cul_de_sac parcel'
    
    return parcel

# Call the function to label the end_road_parcel rows only if both filtered_parcels_seg and filtered_object_ids are non-empty
if not filtered_parcels_seg.empty and len(filtered_object_ids) > 0:
    parcel = label_end_road_parcels(parcel, filtered_parcels_seg, filtered_object_ids)

# %% [markdown]
# ## Label the Curve Parcels

# %%
def label_special_parcels(parcel, filtered_parcel_seg):
    # Create a mask to identify rows in parcel that match parcel_id and parcel_addr
    matching_rows = parcel.merge(
        filtered_parcel_seg[['parcel_id', 'parcel_addr']], 
        on=['parcel_id', 'parcel_addr'], 
        how='inner'
    )
    
    # Update the 'parcel_label' column to 'special parcel' 
    # only for rows where 'parcel_label' is null
    parcel.loc[
        parcel['parcel_label'].isnull() &  # Check for null values
        parcel['parcel_id'].isin(matching_rows['parcel_id']) &
        parcel['parcel_addr'].isin(matching_rows['parcel_addr']), 
        'parcel_label'
    ] = 'curve parcel'
    
    return parcel

# Call the function to label special parcels only if filtered_parcel_seg is non-empty
if not filtered_parcel_seg.empty:
    parcel = label_special_parcels(parcel, filtered_parcel_seg)

# %% [markdown]
# ## label the Special Parcels

# %%
parcel['parcel_label'] = parcel['parcel_label'].fillna('special parcel')

# %%
parcel['parcel_label'].value_counts()

# %% [markdown]
# #  Step 6: Label Parcel Edges

# %% [markdown]
# ## Extract Parcel Edges with Labels

# %%
# Get the centroid or representative points of the road segments
road_centroids = np.array([geom.centroid.coords[0] for geom in road_seg.geometry])
# Build the KDTree based on the centroids of the road segments
road_tree = cKDTree(road_centroids)

# Initialize a list to store the matched road geometries
matched_road_geometries = []

# Iterate over each row in parcel
for idx, parcel_row in parcel.iterrows():
    # Check if Found_Match is True
    if parcel_row['Found_Match'] == True:
        match_addr = parcel_row['match_road_address']
        # Filter road_seg to get rows where road_addr matches match_road_address
        matching_road_segs = road_seg[road_seg['road_addr'] == match_addr]
        
        if not matching_road_segs.empty:
            # Calculate distances between the parcel polygon geometry and matching road_seg geometries
            distances = matching_road_segs.geometry.apply(lambda geom: parcel_row.geometry.distance(geom))
            
            # Find the index of the nearest road geometry
            nearest_index = distances.idxmin()
            
            # Append the nearest road geometry to the list
            matched_road_geometries.append(matching_road_segs.loc[nearest_index].geometry)
        else:
            # If no match is found, append None or an empty geometry
            matched_road_geometries.append(None)
    else:
        # If Found_Match is False or NaN, find the nearest road geometry
        # Get the centroid of the current parcel polygon
        parcel_centroid = np.array(parcel_row.geometry.centroid.coords[0])
        
        # Query the KDTree for the nearest road segment
        _, nearest_index = road_tree.query(parcel_centroid)
        
        # Append the nearest road geometry to the list
        matched_road_geometries.append(road_seg.iloc[nearest_index].geometry)
        
# Add the matched road geometries to parcel
parcel['road_geometry'] = matched_road_geometries

# %%
# Function to explode Polygons into individual boundary line segments
def explode_to_lines(gdf):
    # Create a list to store new rows
    line_list = []

    for index, row in gdf.iterrows():
        # Get the exterior boundary of the polygon
        exterior = row['geometry'].exterior
        # Convert the boundary into LineString segments
        lines = [LineString([exterior.coords[i], exterior.coords[i + 1]]) 
                 for i in range(len(exterior.coords) - 1)]
        
        # Create new rows for each line segment, retaining the original attributes
        for line in lines:
            new_row = row.copy()
            new_row['geometry'] = line
            line_list.append(new_row)
    
    # Use pd.concat to generate the final GeoDataFrame
    line_gdf = pd.concat(line_list, axis=1).T
    line_gdf = gpd.GeoDataFrame(line_gdf, geometry='geometry', crs=gdf.crs)
    
    return line_gdf

# Call the function to explode the line segments
parcel_seg = explode_to_lines(parcel)

# Reset the index by group
parcel_seg['new_index'] = parcel_seg.groupby('parcel_id').cumcount()
parcel_seg.set_index('new_index', inplace=True)
parcel_seg.index.name = None


# Function to calculate the bearing of a geometry
def fun_bearing_ra(geom):
    coords = np.array(geom.coords)
    # Use the first and last coordinates to calculate the bearing
    x1, y1 = coords[0]
    x2, y2 = coords[-1]
    
    # Calculate the bearing using atan2
    bearing = math.atan2(y2 - y1, x2 - x1)
    
    return bearing

def calculate_angle_difference(line1, line2):
    bearing1 = fun_bearing_ra(line1)
    bearing2 = fun_bearing_ra(line2)
    # Calculate the absolute angle difference and ensure it is <= 180 degrees
    delta_theta = bearing2 - bearing1
    
    # Ensure the angle is between -π and π
    if delta_theta > math.pi:
        delta_theta -= 2 * math.pi
    elif delta_theta < -math.pi:
        delta_theta += 2 * math.pi
    
    # Convert the angle to degrees
    angle_between_degrees = math.degrees(abs(delta_theta))
    
    # Return the smaller angle difference (angle or its supplement)
    return min(angle_between_degrees, 180 - angle_between_degrees)


# Check if two segments share a common point (i.e., their start or end point is the same)
def are_segments_connected(line1, line2):
    coords1 = np.array(line1.coords)
    coords2 = np.array(line2.coords)
    
    # Check if the start or end points of the segments are the same
    if np.all(coords1[0] == coords2[0]) or np.all(coords1[0] == coords2[-1]) or \
       np.all(coords1[-1] == coords2[0]) or np.all(coords1[-1] == coords2[-1]):
        return True
    return False

# Function to reorder segments based on the turning point
def reorder_segments_by_turning_point(segments, turning_point_index):
    # Reorder segments starting from the identified turning point
    reordered_segments = segments[turning_point_index:] + segments[:turning_point_index]
    return reordered_segments

# Main function: Process each parcel_id group and return a new GeoDataFrame
def process_parcel_segments(parcel_seg):
    merged_segments = []  # List to store the reordered segments

    # Group the parcel segments by parcel_id and process each group
    for object_id, group in parcel_seg.groupby('parcel_id'):
        segments = group['geometry'].tolist()  # Get the list of line segments for the current group
        original_indices = group.index.tolist()  # Preserve the original indices
        turning_points = []

        # Loop through all adjacent segments to calculate angle differences
        for i in range(1, len(segments)):
            if are_segments_connected(segments[i-1], segments[i]):
                angle_diff = calculate_angle_difference(segments[i-1], segments[i])
                if angle_diff > 15:  # If angle difference is greater than 15 degrees, mark it as a turning point
                    turning_points.append(i)

        # If there are turning points, reorder the segments starting from the first turning point
        if turning_points:
            turning_point_index = turning_points[0]
            reordered_segments = reorder_segments_by_turning_point(segments, turning_point_index)
            reordered_original_indices = reorder_segments_by_turning_point(original_indices, turning_point_index)
        else:
            # If no turning points, retain the original order
            reordered_segments = segments
            reordered_original_indices = original_indices

        # Store the reordered segments and their attributes
        for j, (line, original_index) in enumerate(zip(reordered_segments, reordered_original_indices)):
            row = group.iloc[0].copy()  # Copy the first row's attributes
            row['geometry'] = line
            row['original_index'] = original_index  # Preserve the original index
            row['new_index'] = j  # Assign the new index based on the reordered list
            merged_segments.append(row)

    # Create a new GeoDataFrame for the reordered segments
    updated_gdf = gpd.GeoDataFrame(merged_segments, columns=parcel_seg.columns.tolist() + ['original_index', 'new_index'])
    updated_gdf = updated_gdf.reset_index(drop=True)

    return updated_gdf

# Run the main function and get the new GeoDataFrame
updated_parcel_seg = process_parcel_segments(parcel_seg)
parcel_seg = updated_parcel_seg


# Group parcel_seg by parcel_id and process each group
merged_segments = []

for object_id, group in parcel_seg.groupby('parcel_id'):
    # Get the list of geometries in the current group
    segments = group.geometry.tolist()
    # Start with the first segment
    merged_lines = [segments[0]]  # Start with the first segment
    
    for i in range(1, len(segments)):
        connected = False
        
        # Always compare the current segment with the previous one
        if are_segments_connected(segments[i-1], segments[i]):
            # Calculate the angle difference between the current segment and the previous one
            angle_diff = calculate_angle_difference(segments[i-1], segments[i])
            
            # If the angle difference is less than 15 degrees, merge the adjacent line segments
            if angle_diff < 15:
                # Merge the current and previous segments
                merged_result = linemerge([merged_lines[-1], segments[i]])
                
                # Check if the result is a MultiLineString, if so, skip the merge
                if isinstance(merged_result, LineString):
                    merged_lines[-1] = merged_result
                    connected = True
                else:
                    # Skip the merge if it's a MultiLineString
                    continue
        
        # If no connected segment is found or the angle difference is too large, add the current segment as a new one
        if not connected:
            merged_lines.append(segments[i])
    
    # Keep the merged results and add other attributes
    for line in merged_lines:
        row = group.iloc[0].copy()  # Copy the first attribute row from the group
        row['geometry'] = line
        merged_segments.append(row)

# Create a new GeoDataFrame from the merged line segments
parcel_seg = gpd.GeoDataFrame(merged_segments, columns=parcel_seg.columns)

# Check for MultiLineString geometries and explode them into LineString
exploded_segments = []

for index, row in parcel_seg.iterrows():
    geom = row['geometry']
    
    if isinstance(geom, MultiLineString):
        # Explode the MultiLineString into individual LineStrings
        for line in geom:
            new_row = row.copy()
            new_row['geometry'] = line
            exploded_segments.append(new_row)
    else:
        # Keep the original LineString geometries
        exploded_segments.append(row)

# Create a new GeoDataFrame from the exploded segments
parcel_seg = gpd.GeoDataFrame(exploded_segments, columns=parcel_seg.columns)

# extract useful columns
parcel_seg.drop(columns=['original_index', 'new_index'], inplace=True)
# Reset the index of the final GeoDataFrame
parcel_seg = parcel_seg.reset_index(drop=True)



edge_counts = parcel_seg.groupby('parcel_id').size()
parcel_seg['edge_num'] = parcel_seg['parcel_id'].map(edge_counts)

# Function to create tangent lines at both ends of a line segment
def create_tangents(line):
    coords = list(line.coords)
    if len(coords) < 2:
        return None, None  # Skip invalid geometries
    
    # Create tangents at the start and end of the line segment
    start_tangent = LineString([coords[0], coords[1]])
    end_tangent = LineString([coords[-2], coords[-1]])
    
    return start_tangent, end_tangent

# Function to filter curve segments based on angle difference of tangents > 30 degrees
def filter_curve_segments(parcel_seg, angle_threshold=30):
    filtered_segments = []
    non_filtered_segments = []
    
    for idx, row in parcel_seg.iterrows():
        line = row['geometry']
        start_tangent, end_tangent = create_tangents(line)
        
        if start_tangent and end_tangent:
            angle_diff = calculate_angle_difference(start_tangent, end_tangent)
            row_dict = row.to_dict()  # Convert the entire row to a dictionary
            row_dict['index'] = idx  # Preserve the original index
            
            if angle_diff > angle_threshold:
                # Add the entire row to the filtered list
                filtered_segments.append(row_dict)
            else:
                # Add the entire row to the non-filtered list
                non_filtered_segments.append(row_dict)
    
    # Create DataFrames with the filtered and non-filtered results if data exists
    if filtered_segments:
        filtered_df = pd.DataFrame(filtered_segments).set_index('index')
        filtered_gdf = gpd.GeoDataFrame(filtered_df, crs=parcel_seg.crs, geometry=filtered_df['geometry'])
    else:
        # Initialize an empty GeoDataFrame with the same structure if no data
        filtered_gdf = gpd.GeoDataFrame(columns=parcel_seg.columns, crs=parcel_seg.crs)
    
    if non_filtered_segments:
        non_filtered_df = pd.DataFrame(non_filtered_segments).set_index('index')
        non_filtered_gdf = gpd.GeoDataFrame(non_filtered_df, crs=parcel_seg.crs, geometry=non_filtered_df['geometry'])
    else:
        # Initialize an empty GeoDataFrame with the same structure if no data
        non_filtered_gdf = gpd.GeoDataFrame(columns=parcel_seg.columns, crs=parcel_seg.crs)
    
    return filtered_gdf, non_filtered_gdf

# Call the function to filter curve segments and create two GeoDataFrames
filtered_parcel_seg, non_filtered_parcel_seg = filter_curve_segments(parcel_seg[parcel_seg['edge_num'] == 3])

# Function to create tangent lines and reverse the line if necessary
def create_tangents_with_reversal(line):
    coords = list(line.coords)
    if len(coords) < 2:
        return None, None  # Skip invalid geometries
    
    # Find the points with the smallest and largest y-coordinate (latitude)
    if coords[0][1] < coords[-1][1]:  # If the first point's y is smaller, it's the start point
        start_point = coords[0]
        end_point = coords[-1]
    else:  # Otherwise, the last point is the start point
        start_point = coords[-1]
        end_point = coords[0]

    # Reverse the line if start_point is not the same as coords[0]
    if start_point != coords[0]:
        coords.reverse()  # Reverse the order of coordinates
    
    # Now create tangents based on the (possibly reversed) coordinates
    start_tangent = LineString([coords[0], coords[1]])  # Tangent from the first to the second point
    end_tangent = LineString([coords[-2], coords[-1]])  # Tangent from the second last to the last point

    return start_tangent, end_tangent, LineString(coords)  # Return the tangents and the (possibly reversed) line

# Function to calculate the split point based on the 4/5 rule
def calculate_split_point(line, start_tangent, end_tangent, angle_diff, angle_fraction=0.5):
    coords = list(line.coords)

    # Iterate through the line and find the point where the angle difference is approximately 4/5
    for i in range(1, len(coords) - 1):
        intermediate_tangent = LineString([coords[i - 1], coords[i]])
        current_angle_diff = calculate_angle_difference(start_tangent, intermediate_tangent)
        
        if current_angle_diff >= angle_diff * angle_fraction:
            return coords[i]  # Return the split point

    return coords[-1]  # If no point found, return the endpoint

# Function to process each segment in filtered_parcel_seg
def process_filtered_parcel_seg(filtered_parcel_seg, angle_threshold=30, angle_fraction=0.5):
    new_data = []
    
    for idx, row in filtered_parcel_seg.iterrows():
        line = row['geometry']
        
        # Apply the tangent and reversal function
        start_tangent, end_tangent, adjusted_line = create_tangents_with_reversal(line)
        
        if start_tangent and end_tangent:
            angle_diff = calculate_angle_difference(start_tangent, end_tangent)
            
            if angle_diff > angle_threshold:
                # Calculate the split point based on the angle difference and fraction
                split_point = calculate_split_point(adjusted_line, start_tangent, end_tangent, angle_diff, angle_fraction)
                
                # Add split point to row's data
                row_dict = row.to_dict()
                row_dict['split_point'] = Point(split_point)  # Store the split point as geometry
                row_dict['index'] = idx  # Store the original index
                
                new_data.append(row_dict)
            else:
                # If no split needed, just keep the original row
                row_dict = row.to_dict()
                row_dict['split_point'] = None  # No split point, store None
                row_dict['index'] = idx  # Store the original index
                
                new_data.append(row_dict)

    # Convert the processed data back into a GeoDataFrame
    new_df = pd.DataFrame(new_data).set_index('index')  # Use original index
    new_gdf = gpd.GeoDataFrame(new_df, crs=parcel_seg.crs, geometry='split_point')
    
    return new_gdf

# Check if filtered_parcel_seg is non-empty before processing
if not filtered_parcel_seg.empty:
    # Call the function to process the filtered_parcel_seg
    processed_parcel_seg = process_filtered_parcel_seg(filtered_parcel_seg)
else:
    # Handle the case where filtered_parcel_seg is empty
    processed_parcel_seg = gpd.GeoDataFrame(columns=filtered_parcel_seg.columns, crs=parcel_seg.crs)

# Function to split filtered_parcel_seg using points from processed_parcel_seg
def split_lines_with_points(filtered_parcel_seg, processed_parcel_seg):
    split_segments = []

    for idx, row in filtered_parcel_seg.iterrows():
        line = row['geometry']
        split_point_geom = processed_parcel_seg.loc[idx, 'split_point']  # Get the corresponding point geometry from split_point column
        
        if isinstance(split_point_geom, Point):
            # Check if the split point is on the line
            if line.contains(split_point_geom):
                # If the point is on the line, use it directly for splitting
                split_lines = split(line, split_point_geom)
            else:
                # If the point is not on the line, find the closest point on the line
                projected_distance = line.project(split_point_geom)
                nearest_point = line.interpolate(projected_distance)
                split_lines = split(line, nearest_point)
            
            # Handle GeometryCollection by extracting valid LineString geometries
            if isinstance(split_lines, GeometryCollection):
                split_segments.extend([{
                    **row.to_dict(), 'geometry': geom
                } for geom in split_lines.geoms if isinstance(geom, LineString)])
                continue  # Skip to the next iteration

        # If no valid split point or GeometryCollection, add the original row
        split_segments.append(row.to_dict())
    
    # Convert split_segments to a GeoDataFrame and return
    split_gdf = gpd.GeoDataFrame(split_segments, crs=parcel_seg.crs, geometry='geometry')
    return split_gdf

# Check if both filtered_parcel_seg and processed_parcel_seg are non-empty before processing
if not filtered_parcel_seg.empty and not processed_parcel_seg.empty:
    # Call the function to split lines based on points
    split_parcel_seg = split_lines_with_points(filtered_parcel_seg, processed_parcel_seg)
else:
    # Handle the case where one or both GeoDataFrames are empty
    split_parcel_seg = gpd.GeoDataFrame(columns=filtered_parcel_seg.columns, crs=parcel_seg.crs)

# Function to combine split_parcel_seg and non_filtered_parcel_seg, ensuring parcel_id proximity
def combine_parcel_segs(split_parcel_seg, non_filtered_parcel_seg):
    # Ensure both datasets contain the 'parcel_id' column
    if 'parcel_id' not in split_parcel_seg.columns or 'parcel_id' not in non_filtered_parcel_seg.columns:
        raise ValueError("Both datasets must contain the 'parcel_id' column.")
    
    # Convert parcel_id to string to avoid type errors during sorting
    split_parcel_seg['parcel_id'] = split_parcel_seg['parcel_id'].astype(str)
    non_filtered_parcel_seg['parcel_id'] = non_filtered_parcel_seg['parcel_id'].astype(str)
    
    # Concatenate the two GeoDataFrames and ensure 'crs' and 'geometry' are set
    combined_parcel_seg = gpd.GeoDataFrame(
        pd.concat([split_parcel_seg, non_filtered_parcel_seg], ignore_index=True),
        crs=parcel_seg.crs,  # Use the crs from one of the input GeoDataFrames
        geometry='geometry'  # Ensure the geometry column is correctly set
    )
    
    # Sort by 'parcel_id' to ensure similar parcel_id are together
    combined_parcel_seg_sorted = combined_parcel_seg.sort_values(by='parcel_id')
    
    return combined_parcel_seg_sorted

# Check if both split_parcel_seg and non_filtered_parcel_seg are non-empty before processing
if not split_parcel_seg.empty and not non_filtered_parcel_seg.empty:
    # Call the function to combine the datasets
    reconstr_seg = combine_parcel_segs(split_parcel_seg, non_filtered_parcel_seg)
else:
    # Handle the case where one or both GeoDataFrames are empty
    reconstr_seg = gpd.GeoDataFrame(columns=split_parcel_seg.columns, crs=parcel_seg.crs)


# Check if reconstr_seg is non-empty before concatenating
if not reconstr_seg.empty:
    parcel_seg = pd.concat([parcel_seg[parcel_seg['edge_num'] != 3], reconstr_seg], ignore_index=True).reset_index(drop=True)

parcel_seg = parcel_seg.drop(columns=['edge_num'])
parcel_seg = parcel_seg.set_crs(parcel.crs, allow_override=True)

# %%
def normalize_linestring(line):
    # Ensure the coordinates are in a consistent direction (smallest point first)
    if isinstance(line, LineString):
        coords = list(line.coords)
        if coords[0] > coords[-1]:
            coords.reverse()  # Reverse the order of coordinates to normalize the direction
        return LineString(coords)
    else:
        return line  # If it's not a LineString, keep it as is
    
def check_shared_sides_normalized(parcel_seg, threshold=0.1, distance_threshold=100):
    """
    Check for shared sides in parcel_seg using cKDTree for faster neighbor searches.
    
    Parameters:
    - parcel_seg: GeoDataFrame containing parcel segments.
    - threshold: float, minimum proportion of line length overlap to consider as a shared side.
    - distance_threshold: float, maximum distance between line segment midpoints to be considered for comparison.
    
    Returns:
    - parcel_seg: GeoDataFrame with 'shared_side' column indicating whether a side is shared.
    """
    
    # Normalize all the geometry objects
    parcel_seg['normalized_geom'] = parcel_seg['geometry'].apply(normalize_linestring)
    # Extract the midpoints of each line segment to build the KDTree
    midpoints = np.array([line.interpolate(0.5, normalized=True).coords[0] for line in parcel_seg['normalized_geom']])
    # Build cKDTree with midpoints
    kdtree = cKDTree(midpoints)
    # Initialize the 'shared_side' column as False
    parcel_seg['shared_side'] = False
    
    # Loop over each line and find nearby lines using KDTree
    for i, line1 in parcel_seg.iterrows():
        # Query the KDTree for neighbors within the distance_threshold
        indices = kdtree.query_ball_point(midpoints[i], r=distance_threshold)
        
        for j in indices:
            if i != j:  # Avoid comparing the line with itself
                line2 = parcel_seg.iloc[j]
                intersection = line1['normalized_geom'].intersection(line2['normalized_geom'])
                if not intersection.is_empty:
                    # Calculate the proportion of overlap relative to the length of line1
                    overlap_ratio = intersection.length / line1['normalized_geom'].length
                    if overlap_ratio > threshold:
                        # If the overlap is greater than the threshold, mark as shared side
                        parcel_seg.at[i, 'shared_side'] = True
                        parcel_seg.at[j, 'shared_side'] = True

    # Remove the temporarily generated 'normalized_geom' column
    parcel_seg = parcel_seg.drop(columns=['normalized_geom'])
    return parcel_seg

parcel_seg = check_shared_sides_normalized(parcel_seg)

# %% [markdown]
# ## Regular Inside Parcels Edges

# %%
# calculate the angle between the each parcel seg and nearest road seg
regular_insid_parcel = parcel_seg[parcel_seg['parcel_label'] == 'regular inside parcel']
regular_insid_parcel['parcel_bearing'] = regular_insid_parcel['geometry'].apply(fun_bearing_ra)
regular_insid_parcel['road_bearing'] = regular_insid_parcel['road_geometry'].apply(fun_bearing_ra)
regular_insid_parcel['angle'] = regular_insid_parcel.apply(
    lambda row: calculate_angle_difference(row['geometry'], row['road_geometry']), axis=1
)

# calculate the distance between the each parcel seg and nearest road seg
# regular_insid_parcel['distance_to_road'] = regular_insid_parcel.apply(lambda row: row['geometry'].distance(row['road_geometry']), axis=1)
regular_insid_parcel['distance_to_road'] = regular_insid_parcel.apply(lambda row: row['geometry'].centroid.distance(row['road_geometry']), axis=1)

# %%
# Group by 'parcel_id' and perform the operations within each group
def classify_sides(group):
    # Create a new column 'side'
    group['side'] = None
    
    # Step 1: Find the two rows with the smallest 'angle' values
    smallest_two_angles = group.nsmallest(2, 'angle')
    
    if not smallest_two_angles.empty:
        # Compare 'distance_to_road' between the two rows
        idx_min_distance = smallest_two_angles['distance_to_road'].idxmin()
        idx_max_distance = smallest_two_angles['distance_to_road'].idxmax()
        group.loc[idx_min_distance, 'side'] = 'front'
        group.loc[idx_max_distance, 'side'] = 'rear'
    
    # Step 2: For remaining rows, find shared_side=True and mark as 'Interior side'
    shared_side_true = group[(group['side'].isnull()) & (group['shared_side'] == True)]
    group.loc[shared_side_true.index, 'side'] = 'Interior side'
    
    # Step 3: Label the remaining rows as 'Exterior side'
    group.loc[group['side'].isnull(), 'side'] = 'Exterior side'
    
    return group

# Apply the function to each group
regular_insid_parcel = regular_insid_parcel.groupby('parcel_id').apply(classify_sides)
regular_insid_parcel = regular_insid_parcel.reset_index(level=0, drop=True)

# %% [markdown]
# ## Regular Corner Parcels Edges

# %%
# calculate the angle between the each parcel seg and nearest road seg
regular_corner_parcel = parcel_seg[parcel_seg['parcel_label'] == 'regular corner parcel']
regular_corner_parcel['parcel_bearing'] = regular_corner_parcel['geometry'].apply(fun_bearing_ra)
regular_corner_parcel['road_bearing'] = regular_corner_parcel['road_geometry'].apply(fun_bearing_ra)
regular_corner_parcel['angle'] = regular_corner_parcel.apply(
    lambda row: calculate_angle_difference(row['geometry'], row['road_geometry']), axis=1
)

# calculate the distance between the each parcel seg and nearest road seg
# regular_corner_parcel['distance_to_road'] = regular_corner_parcel.apply(lambda row: row['geometry'].distance(row['road_geometry']), axis=1)
regular_corner_parcel['distance_to_road'] = regular_corner_parcel.apply(lambda row: row['geometry'].centroid.distance(row['road_geometry']), axis=1)

# Apply the function to each group
regular_corner_parcel = regular_corner_parcel.groupby('parcel_id').apply(classify_sides)
regular_corner_parcel = regular_corner_parcel.reset_index(level=0, drop=True)

# %% [markdown]
# ## Special Parcels Edges

# %%
# calculate the angle between the each parcel seg and nearest road seg
special_parcel = parcel_seg[parcel_seg['parcel_label'] == 'special parcel']
special_parcel['parcel_bearing'] = special_parcel['geometry'].apply(fun_bearing_ra)
special_parcel['road_bearing'] = special_parcel['road_geometry'].apply(fun_bearing_ra)
special_parcel['angle'] = special_parcel.apply(
    lambda row: calculate_angle_difference(row['geometry'], row['road_geometry']), axis=1
)

# calculate the distance between the each parcel seg and nearest road seg
special_parcel['distance_to_road'] = special_parcel.apply(lambda row: row['geometry'].centroid.distance(row['road_geometry']), axis=1)

# %%
# Step 2: Handle rows where num_edges = 3
def classify_num_edges_3(group):
    if len(group) == 3:
        idx_min_distance = group['distance_to_road'].idxmin()
        idx_max_distance = group['distance_to_road'].idxmax()
        group.loc[idx_min_distance, 'side'] = 'front'
        group.loc[idx_max_distance, 'side'] = 'rear'
        
        # For the remaining row(s), classify based on shared_side
        remaining_rows = group['side'].isnull()
        group.loc[remaining_rows & (group['shared_side'] == True), 'side'] = 'Interior side'
        group.loc[remaining_rows & (group['shared_side'] == False), 'side'] = 'Exterior side'
    
    return group

# Step 3: Handle rows where num_edges = 4 (reuse your existing function)
def classify_num_edges_4(group):
    # Step 0: Initialize the 'side' column if it doesn't exist
    if 'side' not in group.columns:
        group['side'] = None  # You can also initialize it with np.nan if preferred
    
    # Step 1: Find the two rows with the smallest 'angle' values
    smallest_two_angles = group.nsmallest(2, 'angle')
    
    if not smallest_two_angles.empty:
        # Compare 'distance_to_road' between the two rows
        idx_min_distance = smallest_two_angles['distance_to_road'].idxmin()
        idx_max_distance = smallest_two_angles['distance_to_road'].idxmax()
        group.loc[idx_min_distance, 'side'] = 'front'
        group.loc[idx_max_distance, 'side'] = 'rear'
    
    # Step 2: For remaining rows, find shared_side=True and mark as 'Interior side'
    shared_side_true = group[(group['side'].isnull()) & (group['shared_side'] == True)]
    group.loc[shared_side_true.index, 'side'] = 'Interior side'
    
    # Step 3: Label the remaining rows as 'Exterior side'
    group.loc[group['side'].isnull(), 'side'] = 'Exterior side'
    
    return group

def classify_other_edges(group):
    # Step 0: Initialize the 'side' column if it doesn't exist
    if 'side' not in group.columns:
        group['side'] = None  # You can also initialize it with np.nan if preferred

    if group['num_edges'].iloc[0] not in [3, 4]:
        # Step 1: Filter rows where angle < 20
        valid_rows = group[group['angle'] < 20]

        if not valid_rows.empty:
            # Mark the smallest and largest distances as front and rear within the filtered rows
            idx_min_distance = valid_rows['distance_to_road'].idxmin()
            idx_max_distance = valid_rows['distance_to_road'].idxmax()
            group.loc[idx_min_distance, 'side'] = 'front'
            group.loc[idx_max_distance, 'side'] = 'rear'
        
        # Step 2: Check for angle < 20 and adjacency to front or rear
        for idx, row in group.iterrows():
            if pd.isnull(row['side']) and row['angle'] < 20:
                if group.loc[idx_min_distance, 'geometry'].touches(row['geometry']):
                    group.loc[idx, 'side'] = 'front'
                elif group.loc[idx_max_distance, 'geometry'].touches(row['geometry']):
                    group.loc[idx, 'side'] = 'rear'

        # Step 3: For remaining rows with angle < 20, calculate distance to the nearest front and rear
        front_geom = group.loc[group['side'] == 'front', 'geometry']
        rear_geom = group.loc[group['side'] == 'rear', 'geometry']
        for idx, row in group.iterrows():
            if pd.isnull(row['side']) and row['angle'] < 20:
                # Calculate distance to nearest front and rear
                distance_to_front = row['geometry'].distance(front_geom.iloc[0]) if not front_geom.empty else float('inf')
                distance_to_rear = row['geometry'].distance(rear_geom.iloc[0]) if not rear_geom.empty else float('inf')
                # Label based on the closer distance
                if distance_to_front < distance_to_rear:
                    group.loc[idx, 'side'] = 'front'
                else:
                    group.loc[idx, 'side'] = 'rear'
                
        # Step 4: For edges between two 'front' or two 'rear' edges, and within bounding box
        for idx, row in group.iterrows():
            if pd.isnull(row['side']):
                front_edges = group[group['side'] == 'front']
                rear_edges = group[group['side'] == 'rear']

                # Check if the current edge touches at least two front edges or two rear edges
                front_touch_count = sum(row['geometry'].touches(front_row['geometry']) for front_idx, front_row in front_edges.iterrows())
                rear_touch_count = sum(row['geometry'].touches(rear_row['geometry']) for rear_idx, rear_row in rear_edges.iterrows())

                # Create bounding box for front_edges and rear_edges
                if not front_edges.empty:
                    min_x_front = front_edges.bounds.minx.min()
                    max_x_front = front_edges.bounds.maxx.max()
                    min_y_front = front_edges.bounds.miny.min()
                    max_y_front = front_edges.bounds.maxy.max()
                    front_boundary_box = gpd.GeoSeries([box(min_x_front, min_y_front, max_x_front, max_y_front)])

                if not rear_edges.empty:
                    min_x_rear = rear_edges.bounds.minx.min()
                    max_x_rear = rear_edges.bounds.maxx.max()
                    min_y_rear = rear_edges.bounds.miny.min()
                    max_y_rear = rear_edges.bounds.maxy.max()
                    rear_boundary_box = gpd.GeoSeries([box(min_x_rear, min_y_rear, max_x_rear, max_y_rear)])

                # Check if the current edge is within the front or rear bounding box
                within_front_boundary = row['geometry'].within(front_boundary_box.unary_union) if not front_edges.empty else False
                within_rear_boundary = row['geometry'].within(rear_boundary_box.unary_union) if not rear_edges.empty else False

                # Final condition for labeling
                if front_touch_count >= 2 or within_front_boundary:
                    group.loc[idx, 'side'] = 'front'
                elif rear_touch_count >= 2 or within_rear_boundary:
                    group.loc[idx, 'side'] = 'rear'
        
        # Step 5: Fill remaining NaN sides based on shared_side
        group.loc[group['side'].isnull() & (group['shared_side'] == True), 'side'] = 'Interior side'
        group.loc[group['side'].isnull() & (group['shared_side'] == False), 'side'] = 'Exterior side'
    
    return group

# Combine everything into a single function
def process_special_parcel(special_parcel):
    # Group by parcel_id and classify by num_edges
    def classify_group(group):
        if group['num_edges'].iloc[0] == 3:
            return classify_num_edges_3(group)
        elif group['num_edges'].iloc[0] == 4:
            return classify_num_edges_4(group)
        else:
            return classify_other_edges(group)
    
    # Apply classification by group
    special_parcel = special_parcel.groupby('parcel_id').apply(classify_group)
    special_parcel = special_parcel.reset_index(drop=True)
    
    return special_parcel

# Apply the function to process special_parcel
special_parcel = process_special_parcel(special_parcel)

# %% [markdown]
# ## Jagged Parcels Edges

# %%
# calculate the angle between the each parcel seg and nearest road seg
jagged_parcel = parcel_seg[parcel_seg['parcel_label'] == 'jagged parcel']
jagged_parcel['parcel_bearing'] = jagged_parcel['geometry'].apply(fun_bearing_ra)
jagged_parcel['road_bearing'] = jagged_parcel['road_geometry'].apply(fun_bearing_ra)
jagged_parcel['angle'] = jagged_parcel.apply(
    lambda row: calculate_angle_difference(row['geometry'], row['road_geometry']), axis=1
)

# calculate the distance between the each parcel seg and nearest road seg
jagged_parcel['distance_to_road'] = jagged_parcel.apply(lambda row: row['geometry'].centroid.distance(row['road_geometry']), axis=1)

# %%
def classify_jagged_edges(group):
    # Step 0: Initialize the 'side' column if it doesn't exist
    if 'side' not in group.columns:
        group['side'] = None  # You can also initialize it with np.nan if preferred

    # Step 1: Filter rows where angle < 20 for smallest and largest distances
    valid_rows = group[group['angle'] < 20]
    
    if not valid_rows.empty:
        # Mark the smallest and largest distances as front and rear within the filtered rows
        idx_min_distance = valid_rows['distance_to_road'].idxmin()
        idx_max_distance = valid_rows['distance_to_road'].idxmax()
        group.loc[idx_min_distance, 'side'] = 'front'
        group.loc[idx_max_distance, 'side'] = 'rear'
    
    # Step 2: Check for angle < 20 and adjacency to front or rear
    for idx, row in group.iterrows():
        if pd.isnull(row['side']) and row['angle'] < 20:
            if group.loc[idx_min_distance, 'geometry'].touches(row['geometry']):
                group.loc[idx, 'side'] = 'front'
            elif group.loc[idx_max_distance, 'geometry'].touches(row['geometry']):
                group.loc[idx, 'side'] = 'rear'

    # Step 3: For remaining rows with angle < 20, calculate distance to the nearest front and rear
    front_geom = group.loc[group['side'] == 'front', 'geometry']
    rear_geom = group.loc[group['side'] == 'rear', 'geometry']
    for idx, row in group.iterrows():
        if pd.isnull(row['side']) and row['angle'] < 20:
            # Calculate distance to nearest front and rear
            distance_to_front = row['geometry'].distance(front_geom.iloc[0]) if not front_geom.empty else float('inf')
            distance_to_rear = row['geometry'].distance(rear_geom.iloc[0]) if not rear_geom.empty else float('inf')
            # Label based on the closer distance
            if distance_to_front < distance_to_rear:
                group.loc[idx, 'side'] = 'front'
            else:
                group.loc[idx, 'side'] = 'rear'

    # Step 4: For edges between two 'front' or two 'rear' edges, and within bounding box
    for idx, row in group.iterrows():
        if pd.isnull(row['side']):
            front_edges = group[group['side'] == 'front']
            rear_edges = group[group['side'] == 'rear']

            # Only proceed if there are at least 2 front or 2 rear edges
            if len(front_edges) >= 2 or len(rear_edges) >= 2:
                
                # Check if the current edge touches at least two front edges or two rear edges
                front_touch_count = sum(row['geometry'].touches(front_row['geometry']) for front_idx, front_row in front_edges.iterrows())
                rear_touch_count = sum(row['geometry'].touches(rear_row['geometry']) for rear_idx, rear_row in rear_edges.iterrows())

                # Create bounding box for front_edges if there are at least 2 front edges
                if len(front_edges) >= 2:
                    min_x_front = front_edges.bounds.minx.min()
                    max_x_front = front_edges.bounds.maxx.max()
                    min_y_front = front_edges.bounds.miny.min()
                    max_y_front = front_edges.bounds.maxy.max()
                    front_boundary_box = gpd.GeoSeries([box(min_x_front, min_y_front, max_x_front, max_y_front)])

                # Create bounding box for rear_edges if there are at least 2 rear edges
                if len(rear_edges) >= 2:
                    min_x_rear = rear_edges.bounds.minx.min()
                    max_x_rear = rear_edges.bounds.maxx.max()
                    min_y_rear = rear_edges.bounds.miny.min()
                    max_y_rear = rear_edges.bounds.maxy.max()
                    rear_boundary_box = gpd.GeoSeries([box(min_x_rear, min_y_rear, max_x_rear, max_y_rear)])

                # Check if the current edge is within the front or rear bounding box
                within_front_boundary = row['geometry'].within(front_boundary_box.unary_union) if len(front_edges) >= 2 else False
                within_rear_boundary = row['geometry'].within(rear_boundary_box.unary_union) if len(rear_edges) >= 2 else False

                # Final condition for labeling
                if front_touch_count >= 2 or within_front_boundary:
                    group.loc[idx, 'side'] = 'front'
                elif rear_touch_count >= 2 or within_rear_boundary:
                    group.loc[idx, 'side'] = 'rear'

    # Step 5: Fill remaining NaN sides based on shared_side
    group.loc[group['side'].isnull() & (group['shared_side'] == True), 'side'] = 'Interior side'
    group.loc[group['side'].isnull() & (group['shared_side'] == False), 'side'] = 'Exterior side'

    return group

jagged_parcel = jagged_parcel.groupby('parcel_id').apply(classify_jagged_edges)
jagged_parcel = jagged_parcel.reset_index(level=0, drop=True)

# %% [markdown]
# ## Curve Parcels Edges

# %%
# calculate the angle between the each parcel seg and nearest road seg
curve_parcel = parcel_seg[parcel_seg['parcel_label'] == 'curve parcel']
curve_parcel['parcel_bearing'] = curve_parcel['geometry'].apply(fun_bearing_ra)
curve_parcel['road_bearing'] = curve_parcel['road_geometry'].apply(fun_bearing_ra)
curve_parcel['angle'] = curve_parcel.apply(
    lambda row: calculate_angle_difference(row['geometry'], row['road_geometry']), axis=1
)

# calculate the distance between the each parcel seg and nearest road seg
curve_parcel['distance_to_road'] = curve_parcel.apply(lambda row: row['geometry'].centroid.distance(row['road_geometry']), axis=1)

# %%
def classify_curve_edges(group):
    # Mark the smallest and largest distances as front and rear
    idx_min_distance = group['distance_to_road'].idxmin()
    idx_max_distance = group['distance_to_road'].idxmax()
    group.loc[idx_min_distance, 'side'] = 'front'
    group.loc[idx_max_distance, 'side'] = 'rear'

    # Check for angle < 20 and adjacency to front or rear
    for idx, row in group.iterrows():
        if pd.isnull(row['side']) and row['angle'] < 20:
            if group.loc[idx_min_distance, 'geometry'].touches(row['geometry']):
                group.loc[idx, 'side'] = 'front'
            elif group.loc[idx_max_distance, 'geometry'].touches(row['geometry']):
                group.loc[idx, 'side'] = 'rear'

    # Fill remaining NaN sides based on shared_side
    group.loc[group['side'].isnull() & (group['shared_side'] == True), 'side'] = 'Interior side'
    group.loc[group['side'].isnull() & (group['shared_side'] == False), 'side'] = 'Exterior side'

    return group

curve_parcel = curve_parcel.groupby('parcel_id').apply(classify_curve_edges)
curve_parcel = curve_parcel.reset_index(level=0, drop=True)

# %% [markdown]
# ## Cul_De_Sac Parcels Edges

# %%
# calculate the angle between the each parcel seg and nearest road seg
cul_de_sac_parcel = parcel_seg[parcel_seg['parcel_label'] == 'cul_de_sac parcel']
cul_de_sac_parcel['parcel_bearing'] = cul_de_sac_parcel['geometry'].apply(fun_bearing_ra)
cul_de_sac_parcel['road_bearing'] = cul_de_sac_parcel['road_geometry'].apply(fun_bearing_ra)
cul_de_sac_parcel['angle'] = cul_de_sac_parcel.apply(
    lambda row: calculate_angle_difference(row['geometry'], row['road_geometry']), axis=1
)

# calculate the distance between the each parcel seg and nearest road seg
cul_de_sac_parcel['distance_to_road'] = cul_de_sac_parcel.apply(lambda row: row['geometry'].centroid.distance(row['road_geometry']), axis=1)

# %%
# Modified classify_cul_de_sac_edges function
def classify_cul_de_sac_edges(group):
    """
    This function classifies cul-de-sac parcel edges based on their distance to the road and connectivity.
    It assigns each edge to either 'rear', 'front', 'Interior side', or 'Exterior side' based on certain rules.
    """

    # 1. Mark the farthest edge from the road as 'rear'
    idx_max_distance = group['distance_to_road'].idxmax()
    group.loc[idx_max_distance, 'side'] = 'rear'

    # 2. Select edges where shared_side == False (i.e., edges that are not shared with adjacent parcels)
    non_shared_edges = group[group['shared_side'] == False]

    # 3. If there are multiple non-shared edges, check if they are connected
    if len(non_shared_edges) > 1:
        # Check if any of the non-shared edges are connected to each other
        connected = any(are_segments_connected(row1['geometry'], row2['geometry']) 
                        for _, row1 in non_shared_edges.iterrows() 
                        for _, row2 in non_shared_edges.iterrows() if row1.name != row2.name)

        # 4. If all non-shared edges are connected, mark them as 'front'
        if connected:
            group.loc[non_shared_edges.index, 'side'] = 'front'
        else:
            # 5. If not all are connected, select the closest one to the road and mark it as 'front'
            idx_min_distance = non_shared_edges['distance_to_road'].idxmin()
            group.loc[idx_min_distance, 'side'] = 'front'
    elif len(non_shared_edges) == 1:
        # 6. If there is only one non-shared edge, mark it as 'front'
        group.loc[non_shared_edges.index, 'side'] = 'front'

    # 7. Fill remaining NaN values based on the shared_side property
    group.loc[group['side'].isnull() & (group['shared_side'] == True), 'side'] = 'Interior side'
    group.loc[group['side'].isnull() & (group['shared_side'] == False), 'side'] = 'Exterior side'

    return group

cul_de_sac_parcel = cul_de_sac_parcel.groupby('parcel_id').apply(classify_cul_de_sac_edges)
cul_de_sac_parcel = cul_de_sac_parcel.reset_index(level=0, drop=True)

# %% [markdown]
# ## No Match Address Parcels Edges

# %%
# calculate the angle between the each parcel seg and nearest road seg
no_match_address_parcel = parcel_seg[parcel_seg['parcel_label'] == 'no_match_address']
no_match_address_parcel['parcel_bearing'] = no_match_address_parcel['geometry'].apply(fun_bearing_ra)
no_match_address_parcel['road_bearing'] = None
no_match_address_parcel['angle'] = None

# calculate the distance between the each parcel seg and nearest road seg
no_match_address_parcel['distance_to_road'] = None

# %%
def calculate_temangle_difference(bearing1, bearing2):

    # Calculate the absolute angle difference and ensure it is <= 180 degrees
    delta_theta = bearing2 - bearing1
    
    # Ensure the angle is between -π and π
    if delta_theta > math.pi:
        delta_theta -= 2 * math.pi
    elif delta_theta < -math.pi:
        delta_theta += 2 * math.pi
    
    # Convert the angle to degrees
    angle_between_degrees = math.degrees(abs(delta_theta))
    
    # Return the smaller angle difference (angle or its supplement)
    return min(angle_between_degrees, 180 - angle_between_degrees)


def classify_no_match_address_sides(group):
    # Step 1: Initialize the 'side' column with None
    group['side'] = None
    
    
    # Step 2: Check if there is only one 'False' in shared_side
    if (group['shared_side'] == False).sum() == 1:
        # If there is exactly one 'False', set that row's 'side' to 'front'
        group.loc[group['shared_side'] == False, 'side'] = 'front'
        
        # Step 3: Create a new column 'temp_angle' for angles between 'shared_side=True' and 'side=front'
        # Get the parcel_bearing of the 'front' side
        front_bearing = group.loc[group['side'] == 'front', 'parcel_bearing'].values[0]
        # Calculate the angle difference for each 'shared_side=True' row
        group['temp_angle'] = group.apply(
            lambda row: calculate_temangle_difference(front_bearing, row['parcel_bearing']) if row['shared_side'] == True else None,
            axis=1
        )
        
        # Step 4: Create a new column 'centroid_point' for the midpoint of each geometry and Calculate distance from 'side=None' rows to 'front' row
        group['centroid_point'] = group['geometry'].apply(lambda geom: geom.interpolate(0.5, normalized=True))
        # Get the centroid of the 'front' geometry
        front_centroid = group.loc[group['side'] == 'front', 'centroid_point'].values[0]
        # Calculate distance for each 'side=None' row
        group['distance_to_front'] = group.apply(
            lambda row: row['centroid_point'].distance(front_centroid) if row['side'] is None else None,
            axis=1
        )
        
        # Step 5: Identify the row with side=None, temp_angle < 15, and maximum distance as 'rear'
        candidates = group[(group['side'].isnull()) & (group['temp_angle'] < 15)]
        if not candidates.empty:
            # Find the row with the maximum distance to 'front'
            rear_index = candidates['distance_to_front'].idxmax()
            group.loc[rear_index, 'side'] = 'rear'
            
        # Step 6: For remaining rows, find shared_side=True and mark as 'Interior side'
        shared_side_true = group[(group['side'].isnull()) & (group['shared_side'] == True)]
        group.loc[shared_side_true.index, 'side'] = 'Interior side'
        # Step 7: Label the remaining rows as 'Exterior side'
        group.loc[group['side'].isnull(), 'side'] = 'Exterior side'
        
    else:
        # If there is not exactly one 'False', perform Steps 6 and 7 directly
        shared_side_true = group[(group['side'].isnull()) & (group['shared_side'] == True)]
        group.loc[shared_side_true.index, 'side'] = 'Interior side'
        group.loc[group['side'].isnull(), 'side'] = 'Exterior side'
    
    return group


# Apply the function to each group
no_match_address_parcel = no_match_address_parcel.groupby('parcel_id').apply(classify_no_match_address_sides)
no_match_address_parcel = no_match_address_parcel.reset_index(level=0, drop=True)

no_match_address_parcel = no_match_address_parcel.drop(columns=['temp_angle', 'centroid_point', 'distance_to_front'], errors='ignore')

# %% [markdown]
# ## No Address Parcels Edges

# %%
# calculate the angle between the each parcel seg and nearest road seg
no_address_parcel = parcel_seg[parcel_seg['parcel_label'] == 'parcel without address']
no_address_parcel['parcel_bearing'] = no_address_parcel['geometry'].apply(fun_bearing_ra)
no_address_parcel['road_bearing'] = no_address_parcel['road_geometry'].apply(fun_bearing_ra)
no_address_parcel['angle'] = no_address_parcel.apply(
    lambda row: calculate_angle_difference(row['geometry'], row['road_geometry']), axis=1
)

# calculate the distance between the each parcel seg and nearest road seg
no_address_parcel['distance_to_road'] = no_address_parcel.apply(lambda row: row['geometry'].centroid.distance(row['road_geometry']), axis=1)

# %%
# Group by 'parcel_id' and perform the operations within each group
def classify_no_address_sides(group):
    # Create a new column 'side'
    group['side'] = None
    
    # Step 1: Filter rows where 'angle' is less than 20 degrees
    valid_rows = group[group['angle'] < 20]
    
    if not valid_rows.empty:
        # Mark the smallest and largest distances as front and rear within the filtered rows
        idx_min_distance = valid_rows['distance_to_road'].idxmin()
        idx_max_distance = valid_rows['distance_to_road'].idxmax()
        group.loc[idx_min_distance, 'side'] = 'front'
        group.loc[idx_max_distance, 'side'] = 'rear'
    
    # Step 2: For remaining rows, find shared_side=True and mark as 'Interior side'
    shared_side_true = group[(group['side'].isnull()) & (group['shared_side'] == True)]
    group.loc[shared_side_true.index, 'side'] = 'Interior side'
    
    # Step 3: Label the remaining rows as 'Exterior side'
    group.loc[group['side'].isnull(), 'side'] = 'Exterior side'
    
    return group

# Apply the function to each group
no_address_parcel = no_address_parcel.groupby('parcel_id').apply(classify_no_address_sides)
no_address_parcel = no_address_parcel.reset_index(level=0, drop=True)

# %% [markdown]
# ## Duplicated Address Parcels Edges

# %%
# calculate the angle between the each parcel seg and nearest road seg
duplicated_address_parcel = parcel_seg[parcel_seg['parcel_label'] == 'duplicated address']
duplicated_address_parcel['parcel_bearing'] = duplicated_address_parcel['geometry'].apply(fun_bearing_ra)
duplicated_address_parcel['road_bearing'] = duplicated_address_parcel['road_geometry'].apply(fun_bearing_ra)
duplicated_address_parcel['angle'] = duplicated_address_parcel.apply(
    lambda row: calculate_angle_difference(row['geometry'], row['road_geometry']), axis=1
)

# calculate the distance between the each parcel seg and nearest road seg
duplicated_address_parcel['distance_to_road'] = duplicated_address_parcel.apply(lambda row: row['geometry'].centroid.distance(row['road_geometry']), axis=1)

# %%
duplicated_address_parcel = duplicated_address_parcel.groupby('parcel_id').apply(classify_sides)
duplicated_address_parcel = duplicated_address_parcel.reset_index(level=0, drop=True)

# %% [markdown]
# #  Step 7: Generate Results

# %%
# List of GeoDataFrames to combine
parcel_list = [
    regular_insid_parcel, regular_corner_parcel, special_parcel, jagged_parcel, curve_parcel,
    cul_de_sac_parcel, no_match_address_parcel, no_address_parcel, duplicated_address_parcel
]

# Concatenate all GeoDataFrames in the list and ensure 'crs' and 'geometry' are set
combined_parcel = gpd.GeoDataFrame(
    pd.concat(parcel_list, ignore_index=True),
    crs=parcel_seg.crs,  # Use the crs from the first GeoDataFrame in the list
    geometry='geometry'  # Ensure the geometry column is correctly set
)

combined_parcel['parcel_id'] = combined_parcel['parcel_id'].astype(str)
# Sort by 'parcel_id' to ensure similar parcel_id are together
combined_parcel = combined_parcel.sort_values(by='parcel_id').reset_index(drop=True)

# %%
edge_counts = combined_parcel.groupby('parcel_id').size()
combined_parcel['num_edges'] = combined_parcel['parcel_id'].map(edge_counts)

# %%
def update_label_cul_de_sac(group):
    if group['num_edges'].iloc[0] == 4:
        if 'front' in group['side'].values and 'rear' in group['side'].values:
            group['parcel_label'] = 'cul_de_sac parcel_standard'
        else:
            group['parcel_label'] = 'cul_de_sac parcel_other'
    else:
        # Directly change label if 'num_edges' is not equal to 4
        group['parcel_label'] = 'cul_de_sac parcel_other'
    return group

# Apply the function to each group and update the main DataFrame
updated_parcel_cul_de_sac = combined_parcel[combined_parcel['parcel_label'] == 'cul_de_sac parcel'].groupby('parcel_id', group_keys=False).apply(update_label_cul_de_sac)
combined_parcel.update(updated_parcel_cul_de_sac)

# %%
def update_label_curve(group):
    if group['num_edges'].iloc[0] == 4:
        if 'front' in group['side'].values and 'rear' in group['side'].values:
            group['parcel_label'] = 'curve parcel_standard'
        else:
            group['parcel_label'] = 'curve parcel_other'
    else:
        # Directly change label if 'num_edges' is not equal to 4
        group['parcel_label'] = 'curve parcel_other'
    return group

# Apply the function to each group and update the main DataFrame
updated_parcel_curve = combined_parcel[combined_parcel['parcel_label'] == 'curve parcel'].groupby('parcel_id', group_keys=False).apply(update_label_curve)
combined_parcel.update(updated_parcel_curve)

# %%
def update_label_nomatch(group):
    if group['num_edges'].iloc[0] == 4:
        if 'front' in group['side'].values and 'rear' in group['side'].values:
            group['parcel_label'] = 'no_match_address_standard'
        else:
            group['parcel_label'] = 'no_match_address_other'
    else:
        # Directly change label if 'num_edges' is not equal to 4
        group['parcel_label'] = 'no_match_address_other'
    return group

# Apply the function to each group and update the main DataFrame
updated_parcel_nomatch = combined_parcel[combined_parcel['parcel_label'] == 'no_match_address'].groupby('parcel_id', group_keys=False).apply(update_label_nomatch)
combined_parcel.update(updated_parcel_nomatch)

# %%
def update_label_noaddress(group):
    if group['num_edges'].iloc[0] == 4:
        if 'front' in group['side'].values and 'rear' in group['side'].values:
            group['parcel_label'] = 'no_address_parcel_standard'
        else:
            group['parcel_label'] = 'no_address_parcel_other'
    else:
        # Directly change label if 'num_edges' is not equal to 4
        group['parcel_label'] = 'no_address_parcel_other'
    return group

# Apply the function to each group and update the main DataFrame
updated_parcel_noaddress = combined_parcel[combined_parcel['parcel_label'] == 'parcel without address'].groupby('parcel_id', group_keys=False).apply(update_label_noaddress)
combined_parcel.update(updated_parcel_noaddress)

# %%
def update_label_special(group):
    if group['num_edges'].iloc[0] in [4, 5]:
        if 'front' in group['side'].values and 'rear' in group['side'].values:
            group['parcel_label'] = 'special parcel_standard'
        else:
            group['parcel_label'] = 'special parcel_other'
    else:
        # Directly change label if 'num_edges' is not equal to 4
        group['parcel_label'] = 'special parcel_other'
    return group

# Apply the function to each group and update the main DataFrame
updated_parcel_special = combined_parcel[combined_parcel['parcel_label'] == 'special parcel'].groupby('parcel_id', group_keys=False).apply(update_label_special)
combined_parcel.update(updated_parcel_special)

# %%
parcel_label_summary = combined_parcel.groupby('parcel_id')['parcel_label'].first().reset_index()
# Rename the columns for clarity
parcel_label_summary.columns = ['parcel_id', 'unique_parcel_labels']

confidence_area = parcel[['parcel_id','parcel_addr','landuse','parcel_label','geometry']].copy()
confidence_area['parcel_id'] = confidence_area['parcel_id'].astype(str)
confidence_area = confidence_area.merge(parcel_label_summary, on='parcel_id', how='left')
confidence_area['parcel_label'] = confidence_area['unique_parcel_labels']
confidence_area = confidence_area.drop(columns=['unique_parcel_labels'])


confidence_area['confidence_level'] = np.where(
    confidence_area['parcel_label'].isin(['regular inside parcel', 'regular corner parcel', 'special parcel_standard','curve parcel_standard','cul_de_sac parcel_standard','no_match_address_standard','no_address_parcel_standard']),
    'Yes', 'No'
)

# calculate the area/sqm, and transfer into the sq_acre
confidence_area['area_acre'] = confidence_area['geometry'].area * 0.000247105

# %%
def add_centroids_to_combined_parcel(confidence_area, combined_parcel):
    # Step 1: Calculate the centroid for each geometry and add it to a new column 'centroid_geometry'
    confidence_area['centroid_geometry'] = confidence_area['geometry'].centroid
    # Step 2: Group by 'parcel_id' and get the centroid for each group as a DataFrame
    centroids_by_parcel = confidence_area.groupby('parcel_id')['centroid_geometry'].apply(lambda x: x.iloc[-1]).reset_index()
    
    # Step 3: Add centroid data to the last row of the corresponding group in combined_parcel
    rows_to_add = []  # List to store new rows to be added

    for _, row in centroids_by_parcel.iterrows():
        parcel_id = row['parcel_id']
        centroid_geometry = row['centroid_geometry']
        
        # Get rows in combined_parcel that match the parcel_id
        parcel_group = combined_parcel[combined_parcel['parcel_id'] == parcel_id]
        
        # Add centroid row at the end of the group
        if not parcel_group.empty:
            # Create a new row, setting centroid as geometry, keeping other columns empty or default
            new_row = parcel_group.iloc[-1].copy()
            new_row['geometry'] = centroid_geometry
            new_row['side'] = 'centroid'  # Set the 'side' column value to 'centroid'
            rows_to_add.append(new_row)  # Add new row to list
    
    # Use pd.concat to add all new rows to combined_parcel
    combined_parcel = pd.concat([combined_parcel, pd.DataFrame(rows_to_add)], ignore_index=True)
    
    # Create a helper column to ensure centroid rows appear at the end of each group
    combined_parcel['is_centroid'] = combined_parcel['side'] == 'centroid'
    # Sort by 'parcel_id' and 'is_centroid' so that centroid rows are at the end of each group
    combined_parcel = combined_parcel.sort_values(by=['parcel_id', 'is_centroid'], ascending=[True, True]).reset_index(drop=True)
    # Step 6: Drop the helper column
    combined_parcel = combined_parcel.drop(columns=['is_centroid'])
    
    return combined_parcel

# Use the function to directly update combined_parcel
combined_parcel = add_centroids_to_combined_parcel(confidence_area, combined_parcel)

# %%
combined_parcel.drop(columns='road_geometry').to_file(rf'Dallas_County/{County_name}_combined_parcel.geojson', driver='GeoJSON')
confidence_area.drop(columns='centroid_geometry').to_file(rf'Dallas_County/{County_name}_confidence_area.geojson', driver='GeoJSON')

# %%
# Calculate the number of "Yes"
yes_count = (confidence_area['confidence_level'] == 'Yes').sum()
# Calculate the total length
total_count = len(confidence_area)
# Calculate confidence percentage
confidence_per = (yes_count / total_count) * 100

# Create value counts for parcel labels
value_counts = confidence_area['parcel_label'].value_counts().reset_index()
value_counts.columns = ['parcel_label', 'count']

# Calculate percentage for each parcel type
value_counts['percentage'] = (value_counts['count'] / value_counts['count'].sum()) * 100

# Adding a new column to classify into 'Confident' and 'Non-confident' based on parcel type
value_counts['Confidence'] = value_counts['parcel_label'].apply(
    lambda x: 'Confident' if x in ['regular inside parcel', 'regular corner parcel', 'special parcel_standard','curve parcel_standard','cul_de_sac parcel_standard','no_match_address_standard','no_address_parcel_standard'] else 'Non-confident'
)

# Plotting with seaborn
plt.figure(figsize=(10, 6))
sns.barplot(
    x='parcel_label', 
    y='count', 
    hue='Confidence', 
    data=value_counts, 
    palette={'Confident': 'skyblue', 'Non-confident': 'lightcoral'}
)

# Add percentage labels above each bar
for index, row in value_counts.iterrows():
    plt.text(
        index, 
        row['count'], 
        f"{row['percentage']:.2f}%", 
        ha='center', 
        va='bottom'
    )

# Customizing plot
plt.xticks(rotation=45, ha='right')
plt.xlabel('Parcel Type')
plt.ylabel('Count')
# Add confidence percentage to the title
plt.title(f'Parcel Type Classification with Confidence Levels in {County_name} for all landuse (Confidence: {confidence_per:.2f}%)')
plt.legend(title='Confidence Level', frameon=False)

# Remove top and right borders
sns.despine()
plt.tight_layout()

plt.savefig(f'Dallas_County/{County_name}_confidence_all_landuse.png')

# Show plot
# plt.show()

# %%
# Calculate mean and median for 'Yes' and 'No' confidence levels
yes_median = confidence_area[confidence_area['confidence_level'] == 'Yes']['area_acre'].median()
no_median = confidence_area[confidence_area['confidence_level'] == 'No']['area_acre'].median()


# Calculate minimum and 75th percentile for 'Yes' and 'No' confidence levels
yes_75th = confidence_area[confidence_area['confidence_level'] == 'Yes']['area_acre'].quantile(0.75)
no_75th = confidence_area[confidence_area['confidence_level'] == 'No']['area_acre'].quantile(0.75)
# Filter data within the desired range for better visibility
filtered_data = confidence_area[
    ((confidence_area['confidence_level'] == 'Yes') & (confidence_area['area_acre'] <= yes_75th)) |
    ((confidence_area['confidence_level'] == 'No') & (confidence_area['area_acre'] <= no_75th))
]

# Plotting the filtered histogram with mean and median lines
plt.figure(figsize=(12, 6))
sns.histplot(
    data=filtered_data,
    x='area_acre',
    hue='confidence_level',
    palette={'Yes': 'skyblue', 'No': 'lightcoral'},
    multiple='layer',
    bins=50,
    alpha=0.5
)

# Add mean and median lines for 'Yes' confidence level
plt.axvline(yes_median, color='skyblue', linestyle='--', linewidth=1.5, label="Yes Median")
# Add mean and median lines for 'No' confidence level
plt.axvline(no_median, color='lightcoral', linestyle='--', linewidth=1.5, label="No Median")

# Set plot title and labels
plt.title('Distribution of Area (in Acres) by Confidence Level (Filtered to 75th Percentile) in All Parcel Type')
plt.xlabel('Area (Acres)')
plt.ylabel('Count')

# Add legend for clarity
plt.legend()

plt.savefig(f'Dallas_County/{County_name}_AreaDistribution_all_parceltype.png')
# Display plot
# plt.show()


# %%
# Calculate medians for each parcel label
median_lines = confidence_area.groupby('parcel_label')['area_acre'].median()

# Define parcel groups for color coding
skyblue_labels = [
    'regular inside parcel', 'regular corner parcel', 'special parcel_standard',
    'curve parcel_standard', 'cul_de_sac parcel_standard', 'no_match_address_standard','no_address_parcel_standard'
]
lightcoral_labels = [label for label in confidence_area['parcel_label'].unique() if label not in skyblue_labels]

# Set up the multi-plot layout
num_plots = len(confidence_area['parcel_label'].unique())
fig, axes = plt.subplots(nrows=num_plots // 3 + 1, ncols=3, figsize=(15, 4 * (num_plots // 3 + 1)), constrained_layout=True)

# Flatten axes array for easy iteration
axes = axes.flatten()

# Plot histogram for each unique parcel label
for i, label in enumerate(confidence_area['parcel_label'].unique()):
    ax = axes[i]
    label_data = confidence_area[confidence_area['parcel_label'] == label]
    color = 'skyblue' if label in skyblue_labels else 'lightcoral'
    
    # Plot histogram
    sns.histplot(
        data=label_data,
        x='area_acre',
        color=color,
        bins=50,
        ax=ax
    )
    
    # Plot median line for the group
    median_value = median_lines[label]
    ax.axvline(median_value, color='black', linestyle='--', linewidth=1, label=f'Median: {median_value:.2f}')
    ax.text(
        0.95, 0.98, f'Median: {median_value:.2f}', 
        transform=ax.transAxes, ha='right', va='top', fontsize=10, color='black'
    )

    # Set title and labels
    ax.set_title(f'Distribution for {label}')
    ax.set_xlabel('Area (Acres)')
    ax.set_ylabel('Count')

# Hide any unused subplots
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

# Add a main title to the entire figure
fig.suptitle('Distribution of Area (in Acres) by Confidence Level in Classified Parcel Type', fontsize=16)

plt.savefig(f'Dallas_County/{County_name}_AreaDistribution_classified_parceltype.png')
# Display the plot
# plt.show()

# %% [markdown]
#  ### Count the total time taken

# %%
# record the end time
end_time = time.time()

# print the execution time
execution_time = end_time - start_time
execution_time_in_minutes = execution_time / 60
print(f"Execution time: {execution_time_in_minutes:.2f} minutes")



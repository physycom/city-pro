"""
    In this file we hold functions to analyze trajectories.
    Main object is going to be traj_dataframe.
    NOTE:
    Required Columns:
        - user_id: int: user id.
        - timestamp: int: timestamp.
        - lat: float: latitude.
        - lng: float: longitude.
    Optional Columns:
        NOTE: The name of the columns used for speed, time, time intervals, are going to be standard.
        - speed: float: speed of the user. (m/s)
        - dt_sec: float: time interval. (s)
        - speed_kmh: float: speed of the user. (km/h)
        

    
"""
import skmob as sk
import pandas as pd
import polars as pl
from os.path import isfile
import numpy as np


## --------------- Trajectory DataFrame Functions --------------- ##

def open_and_ensure_df_traj_has_right_columns(df,
                                  FileName,
                                  column_latitude = "latitude",
                                  column_longitude = "longitude",
                                  column_timestamp = "timestamp"
                                  ):
    """
        @brief:
            - This function is used to read a dataframe with columns:
                - latitude
                - longitude
                - timestamp
                - user_id
            Make sure that the columns are the right ones.
                - lat
                - lng
                - datetime
            If the dataframe is not provided, it reads the file.
        @param df: pd.DataFrame or pl.DataFrame:
            - dataframe with the trajectories.
        @param FileName: str:
            - name of the file.
        @param column_latitude: str:
            - name of the column that contains the latitude.
        @param column_longitude: str:
            - name of the column that contains the longitude.
        @param column_timestamp: str:
            - name of the column that contains the timestamp.
        NOTE: USAGE:
            df = open_and_ensure_df_traj_has_right_columns(df = None,
                                                            FileName = "/path/file.csv",
                                                            column_latitude = "latitude",        
                                                            column_longitude = "longitude",
                                                            column_timestamp = "timestamp")
    """
    # Check the input is right
    if isinstance(df,pd.DataFrame):
        pass
    if isinstance(df,pl.DataFrame):
        df = df.to_pandas()
        pass
    if df is None:
        assert FileName is not None, "FileName must be provided, None given."
        assert isfile(FileName), f"FileName must be an existing file. Given:\n{FileName}"
        try:
            df = pd.read_csv(FileName,sep=";")
        except Exception as e:
            try:
                df = pd.read_csv(FileName,sep=",")
            except Exception as e:
                raise ValueError(e)
    # Make Sure "lat" and "lng" are the columns
    df.rename(columns={column_latitude:"lat",
                       column_longitude:"lng"},
                       inplace=True)
    # Make Sure "timestamp" contains int, and can be used to have datetime.
    if isinstance(df[column_timestamp].to_numpy()[0],np.int64):
        df["datetime"] = pd.to_datetime(df[column_timestamp],unit='s')
    elif isinstance(df[column_timestamp].to_list()[0],str):
        try:
            df[column_timestamp] = df[column_timestamp].apply(lambda x: int(x))
            df["datetime"] = pd.to_datetime(df[column_timestamp],unit='s')
        except Exception as e:
            raise ValueError(e)
    else:
        raise ValueError(f"Timestamp must be an integer. type {type(df[column_timestamp].to_numpy()[0])} given.")
    
    return df


def add_segment_distance_user_from_timestamp_and_speed(df,
                                                       user_column,
                                                       timestamp_column,
                                                       speed_column,
                                                       distance_dt_column,
                                                       speed_km_h_column,
                                                       dt_sec_column,
                                                       ):
    """
        @brief:
            - This function is used to add the distance and time to the dataframe.
        @param df: pd.DataFrame:
            - columns: user_column, datetime, lat, lng, speed_column, timestamp_column. see:    open_and_ensure_df_traj_has_right_columns
        @param user_column: str:
            - name of the column that contains the user_id.
        @param timestamp_column: str:
            - name of the column that contains the timestamp.
        @param speed_column: str:
            - name of the column that contains the speed.
        @param distance_dt_column: str:
            - name of the column that will contain the distance.
        @param speed_km_h_column: str:
            - name of the column that will contain the speed in km/h.
        @param dt_sec_column: str:
            - name of the column that will contain the time interval.
        @return:
            Trajdf: skmob.TrajDataFrame:
                - dataframe with the columns:
                    - user_id -> int
                    - datetime -> datetime
                    - lat -> float
                    - lng -> float
                    - speed -> float (m/s)
                    - timestamp -> int
                    - distance_dt_column -> float
                    - speed_km_h_column -> float
                    - dt_sec_column -> float
    """
    assert timestamp_column in df.columns, f"{timestamp_column} must be a column in df."
    assert speed_column in df.columns, f"{speed_column} must be a column in df."
    assert user_column in df.columns, f"{user_column} must be a column in df."
    distance_users = []
    time_users = []
    v_kmhs = []
    for _,df_user in df.groupby(user_column):
        dt = np.diff(df_user[timestamp_column].values)
        dt = np.array(np.append([0],dt))/60
        v = df_user[speed_column].to_numpy()
        v_km_h = v*3.6
        time_users.extend(dt)
        distance_users.extend(v*dt)
        v_kmhs.extend(v_km_h)
    df[distance_dt_column] = distance_users
    df[speed_km_h_column] = v_kmhs
    df[dt_sec_column] = time_users
    df = df.loc[df[speed_km_h_column]<140]
    Trajdf = sk.TrajDataFrame(df)
    return Trajdf


def get_trajdf(FileName,column_latitude, column_longitude, column_timestamp,user_column, timestamp_column,
               speed_column, distance_dt_column, speed_km_h_column, dt_sec_column):
    df = open_and_ensure_df_traj_has_right_columns(df = None,
                                                    FileName = FileName,
                                                    column_latitude = column_latitude,
                                                    column_longitude = column_longitude,
                                                    column_timestamp = column_timestamp)
    # Add Distance and Time to the DataFrame
    Trajdf = add_segment_distance_user_from_timestamp_and_speed(df,
                                                                user_column,
                                                                timestamp_column,
                                                                speed_column,
                                                                distance_dt_column,
                                                                speed_km_h_column,
                                                                dt_sec_column)
    return Trajdf


# 

def generate_t_vect(Trajdf,
                    sampling_time_in_1_hour = 4,
                    column_timestamp = "timestamp"
                    ):
    """
        @brief:
            - Generates a vector of time from the dataframe.
        @param Trajdf: TrajDataFrame:

        @param column_timestamp: str:
            - name of the column that contains the timestamp.
        @param sampling_time_in_1_hour: int:
            - number of intervals in 1 hour. 4 -> 15 minutes.
        @return:
            - np.array: t_vect
    """ 
    min_t,max_t = Trajdf.timestamp.min(),Trajdf.timestamp.max()
    delta_t_hours = int(max_t - min_t)/3600
    n_intervals = int(delta_t_hours*sampling_time_in_1_hour)
    t_vect = np.linspace(min_t,max_t,n_intervals)
    return t_vect
def compute_interval_2_possibly_sharing_mean_users(Trajdf,
                                                   t_vect,
                                                   sampling_time_in_1_hour = 4,
                                                   column_timestamp = "timestamp",
                                                   column_user = "user_id"
                                                   ):
    """
        @brief:
            - Divides users in intervals of time.
        @param Trajdf: TrajDataFrame:
            - dataframe with the trajectories.
        @param column_timestamp: str:
            - name of the column that contains the timestamp.
        @param column_user: str:
            - name of the column that contains the user_id.
        @param sampling_time_in_1_hour: int:
            - number of intervals in 1 hour. 4 -> 15 minutes.
        @return:
            - dict: {interval:mean_users}
    """
    t_vect = generate_t_vect(Trajdf,
                    sampling_time_in_1_hour = sampling_time_in_1_hour,
                    column_timestamp = column_timestamp)
    n_intervals = len(t_vect)
    interval_2_user = {t_vect[i]:[] for i in range(n_intervals)}
    for index_interval_seconds in range(n_intervals):
        if index_interval_seconds < n_intervals - 1:
            df_interval = Trajdf.loc[Trajdf[column_timestamp] < t_vect[index_interval_seconds+1]]
            if len(df_interval)>0:
                df_interval = df_interval.loc[df_interval[column_timestamp] > t_vect[index_interval_seconds]]
                interval_2_user[t_vect[index_interval_seconds]] = df_interval[column_user].to_list()
    return interval_2_user

# Filtering Users

def _get_users_interval_speed_speed(Trajdf,interval = [70,140]):
    """
        @param Trajdf: TrajDataFrame
        @param threshold: threshold for the speed
        @return users_big_speed: list of users with speed
    """
    assert len(interval) == 2, "Interval must be a list of two elements"
    users_big_speed = []
    for user,df_user in Trajdf.groupby("user_id"):
        if df_user["speed_kmh"].mean() > interval[0] and df_user["speed_kmh"].mean() < interval[1]:
            users_big_speed.append(user)
    return users_big_speed

def _get_users_interval_rate_sampling(Trajdf,interval = [40,60]):
    """
        @param Trajdf: TrajDataFrame
        @param interval: threshold for the speed
        @return users_big_speed: list of users with speed
    """
    assert len(interval) == 2, "Interval must be a list of two elements"
    users_small_rate_sampling = []
    for user,df_user in Trajdf.groupby("user_id"):
        if df_user["dt_sec"].mean() > interval[0] and df_user["dt_sec"].mean() < interval[1]:
            users_small_rate_sampling.append(user)
    return users_small_rate_sampling






# ---------------- SHORTEST PATH RECONSTRUCTION ---------------- ##
from scipy.spatial import cKDTree
import networkx as nx
from shapely.geometry import LineString, Point

def get_graph_from_gdf_roads(roads_gdf_projected):
    # Step 4: Build a graph from the LineStrings
    G = nx.Graph()

    # Extract coordinates from LineStrings and add them as nodes
    node_id = 0
    node_coords = {}  # Map (x, y) to node_id
    coords_node = {}  # Map node_id to (x, y)

    for idx, row in roads_gdf_projected.iterrows():
        if isinstance(row.geometry, LineString):
            coords = list(row.geometry.coords)
            
            # Add nodes for each point in the LineString if not already added
            for coord in coords:
                if coord not in node_coords:
                    node_coords[coord] = node_id
                    coords_node[node_id] = coord
                    G.add_node(node_id, x=coord[0], y=coord[1])
                    node_id += 1
            
            # Add edges between consecutive points
            for i in range(len(coords) - 1):
                start_id = node_coords[coords[i]]
                end_id = node_coords[coords[i+1]]
                
                # Calculate Euclidean distance as edge weight
                distance = np.sqrt((coords[i+1][0] - coords[i][0])**2 + 
                                (coords[i+1][1] - coords[i][1])**2)
                
                G.add_edge(start_id, end_id, weight=distance, road_id=idx)
    return G, coords_node


def from_gdf_2_path_geometries(trajectory_gdf_projected,
                               G,
                               coords_node,
                               col_geometry_road_in_traj_gdf="nearest_road_geometry"):
    # Build a KD-Tree for efficient nearest node lookup
    nodes_array = np.array(list(coords_node.values()))
    tree = cKDTree(nodes_array)

    # Step 5: Compute shortest paths between consecutive trajectory points
    paths = []
    path_geometries = []

    # Group by user_id to handle multiple trajectories
    for user_id, user_df in trajectory_gdf_projected.groupby('user_id'):
        # Sort by timestamp to ensure correct order
        user_df = user_df.sort_values('timestamp')
        
        for i in range(len(user_df) - 1):
            start_point = (user_df.iloc[i][col_geometry_road_in_traj_gdf].x, user_df.iloc[i][col_geometry_road_in_traj_gdf].y)
            end_point = (user_df.iloc[i+1][col_geometry_road_in_traj_gdf].x, user_df.iloc[i+1][col_geometry_road_in_traj_gdf].y)
            
            # Find nearest nodes in the graph using KD-Tree
            _, start_node_idx = tree.query(start_point)
            start_node = list(coords_node.keys())[start_node_idx]
            
            _, end_node_idx = tree.query(end_point)
            end_node = list(coords_node.keys())[end_node_idx]
            
            try:
                # Compute shortest path
                path = nx.shortest_path(G, start_node, end_node, weight='weight')
                paths.append(path)
                
                # Create LineString geometry for the path
                path_coords = [coords_node[node] for node in path]
                path_geom = LineString(path_coords)
                
                path_geometries.append({
                    'user_id': user_id,
                    'start_time': user_df.iloc[i].datetime,
                    'end_time': user_df.iloc[i+1].datetime,
                    'geometry': path_geom
                })
            except nx.NetworkXNoPath:
                print(f"No path found between points at {i} and {i+1} for user {user_id}")
            except Exception as e:
                print(f"Error finding path for user {user_id} between points {i} and {i+1}: {e}")
    return paths, path_geometries

def get_closest_point_on_linestring(row):
    """
        Choose the closest point on the road LineString to the trajectory point.
        This function is used to find the closest point on the road LineString to the trajectory point
        and return it as a Point geometry.
        In this way when looking for the shortest path from_gdf_2_path_geometries we are looking at the points in the graph and not other points
    """
    point = row.geometry  # the trajectory point
    line = row.nearest_road_geometry  # the nearest road LineString
    if line is None or point is None:
        return None
    # Project the point onto the line (returns distance along the line)
    distance_on_line = line.project(point)
    # Get the actual point on the line at that distance
    return line.interpolate(distance_on_line)


## --------------- User Pick ---------------- ##
def pick_random_users_by_class(Fcm, user_column, user_class_column="class", seed=None):
    """
    Randomly pick one user from each class (0, 1, 2, 3) from the FCM dataframe.
    
    Parameters:
    -----------
    Fcm : pd.DataFrame or pl.DataFrame
        FCM dataframe containing user data
    user_column : str
        Column name containing user IDs
    user_class_column : str, default="class"
        Column name containing class labels
    seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    dict or None
        Dictionary with class as key and randomly selected user_id as value
        Returns None if any class is missing
        
    Example:
    --------
    selected_users = pick_random_users_by_class(Fcm, "user_id", "class", seed=42)
    # Result: {0: 'user_123', 1: 'user_456', 2: 'user_789', 3: 'user_321'}
    """
    import random
    import numpy as np
    
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    target_classes = [0, 1, 2, 3]
    selected_users = {}
    
    # Handle both pandas and polars dataframes
    if isinstance(Fcm, pd.DataFrame):  # pandas
        for class_id in target_classes:
            class_users = Fcm[Fcm[user_class_column] == class_id][user_column].unique()
            
            if len(class_users) == 0:
                print(f"Warning: No users found for class {class_id}")
                return None
            
            # Randomly select one user from this class
            selected_user = random.choice(class_users)
            selected_users[class_id] = selected_user
            
    else:  # polars
        import polars as pl
        for class_id in target_classes:
            class_users = Fcm.filter(pl.col(user_class_column) == class_id)[user_column].unique().to_list()
            if len(class_users) == 0:
                print(f"Warning: No users found for class {class_id}")
                return None
            
            # Randomly select one user from this class
            selected_user = random.choice(class_users)
            selected_users[class_id] = selected_user
    
    return selected_users


def sample_users_of_given_size_of_points_outside_bbox(CutSizes,UsersAvailable, df_cut_traj, users_cut, cut_size,cut_size_1,number_users_filtered, number_sampled_users_per_size, user_column_cut_traj, size_column_cut_traj):
    """
        Used in pick_user_by_cut_size to sample users of a given size of points outside a bounding box.
        This function checks the number of users filtered by the cut size and samples users accordingly.
        If no users are found, it skips the range.
    """
    if number_users_filtered == 0:
        print(f"No users found for cut size {cut_size} to {cut_size_1}. Skipping this range.")
    elif number_users_filtered < number_sampled_users_per_size:
        user_cut = users_cut[0]  # Take the first user from the range
        UsersAvailable.append(user_cut)
        cut_size = df_cut_traj.filter(pl.col(user_column_cut_traj) == user_cut)[size_column_cut_traj].to_list()[0]
        CutSizes.append(cut_size)
    else:
        # Randomly select users from the filtered list
        sampled_users = np.random.choice(users_cut, size=number_sampled_users_per_size, replace=False)
        for user_cut in sampled_users:
            UsersAvailable.append(user_cut)
            cut_size = df_cut_traj.filter(pl.col(user_column_cut_traj) == user_cut)[size_column_cut_traj].to_list()[0]
            CutSizes.append(cut_size)
    return UsersAvailable, CutSizes


def pick_user_by_cut_size(df_cut_traj, cut_sizes, number_sampled_users_per_size = 2,user_column_cut_traj="id_act",size_column_cut_traj="size_traj"):
    """
    Pick users based on cut sizes from the trajectory dataframe.
    This function samples users from the trajectory dataframe based on specified cut sizes.
    The goal of the function is to choose the trajectories of given sizes
    """
    UsersAvailable = []
    CutSizes = []
    i = 0
    for i in range(len(cut_sizes)):
        if i != 0 and i != len(cut_sizes) - 1:  # Skip the first and last cut sizes
            cut_size = cut_sizes[i]
            cut_size_1 = cut_sizes[i + 1]
            users_cut = df_cut_traj.filter(pl.col(size_column_cut_traj) > cut_size,
                                        pl.col(size_column_cut_traj) < cut_size_1)[user_column_cut_traj].to_list()
            number_users_filtered = len(users_cut)
            UsersAvailable, CutSizes = sample_users_of_given_size_of_points_outside_bbox(CutSizes,UsersAvailable, df_cut_traj, users_cut, cut_size,cut_size_1,number_users_filtered, number_sampled_users_per_size, user_column_cut_traj, size_column_cut_traj)
        elif i == 0:  # First cut size
            cut_size = cut_sizes[i]
            cut_size_1 = cut_sizes[i + 1]
            users_cut = df_cut_traj.filter(pl.col(size_column_cut_traj) == cut_size)[user_column_cut_traj].to_list()
            number_users_filtered = len(users_cut)
            UsersAvailable, CutSizes = sample_users_of_given_size_of_points_outside_bbox(CutSizes,UsersAvailable, df_cut_traj, users_cut, cut_size,cut_size_1,number_users_filtered, number_sampled_users_per_size, user_column_cut_traj, size_column_cut_traj)
        else:
            cut_size = cut_sizes[i]
            cut_size_1 = None
            users_cut = df_cut_traj.filter(pl.col(size_column_cut_traj) > cut_size)[user_column_cut_traj].to_list()
            number_users_filtered = len(users_cut)
            UsersAvailable, CutSizes = sample_users_of_given_size_of_points_outside_bbox(CutSizes,UsersAvailable, df_cut_traj, users_cut, cut_size,cut_size_1,number_users_filtered, number_sampled_users_per_size, user_column_cut_traj, size_column_cut_traj)
    return UsersAvailable, CutSizes  



# Appliable Filters For Trajectories
Condition2Sample = {"rate_sampling":_get_users_interval_rate_sampling,
                    "speed":_get_users_interval_speed_speed,

                       }



if __name__=="__main__":
    import os
    # Info About File, Day and Project
    name_project = "bologna_mdt_center"
    base_name = "bologna_mdt"
    date = "2022-07-01"
    traj_datafram_file = f"{base_name}_{date}_{date}_traj_dataframe.csv"
    FileName = os.path.join(os.environ["WORKSPACE"],"city-pro","output",name_project,traj_datafram_file)
    # Info About Format
    column_latitude = "latitude"
    column_longitude = "longitude"
    column_timestamp = "timestamp"
    # Open and Ensure the DataFrame has the right columns
    user_column = "user_id"
    timestamp_column = "timestamp"
    speed_column = "speed"
    distance_dt_column = "distance_dt"
    speed_km_h_column = "speed_kmh"
    dt_sec_column = "dt_sec"

    df = open_and_ensure_df_traj_has_right_columns(df = None,
                                                   FileName = FileName,
                                                   column_latitude = column_latitude,
                                                   column_longitude = column_longitude,
                                                   column_timestamp = column_timestamp)
    # Add Distance and Time to the DataFrame
    Trajdf = add_segment_distance_user_from_timestamp_and_speed(df,
                                                               user_column,
                                                               timestamp_column,
                                                               speed_column,
                                                               distance_dt_column,
                                                               speed_km_h_column,
                                                               dt_sec_column)
    # Apply Filters
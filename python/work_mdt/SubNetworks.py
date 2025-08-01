import pandas as pd
from shapely.geometry import LineString
import geopandas as gpd
import networkx as nx
from shapely.geometry import LineString
from tqdm import tqdm

def gdf_to_directed_networkx(gdf, 
                             id_column_edge=None, 
                             id_column_source=None,
                             id_column_target=None):
    """
    Convert a GeoDataFrame of road segments to a directed NetworkX graph.

    Parameters:
        gdf (GeoDataFrame): must contain LineString geometries
        id_columns (list[str], optional): columns to carry as edge attributes

    Returns:
        nx.DiGraph: directed road network graph
    """
    # Control that once I have the column for the source I have it also for the target
    if id_column_source is not None:
        is_target_none_if_source_is = id_column_target is None
        if is_target_none_if_source_is:
            raise ValueError("If id_column_source is provided, id_column_target must also be provided.")
    
    G = nx.DiGraph()    
    
    for idx, row in tqdm(gdf.iterrows()):
        geom = row.geometry
        if id_column_source is not None:
            start_id = row[id_column_source]
        if id_column_target is not None:
            end_id = row[id_column_target]
        if not isinstance(geom, LineString):
            continue  # skip invalid geometries

        coords = list(geom.coords)
        start = coords[0]
        end = coords[-1]

        # Optional: round coordinates to reduce duplicates
        start = tuple(round(c, 6) for c in start)
        end = tuple(round(c, 6) for c in end)

        attrs = {
                 'geometry': geom, 
                 'length': geom.length
                }
        if id_column_source is not None:
            G.add_edge(start_id, end_id, **attrs)
            if "coords" not in G.nodes[start_id]:
                G.nodes[start_id]["coords"] = start
            if "coords" not in G.nodes[end_id]:
                G.nodes[end_id]["coords"] = end

    return G



def Compute_Union_Column_gdf_network_simplified_from_class_days(GdfSimplified, Classes, Uninon_column_names, Days):
    """
        Compute the Union Column by checking at the classes columns of all the days.
        - Each day has a column "IntClassOrdered_"+Day that contains the class of the road in the day.
        - Iterate over each day to see wether the road is present or not in the class.
        - If the road is present in the class, assign the class to the Union column for that day.
        - If the road is not present in the class, assign -1 to the Union column for that day.
        - Finally, compute the Union column by taking the maximum value of the Union columns for each road. 
    """
    for Class in Classes:
        Union_column_name = Uninon_column_names[Class]
        roads_class = []
        for row,road_data in GdfSimplified.iterrows():
            is_in_class = False
            for Day in Days:
                if road_data["IntClassOrdered_"+Day] == Class:
                    is_in_class = True
                    roads_class.append(Class)
                    break
            if not is_in_class:
                roads_class.append(-1)
        GdfSimplified[Union_column_name] = roads_class

    # Union    
    union_class_per_road = []
    for row,road_data in GdfSimplified.iterrows():
        classes_per_road = []   
        for Union_column_name in Uninon_column_names:
            if road_data[Union_column_name] != -1:
                classes_per_road.append(road_data[Union_column_name])
            else:
                classes_per_road.append(-1)
        union_class_per_road.append(max(classes_per_road))
    GdfSimplified["Union"] = union_class_per_road
    return GdfSimplified


## -------------- Degree distribution functions -------------- ##

def add_degrees_to_gdf(gdf, 
                       G, 
                       id_column_source = None, 
                       id_column_target = None
                       ):
    """
    Adds 'in_degree' and 'out_degree' columns to a GeoDataFrame
    of road segments based on a directed NetworkX graph G.

    Parameters:
        gdf (GeoDataFrame): with LineString geometries.
        G (networkx.DiGraph): directed graph with node degrees.

    Returns:
        GeoDataFrame: with added columns 'in_degree_start', 'out_degree_start',
                      'in_degree_end', 'out_degree_end'
    """
    in_degrees_start = []
    out_degrees_start = []
    in_degrees_end = []
    out_degrees_end = []

    for _, row in gdf.iterrows():
        geom = row.geometry
        if not isinstance(geom, LineString):
            # Use NaN if geometry is not valid
            in_degrees_start.append(pd.NA)
            out_degrees_start.append(pd.NA)
            in_degrees_end.append(pd.NA)
            out_degrees_end.append(pd.NA)
            continue
        

        start_id = row[id_column_source] 
        end_id = row[id_column_target]


        # Get degrees or fallback to 0 if node is not in graph
        in_degrees_start.append(G.in_degree(start_id) if start_id in G else 0)
        out_degrees_start.append(G.out_degree(start_id) if start_id in G else 0)
        in_degrees_end.append(G.in_degree(end_id) if end_id in G else 0)
        out_degrees_end.append(G.out_degree(end_id) if end_id in G else 0)

    # If id_column_source, id_column_target are 
    # provided, ensure they are in the GeoDataFrame



    # Add the degree columns to the GeoDataFrame
    gdf = gdf.copy()
    gdf['in_degree_start'] = in_degrees_start
    gdf['out_degree_start'] = out_degrees_start
    gdf['in_degree_end'] = in_degrees_end
    gdf['out_degree_end'] = out_degrees_end

    return gdf







## ------------- GREGORIO ----------------- ##
def transform_gdf_greg(GdfSimplified,gdf_single_day):
    """
    Transform the GeoDataFrame GdfSimplified to match the structure of gdf_single_day.
    """
    gdf_greg = GdfSimplified.merge(gdf_single_day, on="geometry", suffixes=('_simplified', '_single_day'))
    gdf_greg.rename(columns={"poly_cid":"id"},inplace=True)
    gdf_greg.rename(columns={"poly_nF":"sourceId"},inplace=True)
    gdf_greg.rename(columns={"poly_nT":"targetId"},inplace=True)
    gdf_greg.rename(columns={"poly_length":"length"},inplace=True)
    class2speed = {0:4,1:21,2:42,3:92}
    gdf_greg["maxSpeed"] = gdf_greg["Union"].apply(lambda x: class2speed.get(x))
    return gdf_greg

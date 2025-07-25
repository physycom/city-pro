import contextily as ctx
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
import os
LIST_COLORS = ["blue", "orange", "green", "red", "purple", "pink", "brown", "gray"]


## ---------- HIERARCHICAL CLASSIFICATION PLOTTING FUNCTIONS ---------- ##

def Plot_road_network_colored_by_hierarchical_class_when_considered_union_over_all_days(GdfSimplified,path_save_plot,Classes):
    """
        Plot the road network colored by hierarchical class when considered union over all days.
        - GdfSimplified: GeoDataFrame with the roads and the Union column.
        - path_save_plot: Path to save the plot.
        This is the NetMob subnetwork visualization.
    """
    # Make sure GdfSimplified is filtered and projected correctly
    GdfSimplified = GdfSimplified.loc[GdfSimplified["Union"] != -1]
    GdfSimplified.to_crs(epsg=3857, inplace=True)

    # Define custom colors for each category
    custom_colors = {Classes[i]:LIST_COLORS[i] for i in range(len(Classes))}

    # Create a figure with fixed dimensions
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot the GdfSimplified with custom colors
    for category, color in custom_colors.items():
        print("plot class", category)
        GdfSimplified[GdfSimplified["Union"] == category].plot(
            ax=ax, color=color, label=category, alpha=1
        )

    # Get the bounds of the data in projected coordinates
    minx, miny, maxx, maxy = GdfSimplified.total_bounds

    # Set limits in the projected coordinate system
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)

    # Add some padding (10% on each side)
    padding_x = 0# (maxx - minx) * 0.1
    padding_y = 0#(maxy - miny) * 0.1
    ax.set_xlim(minx - padding_x, maxx + padding_x)
    ax.set_ylim(miny - padding_y, maxy + padding_y)

    # Add a legend
    ax.legend(loc="upper right")

    # Remove coordinate ticks
    ax.axis('off')

    # Add a black and white basemap
    ctx.add_basemap(ax, crs=GdfSimplified.crs.to_string(), source=ctx.providers.CartoDB.PositronNoLabels)

    # Add a scale bar
    fontprops = fm.FontProperties(size=12)
    scalebar = AnchoredSizeBar(ax.transData,
                            1000, '1 km', 'lower right', 
                            pad=0.1,
                            color='black',
                            frameon=False,
                            size_vertical=1,
                            fontproperties=fontprops)
    ax.add_artist(scalebar)

    # Save the figure
    plt.savefig(path_save_plot, 
                bbox_inches='tight', dpi=200)
    plt.show()


## ---------------- FUZZY CLASSIFICATION PLOTTING FUNCTIONS ---------------- ##

def Plot_road_network_colored_by_hierarchical_class_all_days(GdfSimplified, path_save_plot, Classes, Days):
    """
        Plot the road network colored by fuzzy class.
        - GdfSimplified: GeoDataFrame with the roads and the Union column.
        - path_save_plot: Path to save the plot.
        Separate plots for each day.
    """
    # Define custom colors for each category
    custom_colors = {Classes[i]:LIST_COLORS[i] for i in range(len(Classes))}    
    # Plot the GdfSimplified with custom colors
    count_days = 0
    for class_column_day in Classes:
        fig, ax = plt.subplots(figsize=(10, 10))
        for category, color in custom_colors.items():
            GdfSimplified[GdfSimplified[class_column_day] == category].plot(
                ax=ax, color=color, label=category, alpha=1
            )

        # Set the aspect ratio to make the plot square
        #ax.set_aspect('equal', adjustable='datalim')

        # Add a legend
        ax.legend(loc="upper right")

        # Remove coordinate ticks
        ax.axis('off')

        # Add a black and white basemap
        ctx.add_basemap(ax, crs=GdfSimplified.crs.to_string(), source=ctx.providers.CartoDB.PositronNoLabels)

        # Add a scale bar
        fontprops = fm.FontProperties(size=12)
        scalebar = AnchoredSizeBar(ax.transData,
                                1000, '1 km', 'lower right', 
                                pad=0.1,
                                color='black',
                                frameon=False,
                                size_vertical=1,
                                fontproperties=fontprops)

        ax.add_artist(scalebar)
        
        plt.savefig(os.path.join(path_save_plot,f"union_classes_{Days[count_days]}.png"),bbox_inches='tight',dpi = 200)
        plt.close()
        count_days+=1
        plt.show()    


## -------------- DYNAMIC PLOTTING FUNCTIONS -------------- ##

import folium
from folium.plugins import MarkerCluster
import geopandas as gpd
import branca.colormap as cm

def Plot_road_network_colored_by_hierarchical_class_folium(GdfSimplified, Classes):
    """
        Plot the road network colored by hierarchical class using Folium.
        - GdfSimplified: GeoDataFrame with the roads and the Union column.
        - Classes: List of classes to color the roads.
        This function creates an interactive map using Folium.
        - The map is centered on the mean coordinates of the roads.
        - Each road is colored according to its class.
    """
    # First, convert GdfSimplified to EPSG:4326 (WGS84) for folium
    GdfSimplified_4326 = GdfSimplified.copy()
    GdfSimplified_4326 = GdfSimplified_4326.to_crs(epsg=4326)

    # Get the center of the map
    center = [GdfSimplified_4326.geometry.centroid.y.mean(), GdfSimplified_4326.geometry.centroid.x.mean()]

    # Create a folium map
    m = folium.Map(location=center, zoom_start=13, tiles='CartoDB positron')

    # Define custom colors for the classes
    custom_colors = {Classes[i]:LIST_COLORS[i] for i in range(len(Classes))}    


    # Custom function to style the GeoJSON
    def style_function(feature):
        category = feature['properties']['Union']
        return {
            'fillColor': custom_colors.get(category, 'gray'),
            'color': custom_colors.get(category, 'gray'),
            'weight': 2,
            'fillOpacity': 0.7
        }

    # Add hover functionality
    def highlight_function(feature):
        return {
            'fillColor': custom_colors.get(feature['properties']['Union'], 'gray'),
            'color': 'black',
            'weight': 3,
            'fillOpacity': 0.9
        }

    # Add a GeoJson layer for each category
    for category, color in custom_colors.items():
        # Filter the GeoDataFrame for this category
        gdf_category = GdfSimplified_4326[GdfSimplified_4326['Union'] == category]
        
        if not gdf_category.empty:
            # Convert to GeoJSON
            geojson_data = gdf_category.to_json()
            
            # Add the GeoJSON to the map with popup information
            folium.GeoJson(
                geojson_data,
                name=f"Class {category}",
                style_function=lambda x, category=category: {
                    'fillColor': custom_colors[category],
                    'color': custom_colors[category],
                    'weight': 2,
                    'fillOpacity': 0.7
                },
                highlight_function=highlight_function,
                tooltip=folium.GeoJsonTooltip(
                    fields=['Union'],
                    aliases=['Class:'],
                    style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;")
                )
            ).add_to(m)

    # Add layer control
    folium.LayerControl().add_to(m)

    # Add a scale bar
    folium.plugins.MeasureControl(position='bottomleft', primary_length_unit='meters').add_to(m)

    # Save the map
    #m.save('/home/aamad/codice/city-pro/output/bologna_mdt_center/plots/interactive_union_classes_folium.html')

    # Display the map in the notebook
    return m
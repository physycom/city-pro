"""
    This module is used to plot the trajectories.
    It is in couple with the Trajectories.py module.
    It is thought in the usage of MDT data, but is extensible to other data.
    NOTE:
        The principal object is going to be traj_dataframe.
        Required Columns:
            - user_id: int: user id.
            - timestamp: int: timestamp.
            - lat: float: latitude.
            - lng: float: longitude.
"""

from shapely.geometry import LineString
import contextily as ctx
from ctx_providers import name_2_provider
import matplotlib.pyplot as plt
import geopandas as gpd
from matplotlib_scalebar.scalebar import ScaleBar
import numpy as np
import os
def Plot_traj_feature_colormap(user_trajs,
                              users,
                              column_user,
                              column_plot,
                              label_colormap,
                              PlotDir,
                              SaveName,
                              SaveFig = False,
                              provider_name = "BlackAndWhite",
                              colormap_name = "viridis",
                              crs="EPSG:4326",
                              crs_proj="EPSG:3857",
                              ):
    """
        @brief:
            - This function is used to plot the trajectories of the users selected.
            NOTE: It does that by iterating user by user, therefore the input must be a list of users.
        @param user_trajs: skmob.traj_dataframe:
            - dataframe with the trajectories.
        @param users: list or tuple:
            - list of users to be plotted.
        @param column_user: str:
            - column of the user.
        @param column_plot: str:
            - column to be plotted. -> Example: speed_kmh
        @param label_colormap: str:
            - label of the colormap. -> Example: speed km/h
        @param PlotDir: str:
            - directory to save the plot.
        @param SaveName: str:
            - name of the plot.
        @param SaveFig: bool:
            - save the figure or not.
        @param provider_name: str:
            - name of the provider. -> Example: "BlackAndWhite"
        @param colormap_name: str:
            - name of the colormap. -> Example: "viridis"
        @param crs: str:
            - coordinate reference system. -> Example: "EPSG:4326"
        @param crs_proj: str:
            - projected coordinate reference system. -> Example: "EPSG:3857"

    """
    assert isinstance(users,list) or isinstance(users,tuple) or isinstance(users,np.ndarray), "users must be a list or tuple."
    assert "lat" in user_trajs.columns, "lat must be a column in user_trajs."
    assert "lng" in user_trajs.columns, "lng must be a column in user_trajs."
    fig, ax = plt.subplots(figsize=(10, 10))
    # Set limit color map
    max_col = user_trajs[column_plot].max()
    min_col = user_trajs[column_plot].min()
    for user in users:
        # Plot user by user in the list
        user_traj = user_trajs.loc[user_trajs[column_user] == user] 
        gdf = gpd.GeoDataFrame(user_traj,
                               geometry=gpd.points_from_xy(user_traj.lng,user_traj.lat),
                               crs=crs)
        gdf.set_crs(crs, inplace=True)
        gdf = gdf.to_crs(crs_proj)
        # Plot segment by segment
        for i in range(len(gdf) - 1):
            segment = LineString([gdf.iloc[i].geometry, gdf.iloc[i + 1].geometry])
            col_val = gdf.iloc[i + 1][column_plot]
            ax.plot(*segment.xy,
                    color=plt.cm.viridis(col_val / max_col),
                    linewidth=2)

        # Set plot labels and title
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
    sm = plt.cm.ScalarMappable(cmap=colormap_name,
                               norm=plt.Normalize(vmin=min_col, vmax=max_col))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label(label_colormap)
    ctx.add_basemap(ax, crs=crs_proj,
                    source=name_2_provider[provider_name]
                    )
    scalebar = ScaleBar(1, location='lower left')
    ax.add_artist(scalebar)
    if SaveFig:
        plt.savefig(os.path.join(PlotDir,SaveName))



# Plot Distribution Time Sampling
def Plot_distribution_sampling_times(Trajdf,
                                     column_time_sampling,
                                     cut_sampling,
                                     ax,
                                     PlotDir,
                                     NameFigure,
                                     SaveFig = False):
    """
        @param Trajdf: DataFrame with the trajectories
            - user_id
            - timestamp
            - column_time_sampling: dt_sec -> time between two consecutive points
        Description:
            Plots the sampling rate of the trajectory we use in our analysis.
        Example:
            fig,ax = plt.subplots(1,1,figsize = (10,10))
            name_project = "bologna_mdt_center"
            base_name = "bologna_mdt"
            date = "2022-07-01"
            cut_sampling = 5
            column_time_sampling = "dt_sec"
            PlotDir = os.path.join(os.environ["WORKSPACE"],"plots",name_project,date)
            ax = Plot_distribution_sampling_times(Trajdf,
                                                    column_time_sampling,
                                                    cut_sampling,
                                                    ax,
                                                    PlotDir,
                                                    "SamplingRate.png",
                                                    SaveFig = False)
        @return ax

    """
    n,bins = np.histogram(Trajdf[column_time_sampling].loc[Trajdf[column_time_sampling]<cut_sampling],
                          bins = 100)
    logn = np.log(n)
    logbins = np.log(bins[1:])
    fit_exp = np.polyfit(bins[:-1],logn,1)
    fit_pl = np.polyfit(logbins,n,1)
    exp_ = np.exp(fit_exp[0]*bins[:-1] + fit_exp[1])
    pl_ = fit_pl[1]*bins[1:]**fit_pl[0]
    error_pl = np.sum((n - pl_)**2)
    error_exp = np.sum((n - exp_)**2)
    if error_pl < error_exp:
        sample = "{sample}"
        label = rf"$\langle t_{sample} \rangle: {-round(1/fit_exp[0],2)} s $"
#        ax.plot(bins[:-1],pl_,label = "")
    else:
        sample = "{sample}"
        label = rf"$\langle t_{sample} \rangle: {round(np.sum(bins[:-1]*n)/np.sum(n),2)} s $"
#        ax.plot(bins[1:],exp_,label = "")

    ax.scatter(bins[:-1],n,label = label)
    ax.set_xlabel("dt (s)")
    ax.set_ylabel("count")
    ax.set_xlim(0,cut_sampling)
    ax.set_yscale("log")
#    ax.text(bins[-20],n[0],label)
    if SaveFig:
        plt.savefig(os.path.join(PlotDir,NameFigure),dpi = 200)
        plt.close()
    return ax,label

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib_scalebar.scalebar import ScaleBar
import contextily as ctx

def create_trajectory_animation_gif(second_largest_user_data, roads_gdf, paths_gdf, trajectory_gdf, 
                                   speed_km_h_column, user_chosen, local_crs, 
                                   gif_filename="trajectory_animation.gif", fps=2):
    """
    Create an animated GIF showing trajectory and speed evolution over time.
    
    Parameters:
    -----------
    second_largest_user_data : DataFrame (Polars or pandas)
        User data with datetime and speed columns
    roads_gdf, paths_gdf, trajectory_gdf : GeoDataFrame
        Geographic data for plotting
    speed_km_h_column : str
        Column name for speed data
    user_chosen : str/int
        User identifier
    local_crs : str
        Local coordinate reference system
    gif_filename : str
        Output filename for the GIF
    fps : int
        Frames per second for the GIF
    """
    
    # Convert to pandas if it's a Polars DataFrame
    if hasattr(second_largest_user_data, 'to_pandas'):
        user_data_pd = second_largest_user_data.to_pandas()
    else:
        user_data_pd = second_largest_user_data
    
    # Pre-calculate data transformations to avoid repeated operations
    roads_gdf_local = roads_gdf.to_crs(local_crs)
    paths_gdf_wgs84 = paths_gdf.to_crs("EPSG:4326")
    
    # Set up the figure with proper spacing and equal heights
    fig = plt.figure(figsize=(16, 8))
    
    # Create gridspec for equal height subplots
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], hspace=0.3, wspace=0.3)
    ax_speed = fig.add_subplot(gs[0])
    ax_map = fig.add_subplot(gs[1])
    
    # Pre-configure ax_speed (speed plot) with fixed settings
    ax_speed.set_xticks(user_data_pd["datetime"][::8])
    ax_speed.set_xticklabels(
        user_data_pd["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")[::8], 
        rotation=90
    )
    ax_speed.set_xlabel("Time")
    ax_speed.set_ylabel("Speed km/h")
    ax_speed.set_ylim(0, 110)
    ax_speed.set_xlim(
        user_data_pd["datetime"].iloc[0], 
        user_data_pd["datetime"].iloc[-1]
    )
    ax_speed.grid(True, alpha=0.3)
    ax_speed.set_title(f'Speed Evolution - User {user_chosen}')
    
    # Pre-configure ax_map (map plot)
    ax_map.set_title('Trajectory Points and Computed Shortest Paths')
    ax_map.set_xlabel('Longitude')
    ax_map.set_ylabel('Latitude')
    
    # Plot static elements on map
    roads_gdf_local.plot(ax=ax_map, color='gray', linewidth=0.5, alpha=0.5)
    
    # Add basemap and scalebar (these won't change)
    ctx.add_basemap(ax_map, crs=local_crs, source=ctx.providers.OpenStreetMap.Mapnik)
    ax_map.set_aspect('equal')
    scalebar = ScaleBar(1, location='lower left')
    ax_map.add_artist(scalebar)
    
    def animate(frame):
        """Animation function for each frame."""
        t = frame + 1  # Start from 1 as in original code
        
        if t <= len(user_data_pd):
            # Clear previous dynamic elements
            ax_speed.clear()
            ax_map.clear()
            
            # Re-plot static map elements
            roads_gdf_local.plot(ax=ax_map, color='gray', linewidth=0.5, alpha=0.5)
            
            # Plot speed data up to current time
            ax_speed.plot(
                user_data_pd["datetime"][:t],
                user_data_pd[speed_km_h_column][:t], 
                label=f"User {user_chosen}",
                color='blue',
                linewidth=2
            )
            
            # Re-configure speed plot (since we cleared it)
            ax_speed.set_xticks(user_data_pd["datetime"][::8])
            ax_speed.set_xticklabels(
                user_data_pd["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")[::8], 
                rotation=90
            )
            ax_speed.set_xlabel("Time")
            ax_speed.set_ylabel("Speed km/h")
            ax_speed.set_ylim(0, 110)
            ax_speed.set_xlim(
                user_data_pd["datetime"].iloc[0], 
                user_data_pd["datetime"].iloc[-1]
            )
            ax_speed.grid(True, alpha=0.3)
            ax_speed.set_title(f'Speed Evolution - User {user_chosen}')
            ax_speed.legend()
            
            # Plot trajectory elements up to current time
            if t <= len(paths_gdf_wgs84):
                paths_gdf_wgs84[:t].to_crs(local_crs).plot(
                    ax=ax_map, color='green', linewidth=3, alpha=0.8
                )
            
            if t <= len(trajectory_gdf):
                # Plot trajectory points with speed-based coloring
                traj_subset = trajectory_gdf[:t].to_crs(local_crs)
                if speed_km_h_column in trajectory_gdf.columns:
                    traj_subset.plot(
                        ax=ax_map, 
                        c=trajectory_gdf[speed_km_h_column][:t],
                        cmap='viridis',
                        markersize=50, 
                        alpha=0.8,
                        marker='*',
                        vmin=0,
                        vmax=110
                    )
                else:
                    traj_subset.plot(
                        ax=ax_map, color='red', markersize=50, alpha=0.8, marker='*'
                    )
            
            # Re-configure map plot
            ax_map.set_title('Trajectory Points and Computed Shortest Paths')
            ax_map.set_xlabel('Longitude')
            ax_map.set_ylabel('Latitude')
            ctx.add_basemap(ax_map, crs=local_crs, source=ctx.providers.OpenStreetMap.Mapnik)
            ax_map.set_aspect('equal')
            
            # Add progress indicator
            progress = f"Frame {t}/{len(user_data_pd)}"
            fig.suptitle(progress, fontsize=14)
    
    # Create animation
    frames = range(len(user_data_pd))
    anim = animation.FuncAnimation(
        fig, animate, frames=frames, interval=1000//fps, repeat=True, blit=False
    )
    
    # Save as GIF
    print(f"Creating GIF with {len(frames)} frames...")
    anim.save(gif_filename, writer='pillow', fps=fps, dpi=100)
    print(f"GIF saved as {gif_filename}")
    
    plt.close(fig)
    return anim

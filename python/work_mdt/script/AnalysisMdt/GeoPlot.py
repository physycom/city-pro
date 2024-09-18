import numpy as np
import os
import imageio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
import geoplot.crs as gcrs
import geoplot as gplt

def BuildListStepsGivenDay(GpdClasses,StrDay,StrMetric):
    """
        This Functions Returns a Dict {StartInterval:{RoadIdx:TimePercorrence}} the video Of (Time Percorrence or Average Speed) of the day StrDay.
        NOTE: Assumes existence of StrMetric + StartInterval + "_" + str(Class) + "_" + StrDay columns in GpdClasses
    """
    # {StrMetric_Class_StrDay: [StartInterval]}
    Classes = []
    StartIntervals = []
    for Col in GpdClasses.columns:
        print(Col)
        if Col.startswith(StrMetric):
            StartIntervals.append(Col.split("_")[1])
            if "-" not in Col.split("_")[2]:
                Classes.append(Col.split("_")[2])
    Classes = np.unique(Classes)
    StartIntervals = np.unique(StartIntervals)
#    print("Classes: ",Classes)
#    print("StartIntervals: ",StartIntervals)
    #     
    Hour2Road2ListMetric = {time: {Road:[] for Road in GpdClasses["poly_lid"].to_numpy().astype(int)} for time in StartIntervals}
    Hour2Road2Metric = {time: {Road:0 for Road in GpdClasses["poly_lid"].to_numpy().astype(int)} for time in StartIntervals}
    print("Start to Aggregate Time percorrence or average Speed by Class ")
    for Class in Classes:
        for StartInterval in StartIntervals:
            Col = StrMetric + StartInterval + "_" + str(Class) + "_" + StrDay
            # NOTE: The roads that have no records for the road are -1 and so asking to filter with>0 makes the job
            RoadsWithColBigger0 = GpdClasses.loc[GpdClasses[Col]>0]   
            print("For Class ",Class," At Time: ",StartInterval," the available roads are: ",len(RoadsWithColBigger0))         
            for Road in RoadsWithColBigger0["poly_lid"].to_numpy().astype(int):
                # {TimeHour:{RoadIdx:[TimePercorrence]}
                if not np.isnan(GpdClasses.at[Road,Col]): 
                    Hour2Road2ListMetric[StartInterval][Road].append(GpdClasses.at[Road,Col])
                    if len(Hour2Road2ListMetric[StartInterval][Road])>1:
                        print("Road: ",Road," At Time: ",StartInterval," Adds Time Percorrence: ",GpdClasses.at[Road,Col],"Iteration Class: ",Class," Total number of Time Percorrence: ",len(Hour2Road2ListMetric[StartInterval][Road]))
#                    print("Road: ",Road," At Time: ",StartInterval," Adds Time Percorrence: ",GpdClasses.at[Road,Col],"Iteration Class: ",Class," Total number of Time Percorrence: ",len(Hour2Road2ListMetric[StartInterval][Road]))
    # Store the Average of the Metric for each Road
    for StartInterval in StartIntervals:
        for Road in GpdClasses["poly_lid"].to_numpy().astype(int):
            if not np.isnan(Road):
                if not np.isnan(Hour2Road2ListMetric[StartInterval][Road]) and len(Hour2Road2ListMetric[StartInterval][Road]) > 0:
                    Hour2Road2Metric[StartInterval][Road] = np.mean(Hour2Road2ListMetric[StartInterval][Road])

                else:
                    Hour2Road2Metric[StartInterval][Road] = 0
            else:
                print("Time Interval: ",StartInterval)
                print("Road is nan")
    # Substitute to the columns the average of the metric
    for StartInterval in StartIntervals:
        Col = StrMetric + StartInterval + "_" + StrDay
        GpdClasses[Col] = np.zeros(len(GpdClasses))
        for Road in GpdClasses["poly_lid"].to_numpy().astype(int):
            if not np.isnan(Road) and Road>0:
                GpdClasses.at[Road,Col] = Hour2Road2Metric[StartInterval][Road]
            else:
                print("Time Interval: ",StartInterval)
                print("Road is nan")
    return GpdClasses
# Evolution Time Percorrence and Average Speed Video Road.

# This function is thougth to plot daily evolution of traffic.
#NOTE: The video uses geoplot on the GeoJson of the city that must contains the columns with these info independently
def VideoEvolutionTimePercorrence(GpdClasses,StrMetric,StrDay,PlotDir):
    images = []
    OrderedCols2Hours = {0:"00:00:00",1:"00:15:00",2:"00:30:00",3:"00:45:00",4:"01:00:00",5:"01:15:00",6:"01:30:00",7:"01:45:00",8:"02:00:00",9:"02:15:00",
               10:"02:30:00",11:"02:45:00",12:"03:00:00",13:"03:15:00",14:"03:30:00",15:"03:45:00",16:"04:00:00",17:"04:15:00",18:"04:30:00",
               19:"04:45:00",20:"05:00:00",21:"05:15:00",22:"05:30:00",23:"05:45:00",24:"06:00:00",25:"06:15:00",26:"06:30:00",27:"06:45:00",
               28:"07:00:00",29:"07:15:00",30:"07:30:00",31:"07:45:00",32:"08:00:00",33:"08:15:00",34:"08:30:00",35:"08:45:00",36:"09:00:00",
               37:"09:15:00",38:"09:30:00",39:"09:45:00",40:"10:00:00",41:"10:15:00",42:"10:30:00",43:"10:45:00",44:"11:00:00",45:"11:15:00",
               46:"11:30:00",47:"11:45:00",48:"12:00:00",49:"12:15:00",50:"12:30:00",51:"12:45:00",52:"13:00:00",53:"13:15:00",54:"13:30:00",
               55:"13:45:00",56:"14:00:00",57:"14:15:00",58:"14:30:00",59:"14:45:00",60:"15:00:00",61:"15:15:00",62:"15:30:00",63:"15:45:00",
               64:"16:00:00",65:"16:15:00",66:"16:30:00",67:"16:45:00",68:"17:00:00",69:"17:15:00",70:"17:30:00",71:"17:45:00",72:"18:00:00",
               73:"18:15:00",74:"18:30:00",75:"18:45:00",76:"19:00:00",77:"19:15:00",78:"19:30:00",79:"19:45:00",80:"20:00:00",81:"20:15:00",
               82:"20:30:00",83:"20:45:00",84:"21:00:00",85:"21:15:00",86:"21:30:00",87:"21:45:00",88:"22:00:00",89:"22:15:00",90:"22:30:00",
               91:"22:45:00",92:"23:00:00",93:"23:15:00",94:"23:30:00",95:"23:45:00"}

    for IntTime in OrderedCols2Hours.keys():
        StartInterval = OrderedCols2Hours[IntTime]
        Col = StrMetric + StartInterval + "_" + StrDay
        if Col in GpdClasses.columns:
            filtered_gdf = GpdClasses.loc[GpdClasses[Col] > 0].dropna(subset=['geometry'])
            SavePath = os.path.join(PlotDir,Col + ".png")
            StepPlot(filtered_gdf,Col,GpdClasses[["geometry","poly_lid"]],SavePath)
            images.append(imageio.v2.imread(SavePath))
#    imageio.v2.mimsave(os.path.join(PlotDir,Col +".gif"), images, duration = 1)
    return 'movie.gif'


def StepPlot(filtered_gdf,Column,GpdClasses,SavePath,dpi = 200):
    filtered_gdf = filtered_gdf[filtered_gdf.geometry.is_valid]
    filtered_gdf = filtered_gdf.dropna(subset=['geometry'])
    filtered_gdf = filtered_gdf[filtered_gdf.geometry.apply(lambda geom: np.isfinite(geom.bounds).all())]
    if len(filtered_gdf) == 0:
        pass
    else:
        fig = plt.figure(figsize=(10, 8), dpi=dpi)
        gs = GridSpec(1, 2, width_ratios=[20, 1])
        ax = fig.add_subplot(gs[0,0],projection=gcrs.AlbersEqualArea())        
        GpdClasses.to_crs({'init': 'epsg:4326'})
        GpdClasses.plot(ax = ax,color = "grey",alpha = 0.3)
        gplt.sankey(
            filtered_gdf,
            scale= Column,
            limits=(0.1, 2),
            hue= Column,
            cmap = 'inferno',
            ax = ax
        )
        sm = plt.cm.ScalarMappable(cmap='inferno', norm=LogNorm(vmin=min(filtered_gdf[Column].to_numpy()), vmax=max(filtered_gdf[Column].to_numpy())))
        cax = fig.add_subplot(gs[0,1])
        # Create a ScalarMappable object for the colorbar
        # Empty array for the data range
        sm.set_array([])
        # Add the colorbar to the figure
        cbar = fig.colorbar(sm, cax=cax)
        # Set the colorbar label
        StartInterval = Column.split("_")[1]
        Colorbar = Column.split("_")[0]
        StrDay = Column.split("_")[2]
        cbar.set_label(Colorbar)
        ax.set_title(StartInterval)
        ax.annotate(f'Source: Tim Dataset, {StrDay}',
           xy=(0.1, .08), xycoords='figure fraction',
           horizontalalignment='left', verticalalignment='top',
           fontsize=10, color='#555555')
#        filtered_gdf.plot(column = Column,
#                     cmap= "inferno",
#                    scheme = Scheme,
#                    legend = True,
#                    legend_kwds = {'loc': 'upper right'},
#                    figsize = (8,5),
#                    )
    plt.savefig(SavePath,dpi = dpi)
    plt.close()
    return SavePath  

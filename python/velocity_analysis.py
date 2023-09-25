import folium
import geopandas as gpd
import pandas as pd
import os
import numpy as np
import json
import argparse
from itertools import combinations
import matplotlib.pyplot as plt 
import datetime
from collections import defaultdict
from shapely.geometry import Point,Polygon
def ifnotmkdir(dir_):
    if not os.path.exists(dir_):
        os.mkdir(dir_)
    return dir_
    
class analyzer_simulation:
    def __init__(self,config_):
        ##### TIME CONFIG ######
        if 'dt' in config_:
            self.dt = config_['dt']
        else:
            self.dt = 900,
        self.day_in_sec = 86400, 
        self.iterations = self.day_in_sec/self.dt                    
        self.time_format = "%Y-%m-%d %H:%M:%S"  
        if 'start_date' in config_:
            self.start_date = datetime.datetime.strptime(config_['start_date'],self.time_format)
        else:
            self.start_date = "2022-05-12 00:00:00"
        if 'end_date' in config_:
            self.end_date = datetime.datetime.strptime(config_['end_date'],self.time_format)
        else:
            self.end_date = "2022-05-13 00:00:00"
        if 'day_analysis' in config_:
            self.day_analysis = config_["day_analysis"]
        else:
            self.day_analysis = "2022-05-12"
        
        if "bin_time" in config_:
            self.bin_time = config_['bin_time']
        else:
            self.bin_time = 15
        self.iterations = self.day_in_sec/self.dt
        self.time_format = "%Y-%m-%d %H:%M:%S"
        self.start_date = datetime.datetime.strptime(config_['start_date'],self.time_format)
        self.end_date = datetime.datetime.strptime(config_['end_date'],self.time_format)      
        ###### INPUT-OUTPUT DIRS #######
        if "prefix_name" in config_:
            self.prefix_name = config_["prefix_name"]
        else:
            self.prefix_name = "bologna_mdt_2022-05-12_2022-05-12"
        if 'cartout_basename' in config_:
            self.cartout_basename = config_['cartout_basename']
        else:
            self.cartout_basename = "/home/aamad/codice/city-pro/output/bologna_mdt_detailed/" 
        if 'working_dir' in config_:
            self.working_dir = config_['working_dir']
            ifnotmkdir(self.working_dir)
        else:
            self.working_dir = "/home/aamad/codice/city-pro/output/bologna_mdt_detailed/2022-05-12"
            ifnotmkdir(self.working_dir)
        if "plot_dir" in config_:
            self.plot_dir = config_['plot_dir']
            ifnotmkdir(self.plot_dir)
        else:
            self.plot_dir = "/home/aamad/codice/city-pro/output/bologna_mdt_detailed/2022-05-12/plot"
            ifnotmkdir(self.plot_dir)
        ##### VELOCITY CLASS INITIALIZATION ######
        if "b_fcm" in config_ and config_['b_fcm']==True:
            self._get_dffcm()
            self.class_ = range(len(self.df_fcm['class'].unique())-1)
            self.b_fcm_centers = True
        else:
            self.b_fcm_centers = False
        ### STATS ####
        if "b_stats" in config_ and config_['b_stats']==True:
            self._get_stats()
            self.b_stats = True
        else:
            self.b_stats = False
        ### PRESENCES ####
        if "b_presence" in config_ and config_['b_presence']==True:
            self._get_presence()
            self.b_presence = True
        else:
            self.b_presence = False
        ### TIMED FLUXES ###
        if "b_tf" in config_ and config_['b_tf']==True:
            self._get_tfd()
            self.b_tf = True
        else:
            self.b_tf = False
        #### GEOJSON ######
        if 'file_geojson' in config_ and 'b_geojson' in config_ and config_['b_geojson']==True: 
            self.file_geojson = config_['file_geojson']
            self._get_geojson()
        else:
            self.file_geojson = "/home/aamad/codice/city-pro/bologna-provincia.geojson"
        self.list_all_files = []
        # SUBNETS        
        self.df_intersection_velsub = pd.read_csv(os.path.join(self.cartout_basename,self.prefix_name + self.save_label[0] + "velocity_subnet.csv"),";")
        self.df_complete_intersection_traj = pd.read_csv(os.path.join(self.cartout_basename,self.prefix_name + self.save_label[0] + "dati_regione.csv"),";")
        self.bounding_box = [(44.463121,11.287085),(44.518165,11.287085),(44.518165,11.367472),(44.463121,11.367472)]
        self.class_int2str = defaultdict()
        self.dict_complete_complement = defaultdict()
        self.dict_complementvelsub = defaultdict()   
        self.subname2file = defaultdict() 
        
        self._initialize_dict_complementvelsub()
        self.initialize_subname2file()
    def _get_geojson(self):
        try:
            self.geoj = gpd.read_file(self.file_geojson,driver = 'GeoJSON')
            self.b_geoj = True
        except:
            raise Exception('Error file not found: {}'.format(self.file_geojson))
                
    def _get_dffcm(self):
        try:        
            self.df_fcm = pd.read_csv(os.path.join(self.cartout_basename,self.prefix_name + "_fcm.csv"),";")
            self.b_df_fcm = True
        except:
            raise Exception('Error file not found: {}'.format(os.path.join(self.cartout_basename,self.prefix_name + "_fcm.csv")))

    def _get_presence(self):
        try:
            self.df_presence = pd.read_csv(os.path.join(self.cartout_basename,self.prefix_name + "_presence.csv"),";")
            self.b_df_presence = True
        except:
            raise Exception('Error file not found: {}'.format(os.path.join(self.cartout_basename,self.prefix_name + "_presence.csv")))
     
    def _get_stats(self):
        try:
            self.df_stats = pd.read_csv(os.path.join(self.cartout_basename,self.prefix_name + "_stats.csv"),";")
            self.b_df_stats = True
        except:
            raise Exception('Error file not found: {}'.format(os.path.join(self.cartout_basename,self.prefix_name + "_stats.csv")))

    def _get_tfd(self):
        try:
            self.tfd = pd.read_csv(os.path.join(self.cartout_basename,self.prefix_name + "_timed_fluxes.csv"),";")
            self.b_tfd = True
        except:
            raise Exception('Error file not found: {}'.format(os.path.join(self.cartout_basename,self.prefix_name + "_timed_fluxes.csv")))
        
    def _class_int2str(self):
        '''
        Creates dict_vel: {0:vel0,1:vel1,...} vel0<vel1<...
                dict_class_int2str: {0:i slowest,...,i quickest}
        '''
        self.dict_vel = defaultdict()
        for f,fc in self.df_fcm.groupby('class'):
            if f!=10:
                vel = np.mean(fc['av_speed'].to_numpy())
                self.dict_vel[f] = vel
        self.dict_vel = dict(sorted(self.dict_vel.items(),key = lambda item:item[1]))
        self.class_int2str = defaultdict(dict)
        self.number_classes = len(self.df_fcm.groupby('class')) - 1
        for i in range(self.number_classes):
            if i<self.number_classes/2:
                self.class_int2str[list(self.dict_vel.keys())[i]] = '{} slowest'.format(i+1)
            elif i<self.number_classes==2:
                self.class_int2str[list(self.dict_vel.keys())[i]] = 'middle velocity class'
            else:
                self.class_int2str[list(self.dict_vel.keys())[i]] = '{} quickest'.format(self.number_classes - i)             

    def _initialize_dict_complementvelsub(self):
        
        """
        dict_complement: {0 slowest: df_complement['start_bin','end_bin','poly_id','number_people_poly','total_number_people','av_speed','time_percorrence']}
        """
        for k in self.class_int2str.keys():
            try:
                self.dict_complementvelsub[self.class_int2str[k]] = pd.read_csv(os.path.join(self.cartout_basename,self.prefix_name + self.save_label[1] + str(k) + "_velocity_subnet.csv"),";")
                self.dict_complete_complement[self.class_int2str[k]] = pd.read_csv(os.path.join(self.cartout_basename,self.prefix_name + self.save_label[1] + str(k) + "_dati_regione.csv"),";")
            except:
                pass
    def plot_time_number_people_complete_intersection(self):
        mask = [True if x!=-1 else False for x in self.df_intersection_velsub['av_speed'].to_numpy()]
        tmp_vel = self.df_intersection_velsub.loc[mask]
        x = []
        y = []
        df_xy = tmp_vel.dropna(how = 'any')
        for t,dft in df_xy.groupby('start_bin'):
            x.append(np.mean(dft['total_number_people'].to_numpy()))    
            y.append(datetime.datetime.strftime(datetime.datetime.fromtimestamp(t),self.time_format))
        fig,ax = plt.subplots(1,1,figsize = (10,8))
        plt.scatter(y,x)
        plt.xticks(rotation = 90)
        plt.xlabel('time')
        plt.ylabel('number of people moving')
        plt.savefig(os.path.join(self.plot_dir,'total_count_complete_intersection_{}.png'.format(self.day_analysis)),dpi = 200)
        plt.show()        

    def plot_time_number_people(self):
        char_diagram = []
        time = []
        for h,_ in self.df_timed_fluxes.groupby('time'):
            char_diagram.append(sum(self.df_complete_intersection_traj['total_fluxes'].to_numpy()))
            time.append(h)        
        fig,ax = plt.subplots(1,1,figsize = (10,8))
        plt.scatter(time,char_diagram)
        plt.xticks(rotation = 90)
        plt.xlabel('time')
        plt.ylabel('number of people moving')
        plt.savefig(os.path.join(self.plot_dir,'total_count_{}.png'.format(self.day_analysis)),dpi = 200)
        plt.show()        
        
        return True
        
    def fondamental_diagram_complete_intersection(self):
        fig,ax = plt.subplots(1,1,figsize = (10,10))
        mask = [True if x!=-1 else False for x in self.df_intersection_velsub['av_speed'].to_numpy()]
        tmp_vel = self.df_intersection_velsub.loc[mask]
        x = []
        y = []
        df_xy = tmp_vel.dropna(how = 'any')
        for t,dft in df_xy.groupby('start_bin'):
            x.append(np.mean(dft['total_number_people'].to_numpy()))#*np.mean(dft['av_speed'].to_numpy()))    
            y.append(np.mean(dft['av_speed'].to_numpy()))        
        u = [x[i+1]-x[i] for i in range(len(x)-1)]
        v = [y[i+1]-y[i] for i in range(len(y)-1)]
        u.append(x[len(x)-1] -x[0])
        v.append(y[len(y)-1] -y[0])
        plt.scatter(x,y)
        plt.quiver(x,y,u,v,angles='xy', scale_units='xy', scale=1,width = 0.0025)
        plt.xlabel('number people')
        plt.ylabel('velocity')
        plt.title('subnetwork complete intersection')
        self.pth_complete_intersection = ifnotmkdir(os.path.join(self.cartout_basename,"complete_intersection"))
        plt.savefig(os.path.join(self.pth_complete_intersection,'{}flux_in_DV_complete_intersection.png'.format(self.day_analysis)),dpi = 200)
#        plt.savefig(os.path.join(pth,'{}_trajectory_in_DV_complete_intersection.png'.format(self.day_analysis)),dpi = 200)
        plt.show()
        return x,y
        
    def velocity_density(self):
        fig,ax = plt.subplots(1,1,figsize = (10,10))
        mask = [True if x!=-1 else False for x in self.df_intersection_velsub['av_speed'].to_numpy()]
        tmp_vel = self.df_intersection_velsub.loc[mask]
        x = pd.DataFrame(tmp_vel).dropna(how = 'any')['total_number_people'].to_numpy()
        y = pd.DataFrame(tmp_vel).dropna(how = 'any')['av_speed'].to_numpy()
        
        # 2a) DENSITà VELOCITà (CLASSE LENTA)
        _,binsx = np.histogram(x,range = (min(x),max(x)))
        _,_ = np.histogram(y,range = (min(y),max(y)))
        y_avg = np.zeros(len(binsx))
        y_dev = np.zeros(len(binsx))
        for dx in range(len(binsx)-1):
            idx_ = np.array([True if xi>=binsx[dx] and xi<=binsx[dx+1] else False  for xi in x])
            y_avg[dx] += np.mean(y[idx_])
            y_dev[dx] = np.std(y[idx_])
        print('mean:\t',y_avg[:-1],'\nstd-dev:\t',y_dev[:-1],'\ndev/mean:\t',y_dev[:-1]/y_avg[:-1])
        fig,ax = plt.subplots(1,1,figsize = (10,10))
        plt.plot(binsx[:-1],y_avg[:-1])
        plt.plot(binsx[:-1],y_avg[:-1]+y_dev[:-1])
        plt.plot(binsx[:-1],y_avg[:-1]-y_dev[:-1])
        plt.xlabel('number people')
        plt.ylabel('velocity')
        plt.title("subnetwork complete intersection")
        plt.legend(['mean','mean+std','mean-std'])
        self.pth_complete_intersection = ifnotmkdir(os.path.join(self.cartout_basename,"complete_intersection"))
        plt.savefig(os.path.join(self.pth_complete_intersection,'DV_{}.png'.format(self.day_analysis)),dpi = 200)
        plt.show()
        
    def compute_partition_traj_velocity_xsubnet(self):
        
        return True
    def qualcosa(self):
        for k in self.dict_complete_complement.keys():
            df_traj = self.dict_complete_complement[self.dict_complete_complement[k]]                                     
            fig,ax = plt.subplots(1,1,figsize = (10,10))
            mask = [True if x!=-1 else False for x in df_traj['av_speed'].to_numpy()]
            tmp_vel = df_traj.loc[mask]
            x = []
            y = []
            df_xy = tmp_vel.dropna(how = 'any')
            for t,dft in df_xy.groupby('start_bin'):
                x.append(np.mean(dft['total_number_people'].to_numpy()))    
                y.append(np.mean(dft['av_speed'].to_numpy()))        
        return True
    
    def geoj2gdf(self):
        '''
        For each subnet I need a name and the set of roads.
        
        '''
        self._get_geojson()
        for subname in self.subname2file.keys():
            try:
                file_complement1 = open(self.subname2file[subname],'r')        
                list_complement1 = file_complement1.read().split(" ")   
                mask = [True if str(x) in list_complement1 else False for x in self.geoj['poly_lid']]    
                gdf = self.geoj.loc[mask] 
                masknot = [False if str(x) in list_complement1 else True for x in self.geoj['poly_lid']]    
                gdfnot =  self.geoj.loc[masknot]
                self.plot(gdf,subname,self.plot_dir,gdfnot) 
            except:
                print(self.subname2file[subname],' not found')
        return gdf,gdfnot

    def plot(self,gdf,string_,working_dir,gdfnot):
        network = gdf.to_crs({'init': 'epsg:4326'})
        networknot = gdfnot.to_crs({'init': 'epsg:4326'})    
        fig,ax = plt.subplots(1,1,figsize = (20,15))
        plt.title(string_)
        try:
            networknot.plot(ax = ax,color = "green")    
            network.plot(ax = ax,color = "blue")
            plt.savefig(os.path.join(working_dir,string_ +'.png'),dpi = 200)
            plt.show()
        except ValueError:
            print('direct ',len(gdf))
            print('negated ',len(gdfnot))
            print('empty intersection')
        return fig,ax    

    def initialize_subname2file(self):
        for i in range(self.number_classes):
#            self.subname2file["subnet_{}".format(i)] = os.path.join(self.cartout_basename,self.prefix_name+"{}".format(i)+".csv")
            self.subname2file["complete_complement_{}".format(i)] = os.path.join(self.cartout_basename,self.prefix_name+"{}".format(i)+"_complete_complement.csv")
        self.subname2file["complete_intersection"] = os.path.join(self.cartout_basename,self.prefix_name+"_complete_intersection.csv")

    def plot_velocity_all_classes(self):
        y_bins = 750
        fig,ax = plt.subplots(1,1,figsize= (15,12))
        legend = []
        x = np.arange(12)*5
        y = np.arange(21)*y_bins
        for cl,df in self.df_fcm.groupby('class'):

        #    fig,ax = plt.subplots(1,1,figsize= (15,12))
            if cl!=10:
                plt.hist(df['av_speed'].to_numpy(),bins = 50,range = [0,50])
                av_speed = np.mean(df['av_speed'].to_numpy())
                legend.append(self.class_int2str[cl] + ' vel: ' + str(round(av_speed,3)) +' m/s')
        ax.set_xticks(x)
        ax.set_yticks(y)

        ax.set_xlabel('average speed (m/s)')
        ax.set_ylabel('Count')
        legend_ = plt.legend(legend)
        frame = legend_.get_frame()
        frame.set_facecolor('white')
        plt.savefig(os.path.join(self.plot_dir,'average_speed_4_classes_1.png'),dpi = 200)
        plt.show()

        fig,ax = plt.subplots(1,1,figsize= (15,12))
        legend = []
        x = np.arange(12)*5
        y = np.arange(21)*y_bins
        aggregated = []
        for cl,df in self.df_fcm.groupby('class'):
            if cl!=10:
                aggregated.extend(df['av_speed'].to_numpy())
        plt.hist(aggregated,bins = 50,range = [0,50])
        ax.set_xticks(x)
        ax.set_yticks(y)

        ax.set_xlabel('average speed (m/s)')
        ax.set_ylabel('Count')
        plt.savefig(os.path.join(self.plot_dir,'av_speed_aggregated.png'),dpi = 200)
        plt.show()
        mask = [True if x in [0,2,3,4] else False for x in self.df_fcm['class']]
        np.mean(self.df_fcm.loc[mask ]['av_speed'])


# BOUNDING BOX ANALYSIS
# def stats_boundingbox_class(self):
        """
        Informs how many points per class are inside some bounding box.
        Essentially built to understand if the algorithm is making wrong 
        assignment. That is: the quickest class cannot be found
        in a statistically relevant proportion in the city center.
        """
        for cl,dfc in self.df_fcm.groupby('class'):
            if cl != 10:
                mask_id = [True if x in dfc['id_act'].to_numpy() else False for x in self.df_stats['id_act'].to_numpy()]
                tmp = self.df_stats.loc[mask_id]
                self.class_int2str[cl]
                self.spatial_filtering_per_country(tmp)
                
        
    def info_plot(self,frontin,tailin,points_in_box,points_in_box_fcm,tmp,cl):
        self.pth_output = ifnotmkdir(os.path.join(self.cartout_basename,'output_files'))
        with open(os.path.join(self.pth_output,'statistics_class.txt'),'w') as f:
            f.write(self.class_int2str[cl],' number of front points: {0}, tail points: {1} in the bounding box:\t'.format(len(frontin),len(tailin)))
            f.write(self.class_int2str[cl],' number of points in box:\t',len(points_in_box))
            f.write(self.class_int2str[cl],' number of points in box fcm:\t',len(points_in_box_fcm))
            f.write(self.class_int2str[cl],' lenght of stats in class 1:\t',len(tmp))
            f.write(self.class_int2str[cl],' fraction of identification in bounding box:\t',len(points_in_box)/len(tmp))
            f.write(self.class_int2str[cl],' average velocity in bounding box:\t',np.mean(points_in_box_fcm['av_speed'].to_numpy()))
            f.close()
        fig,ax = plt.subplots(1,1,figsize = (10,8))
        n,bins = np.histogram(points_in_box_fcm['av_speed'].to_numpy())
        plt.scatter(bins[:-1],n)
        plt.xlabel('average velocity (m/s)')
        plt.ylabel('count')
        plt.title(self.class_int2str[cl])
        plt.savefig(os.path.join(self.cartout_basename,'velocity_distribution_{}_in_bb_center.png'.format(self.class_int2str[cl])),dpi = 200)
        fig,ax = plt.subplots(1,1,figsize = (10,8))
        n,bins = np.histogram(points_in_box_fcm['lenght'].to_numpy())
        plt.scatter(bins[:-1],n)
        plt.xlabel('lenght (m)')
        plt.ylabel('count')
        plt.title(self.class_int2str[cl])
        plt.savefig(os.path.join(self.cartout_basename,'distance_distribution_{}_in_bb_center.png'.format(self.class_int2str[cl])),dpi = 200)
        fig,ax = plt.subplots(1,1,figsize = (10,8))
        n,bins = np.histogram(points_in_box_fcm['time'].to_numpy())
        plt.scatter(bins[:-1],n)
        plt.xlabel('time (s)')
        plt.ylabel('count')
        plt.title(self.class_int2str[cl])
        plt.savefig(os.path.join(self.cartout_basename,'time_distribution_{}_in_bb_center.png'.format(self.class_int2str[cl])),dpi = 200)

    def plot_conditional3D(points_in_box,key1,key2):


        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        X = points_in_box[key1].to_numpy() 
        Y = points_in_box[key2].to_numpy()
        Z,xedges,yedges = np.histogram2d(X,Y)
        Z = Z.T
        x = np.array(xedges)[:-1]
        y = np.array(yedges)[:-1]
        xmax = max(X)
        xmin = min(X)
        ymax = max(Y)
        ymin = min(Y)
        zmax = np.max(Z)
        ax.bar(x, y, zs=Z, zdir='y', alpha=0.8)
        ax.set_xlabel(key1)
        ax.set_ylabel(key2)
        ax.set_zlabel('count')
        # On the y-axis let's only label the discrete values that we have data for.
        plt.show()    


    def spatial_filtering_per_country(self,tmp):
        ''' 
        Filters stats file for each class (tmp):
            Select those points that are inside the bounding box
        Produces:
            points_in_box -> used in countor plot
        '''
        l=list(zip(tmp['front_lat'],tmp['front_lon']))
        l1=list(zip(tmp['tail_lat'],tmp['tail_lon']))
        pointsfront = [Point(ll) for ll in l]
        pointstail = [Point(ll) for ll in l1]
        polygon= Polygon(self.bounding_box)
        mask=[point.within(polygon) for point in pointsfront]
        mask1=[point.within(polygon) for point in pointstail]
        frontin=tmp.loc[mask]['id_act'] 
        tailin=tmp.loc[mask1]['id_act']
        maskfcm = [True if x in frontin.to_numpy() or x in tailin.to_numpy() else False for x in self.df_fcm['id_act'].to_numpy()]
        maskstats = [True if x in frontin.to_numpy() or x in tailin.to_numpy() else False for x in tmp['id_act'].to_numpy()]
        points_in_box = tmp.loc[maskstats]
        coords_in_box = list(zip(points_in_box['front_lat'],points_in_box['front_lon']))
        coords_in_box1 = list(zip(points_in_box['tail_lat'],points_in_box['tail_lon']))
        points_in_box_fcm = self.df_fcm.loc[maskfcm]
        self.info_plot(self,frontin,tailin,points_in_box,points_in_box_fcm,tmp)
        self.plot_conditional3D(points_in_box,'time','av_speed')
        self.plot_conditional3D(points_in_box,'lenght','av_speed')
        
        return points_in_box,coords_in_box,coords_in_box1
    def plot_points_bb(self):
        '''
            Creates an html map where each point is an ending or start point of a trajectory found
            inside the bounding box.
        '''
        for cl in self.class_int2str.keys():
            list_act = self.df_fcm.groupby('class').get_group(cl)['id_act'].to_numpy()
            mask = [True if x in list_act else False for x in self.df_stats['id_act'].to_numpy()]
            tmp = self.df_stats.loc[mask]
            points_in_box,coords_in_box,coords_in_box1 = self.spatial_filtering_per_country(tmp)
            center = np.mean(np.asarray(coords_in_box),axis = 0)
            map_ = folium.Map(location=center, zoom_start=9,control_scale=True)
            c = 0
            for coord in np.asarray(coords_in_box)[:400]:
                folium.Marker([coord[0],coord[1]], popup=str(points_in_box['av_speed'].to_numpy()[c]) + ' (m/s)  ' + str(points_in_box['id_act'].to_numpy()[c]) + ' ' + str(points_in_box['lenght'].to_numpy()[c]) + ' m').add_to(map_)
                folium.Marker([coords_in_box1[c][0],coords_in_box1[c][1]], popup=str(points_in_box['av_speed'].to_numpy()[c]) + ' (m/s)  ' + str(points_in_box['id_act'].to_numpy()[c]) + ' ' + str(points_in_box['lenght'].to_numpy()[c]) + ' m').add_to(map_)
                c+=1
            self.pth_folium_maps = ifnotmkdir(os.path.join(self.cartout_basename,'folium_maps'))
            map_.save(os.path.join(self.pth_folium_maps,'map-{}-insidebox_start_end.html'.format(self.class_int2str[cl]))) 
    def plot_number_peopleintime(self):
        self._get_tfd()
        char_diagram = []
        time = []
        for h,df in self.tfd.groupby('time'):
            char_diagram.append(sum(df['total_fluxes'].to_numpy()))
            time.append(h)
            fig,ax = plt.subplots(1,1,figsize = (10,8))
            plt.scatter(time,char_diagram)
            plt.xticks(rotation = 90)
            plt.xlabel('time')
            plt.ylabel('number of people moving')
            plt.savefig(os.path.join(self.plot_dir,'total_count_{}.png'.format(self.start_date)),dpi = 200)
            plt.show()
if __name__ == '__main__':
    combs = combinations("01234",2)    
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='directory configuaration file', required=True)
    args = parser.parse_args()
    with open(args.config,'r') as f:
         jsonfile = f.read()
    config_ = json.loads(jsonfile)
    anal = analyzer_simulation(config_)
    
    if config_["plot_complete_intersection_map"] or config_['complete_analysis']:
        anal.plot_time_number_people_complete_intersection()
        anal.fondamental_diagram_complete_intersection()
    if config_['fcm_analysis'] or config_['complete_analysis']:
        anal._get_dffcm()
        anal._class_int2str()
        
        anal.velocity_density()
        anal.geoj2gdf()
        # ANALYSIS
        anal.plot_velocity_all_classes()
    # IMPORTING FILE OF INTEREST
    if config_["presence_analysis"] or config_['complete_analysis']:
        
    if config_["stats_analysis"] or config_['complete_analysis'] and not config_['fcm_analysis']:
        anal._get_stats()
        anal._get_dffcm()
    elif config_["stats_analysis"] or config_['complete_analysis'] and config_['fcm_analysis']:
        anal._get_stats()
    else:
        pass
    if config_['control_box']:
        anal.plot_points_bb()
    if config_['stats_nounding_box']:
        anal.stats_boundingbox_class()
    if config_['timed_fluxes']:
        anal.plot_number_peopleintime()
        
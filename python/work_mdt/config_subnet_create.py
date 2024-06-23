'''
README
This script is used to create config_subnet for each of the directories in ../work_geo/date/ date = list of directories dates []
Creates: basg script to launch all the config_subnet: ./work_geo/bologna_mdt_detailed/all_subnets.sh

After this is run, launch:
./work_geo/bologna_mdt_detailed/all_subnets.sh
that will launch all the config_subnet.py
python3 ./python/fluxes_subnet.py -c ./work_geo/bologna_mdt_detailed/date/config_bologna.json

THE LISTS  OF IMPORTANCE:
['2022-01-31', '2022-05-12', '2022-07-01',  '2022-12-30', '2022-12-31', '2023-01-01','2023-01-11','2023-03-18'] -> dates that are  analyzed
["0","1","2","3"] -> subnets that are analyzed
They must be consistent with:
- directories with subenet files
- num_tm: in ./work_geo/date/config_bologna.json 
'''
import json
import os
import pandas as pd
import numpy as np
from collections import defaultdict
def get_path_files():
    local_dir = os.getcwd()
    base_dir = os.path.dirname(local_dir)
    linked_dir = os.path.join(base_dir,'/home/aamad/codice/city-pro/')
    real_path = os.path.realpath(linked_dir)
    list_dirs = os.listdir(os.path.join(real_path,'work_geo','bologna_mdt_detailed'))
    return list_dirs,linked_dir
def ifnotmkdir(dir_):
    if not os.path.exists(dir_):
        os.mkdir(dir_)
    return dir_

def get_classes(config_,complete_name):
    fcm_file = os.path.join(config_['working_dir'],complete_name +"_fcm.csv")
    df_fcm = pd.read_csv(fcm_file,';')
    class2color = {}
    colors = ['red','blue','yellow','orange','purple','pink','brown','black','grey','green']
    dict_vel = {}
    for f,fc in df_fcm.groupby('class'):
        if f!=10 and f!=11:
            vel = np.mean(fc['av_speed'].to_numpy())
            if vel<50:
                dict_vel[f] = vel
                print('velocity of class {0} is: {1} m/s'.format(f,vel))
            else:
                print('velocity of class {0} is: {1} m/s'.format(f,vel))
    dict_vel = dict(sorted(dict_vel.items(),key = lambda item:item[1]))
    number_classes = len(dict_vel.keys())
    dict_name = defaultdict()
    for i in range(number_classes):
        if i<number_classes/2:
            dict_name[list(dict_vel.keys())[i]] = '{} slowest'.format(i+1)
        elif i<number_classes==2:
            dict_name[list(dict_vel.keys())[i]] = 'middle velocity class'
        else:
            dict_name[list(dict_vel.keys())[i]] = '{} quickest'.format(number_classes - i)                     
        class2color[list(dict_vel.keys())[i]] = colors[i]        
    return class2color,dict_name
    
if __name__=='__main__':
    # INITIALIZE BASH SCRIPT
    bash_script = "#!/bin/bash\n"
    # AGGREGATED_LISTS
    config_aggregated = {}
    config_aggregated['fundamental_diagram_dirs'] = []

    # SETTING LOCAL PATHS
    list_dirs,base_path = get_path_files()
    config = {}
    for dir_ in list_dirs:
        if dir_ in ['2022-08-05','2022-01-31', '2022-05-12', '2022-07-01','2022-11-11','2022-12-30','2022-12-31','2023-01-01','2023-01-11','2023-03-18']: #  
            config_dir = os.path.join(base_path,'work_geo','bologna_mdt_detailed',dir_,'plot_subnet.json')
            # WORKING DIR
            config['working_dir'] = os.path.join(base_path,'output','bologna_mdt_detailed',dir_) # 
            prefix_name = 'bologna_mdt'
            ifnotmkdir(config['working_dir'])
            # PLOTTING DIR
            config["plot_dir"] = os.path.join(config['working_dir'],'plot')
            # WHERE TO SAVE SUBNET FILES
            config["cartout_basename"] = os.path.join(config['working_dir'])
            # GEOJSON FILE
            config['file_geojson'] = os.path.join(base_path,"bologna-provincia.geojson")
            # WEIGHTS FILE
            config['weights_dir'] = os.path.join(base_path,'output','bologna_mdt_detailed',"weights",prefix_name+"_"+ dir_+"_"+dir_+'.fluxes.sub')
            # COMPLETE NAME
            complete_name = prefix_name+ "_"+ dir_+"_"+dir_
            config['prefix_name'] = complete_name
            # CLASS 2 COLOR
            class2colors,dict_name = get_classes(config,complete_name)
            config['colors'] = list(class2colors.values())
            config['class_names'] = list(dict_name.values())
            config['class_idx'] = list(dict_name.keys())
            # TIME SETTINGS
            config["start_date"] =dir_ +" 00:00:00"
            config["end_date"]=dir_ +" 23:59:59"
            config["bin_time"] = 15
            config["day_in_sec"] = 86400 
            config["dt"]= 900
            config["day_analysis"]= dir_
            # SUBNETS            
            config["save_label"] = ["_complete_intersection_","_complete_complement_"]
            config['subnet_txt_files_complete_intersection'] = os.path.join(config["working_dir"],complete_name+"_complete_intersection"+".txt") 
            config['subnet_txt_files_complement'] = []
            config['subnet_txt_files'] = []
            config['subnet_csv_files_complete_intersection'] = os.path.join(config["working_dir"],complete_name+"_complete_intersection_velocity_subnet.csv") 
            config['subnet_csv_files_complement'] = []
            config['subnet_csv_files'] = []
            config['hierarchical_subnets'] = []
            list_fundamental_diagram_class = []
            for n in class2colors.keys():
                config['subnet_txt_files'].append(os.path.join(config["working_dir"],complete_name+"class_"+str(n)+".txt"))
                config['subnet_txt_files_complement'].append(os.path.join(config["working_dir"],complete_name+str(n)+"_complete_complement"+".txt"))
                config['subnet_csv_files'].append(os.path.join(config["working_dir"],complete_name+"_class_"+str(n)+"_velocity_subnet.csv"))
                config['subnet_csv_files_complement'].append(os.path.join(config["working_dir"],complete_name+"_complete_complement_"+str(n)+"_velocity_subnet.csv"))
                list_fundamental_diagram_class.append(os.path.join(config["working_dir"],'class_{}_for_fondamental_diagram.csv'.format(n)))
                config['hierarchical_subnets'].append(os.path.join(config["working_dir"],complete_name+str(n)+"_class_subnet.txt"))
#                list_colors.append(class2colors[n])
#                list_names.append(dict_name[n])
#                list_indices.append(n)
#                list_subnet_txt.append(os.path.join(config["working_dir"],complete_name+"class_"+str(n)+".txt"))
 #               list_complement_txt.append(os.path.join(config["working_dir"],complete_name+str(n)+"_complete_complement"+".txt"))
 #               list_subnet_csv.append(os.path.join(config["working_dir"],complete_name+"class_"+str(n)+"_velocity_subnet.csv"))
  #              list_complement_csv.append(os.path.join(config["working_dir"],complete_name+str(n)+"_complete_complement_velocity_subnet"+".csv"))
                
            config_aggregated['fundamental_diagram_dirs'].append(list_fundamental_diagram_class)
#            config_aggregated['timed_fluxes'].append(os.path.join(config["working_dir"],complete_name+"_timed_fluxes"+".csv"))
#            config_aggregated['fcm_files'].append(os.path.join(config["working_dir"],complete_name+"_fcm"+".csv"))
#            config_aggregated['complete_intersection_txt'].append(os.path.join(config["working_dir"],complete_name+"_complete_intersection"+".txt"))
#            config_aggregated['colors'].append(list_colors)
#            config_aggregated['classes_names'].append(list_names)
#            config_aggregated['classes_indices'].append(list_indices)
#            config_aggregated['single_subnet_txt'].append(list_subnet_txt)
#            config_aggregated['complement_subnet_txt'].append(list_complement_txt)
#            config_aggregated['complete_intersection_velocity_subnet'].append(os.path.join(config["working_dir"],complete_name+"_complete_intersection_velocity_subnet"+".csv"))
#            config_aggregated['single_subnet_csv'].append(list_subnet_csv)
 #           config_aggregated['complement_subnet_csv'].append(list_complement_csv)
            
            
            with open(config_dir,'w') as f:
                json.dump(config,f,indent=2)
                
            # WRITE BASH THAT LAUNCHES plot_subnet.py for all the config files created            

            bash_script += "python3 ./python/fluxes_subnet.py -c {}\n".format(config_dir)

            # Write the generated Bash script to a file
            with open(os.path.join(base_path,"all_subnets.sh"), "w") as file:
                file.write(bash_script)
                
    # AGGREGATED FONDAMENTAL DIAGRAM
    with open(os.path.join(base_path,"config_aggregated_fundamental_diagram.json"), "w") as file:
        json.dump(config_aggregated,file,indent=2)
    
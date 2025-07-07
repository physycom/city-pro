import pandas as pd
import numpy as np

def compute_avspeed_and_variance_per_class_Fcm(Fcm):
    """
    @params Fcm: DataFrame with trajectory aggragated info.
        - In particular this function extracts informations about the average speed and variance speed per class
    """
    classes = []
    # Prepare average speed variance speed per class
    average_speed_class_from_Fcm = []
    variance_speed_class_from_Fcm = []
    for class_,df_class in Fcm.groupby("class"):
        print("class: ",class_)
        average_speed_class_from_Fcm.append(np.mean(df_class["speed_kmh"].to_numpy()))
        variance_speed_class_from_Fcm.append(np.std(df_class["speed_kmh"].to_numpy()))
        classes.append(class_)
    DfVariability = pd.DataFrame({"av_speed_kmh":average_speed_class_from_Fcm,
                                "var_speed_kmh":variance_speed_class_from_Fcm,
                                "class":classes})
    return DfVariability


# Prepare the vectors (<v>,<v>/<sigma>)
def compute_distribution_speed_from_trajectories_per_class(Fcm,Trajdf,Df_v_boundary):
    """
        @params Fcm: DataFrame with trajectory aggragated info.
        @params Trajdf: DataFrame with trajectories
        @description: This function computes the distribution of the speed and the ratio between the average speed and the variance speed
        @output: DfScatter
            - class
            - av_speed_kmh
            - var_speed_kmh
            - var_speed_over_av_speed
        Per each user I have the average and variance speed
    """
    average_speeds = []
    variance_speeds_over_average_speeds = []
    variance_speeds = []
    classes = []
    user_id = []
    partition_in_class_trajectory_all = {class_: [] for class_ in Df_v_boundary["class"].unique()}
    for class_,df_class in Fcm.groupby("class"):
        print("class: ",class_)
        UsersFcm = df_class["id_act"].to_list()
        TrajdfClass = Trajdf[Trajdf["user_id"].isin(UsersFcm)]              
        for user,traj_user in TrajdfClass.groupby("user_id"):
            partition_in_class_trajectory = compute_partition_class_traj(traj_user,Df_v_boundary)                       # NOTE: Compute P(v_trj|c) for each user
            for key in partition_in_class_trajectory.keys():                                                      
                partition_in_class_trajectory_all[key].append(partition_in_class_trajectory[key])                           # NOTE: Append normalized P(v_trj|c) for each user
            average_speed = np.mean(traj_user["speed_kmh"])
            variance_speed = np.std(traj_user["speed_kmh"])
            variance_speeds_over_average_speed = variance_speed/average_speed
            average_speeds.append(average_speed)
            variance_speeds.append(variance_speed)
            variance_speeds_over_average_speeds.append(variance_speeds_over_average_speed)
            classes.append(class_)
            user_id.append(user)
    DfScatter = {"av_speed_kmh":average_speeds,
                            "var_speed_kmh":variance_speeds,
                            "var_speed_over_av_speed":variance_speeds_over_average_speeds,
                            "class":classes,
                            "user_id":user_id}
    for key in partition_in_class_trajectory.keys():
        DfScatter[key] = partition_in_class_trajectory_all[key]
    DfScatter = pd.DataFrame(DfScatter)
    return DfScatter


def compute_class2_v_mean_v_min_v_max(Fcm):
    """
    @params Fcm: DataFrame with trajectory aggragated info.
        @description: This function computes the mean, min and max speed per class
        @output: Df_v_boundary
            - v_mean
            - v_min
            - v_max
            - class
        These are the classes speed and speed standard deviation boundaries that will be used to classify the trajectories transitions from class to class

    """
    classes = []
    v_mean = []
    v_min = []
    v_max = []
    for class_,df_class in Fcm.groupby("class"):
        print("class: ",class_)
        v_ = np.mean(df_class["speed_kmh"].to_numpy())
        v_2 = np.std(df_class["speed_kmh"].to_numpy())
        v_min_ = v_ - v_2
        v_max_ = v_ + v_2
        v_mean.append(v_)
        v_min.append(v_min_)
        v_max.append(v_max_)
        classes.append(class_)
    Df_v_boundary = pd.DataFrame({"v_mean":v_mean,
                                "v_min":v_min,
                                "v_max":v_max,
                                "class":classes})
    return Df_v_boundary
def compute_partition_class_traj(traj_user,Df_v_boundary):
    """
        @describe: This function computes the partition of the trajectory of a user in the classes
        @params traj_user: DataFrame with the trajectory of a user
        @params Df_v_boundary: DataFrame with the boundaries of the classes
        @output: class_2_fraction
            - class
    """
    class_2_fraction = {class_: 0 for class_ in Df_v_boundary["class"].unique()}
    for row,point_info in traj_user.iterrows():
        distance_average_speed_point_2_class_judge = 1000 
        # per each point
        for row_class,class_info in Df_v_boundary.iterrows():
            speed_distance_to_class = np.abs(point_info["speed_kmh"] - class_info["v_mean"])
            # compare what is the closest class in speed
            if speed_distance_to_class < distance_average_speed_point_2_class_judge:
                distance_average_speed_point_2_class_judge = speed_distance_to_class
                class_judge = class_info["class"]
            # assign the class to the fraction
        class_2_fraction[class_judge] += 1
    for key in class_2_fraction.keys():
        class_2_fraction[key] = class_2_fraction[key]/len(traj_user)
    return class_2_fraction


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

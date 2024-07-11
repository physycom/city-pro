import numpy as np
def GenerateDay2DictClassAvSpeed(ListDailyNetwork,ReferenceFcmCenters,DictClass2AvSpeed,RefIntClass2StrClass,Day2IntClass2StrClass,Day2StrClass2IntClass):
    for MobDate in ListDailyNetwork:
        # Compare the class speed 
        for class_,Df in MobDate.FcmCenters.group_by("class"):
            # to all the reference class speeds.
            IntClassBestMatch = 0
            MinDifferenceInSpeed = 100000
            # Choose the speed that is closest to the reference
            for class_ref,Df_ref in ReferenceFcmCenters.group_by("class"):
                if np.abs(Df["av_speed"].to_list()[0] - Df_ref["av_speed"].to_list()[0]) < MinDifferenceInSpeed:
                    MinDifferenceInSpeed = np.abs(Df["av_speed"].to_list()[0] - Df_ref["av_speed"].to_list()[0])
                    IntClassBestMatch = class_ref
                else:
                    pass
            # Fill preparatory dictionary
            DictClass2AvSpeed[MobDate.StrDate][class_] = IntClassBestMatch
        # {StrDay: {IntClassStrDay: IntClassRefDay}}            
        for Class in DictClass2AvSpeed[MobDate.StrDate].keys():
            # {StrDay: {IntClassStrDay: StrClassRefDay}}
            Day2IntClass2StrClass[MobDate.StrDate][Class] = RefIntClass2StrClass[DictClass2AvSpeed[MobDate.StrDate][Class]]
            Day2StrClass2IntClass[MobDate.StrDate][RefIntClass2StrClass[DictClass2AvSpeed[MobDate.StrDate][Class]]] = Class
        return Day2IntClass2StrClass,Day2StrClass2IntClass,DictClass2AvSpeed

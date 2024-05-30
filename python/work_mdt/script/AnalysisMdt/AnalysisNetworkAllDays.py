from AnalysisNetwork1Day import *

class NetworkAllDays:
    
    self.StrDates = [] 
    def __init__(self,ListDailyMobilities): 
        self.ListDailyMobilities = ListDailyMobilities       
        for MobDate in self.ListDailyMobilities:
            self.StrDates.append(MobDate.StrDate)
    def ConcatenateAllFcms(self,label):
        self.ConcatenatedFcm = None
        for DailyMob in ListDailyMobilities:
            if self.ConcatenatedFcm is None:
                self.ConcatenatedFcm = DailyMob.Fcm
            else:
                self.ConcatenatedFcm = pd.concatenate(self.ConcatenatedFcm,DailyMob.Fcm)

    def PlotAverageDistribution(self,label,bins = 100):
        """
            Input:
                label: str -> time, length, av_speed, p, a_max, class
            Returns:
                n, bins of velocity distribution
        """
        if self.ConcatenatedFcm is None:
            print("No plot Average, lack of concatenated fcm")
        else:
            fig, ax = plt.subplots(1,1,figsize = (10,8))
            ax.hist(self.ConcatenatedFcm[label], bins = bins)   
            ax.set_xlabel(label)
            ax.set_ylabel("count")
            plt.savefig(os.path.join(self.SaveDir,"Average_{}.png".format(label),dpi = 200))

            
    def PlotDistributionPerClass(fcm,label,class_):
        """
            Input:
                fcm: DataFrame
                label: str -> time, length, av_speed, p, a_max, class
            NOTE: TODO in MobilityDaily put a section that allows to take into account
                        the classes in terms of their velocities
        """
        if self.ConcatenatedFcm is None:
            print("No plot Average, lack of concatenated fcm")
        else:
            fig, ax = plt.subplots(1,1,figsize = (10,8))
            ax.hist(self.ConcatenatedFcm.groupby("class").get_group(label), bins = bins)   
            ax.set_xlabel(label)
            ax.set_ylabel("count")
            ax.set_title("Class ")
            plt.savefig(os.path.join(self.SaveDir,"Average_{}.png".format(label),dpi = 200))


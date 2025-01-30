import polars as pl
import logging
from collections import defaultdict
logger = logging.getLogger(__name__)
from numpy import logical_and,sort
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def AssociateHierarchicalClass2Users(Fcm,FcmNew):
    """
        @brief: Associate the hierarchical class to the users
        @param Fcm: DataFrame with the Fcm
        @param FcmNew: DataFrame with the FcmNew
    """
    logger.info("Add hierarchical class to users in Fcm")
    logger.debug("Fcm New: {}".format(FcmNew.head()))
    logger.debug("Fcm: {}".format(Fcm.head()))
    FcmNew = FcmNew.with_columns([FcmNew['class'].alias('class_new')])
    if "class_new_right" in Fcm.columns:
        Fcm = Fcm.drop(['class_new_right'])
    if "class_new" in Fcm.columns:
        Fcm = Fcm.drop(['class_new'])
    Fcm = Fcm.drop(['class_new',"class_new_right"])
    Fcm = Fcm.join(FcmNew[['id_act', 'class_new']], on='id_act',suffix='')
    return Fcm



def GroupByClassAndTimeDeparture(Fcm,BinTimeStamp,ClassCol = 'class_new',TimeDepartureCol = 'start_time'):
    """
        @brief: Group the Fcm by class and time_departure
        @param Fcm: DataFrame with the Fcm

    """
    logger.info("Group by class and time_departure")
    OrderedClass2TimeDeparture2UserId = defaultdict()
    # Divide Trajectories By Class
    for Class,FcmClass in Fcm.groupby(ClassCol):
        if Class not in OrderedClass2TimeDeparture2UserId.keys():
            OrderedClass2TimeDeparture2UserId[Class] = defaultdict()
        for t in range(len(BinTimeStamp)-1):
            # Select Time Bin            
            BinTime = BinTimeStamp[t]
            BinTimeNext = BinTimeStamp[t+1]
            # Select Trajectories in the Time Bin 
            mask = (FcmClass[TimeDepartureCol] >= BinTime) & (FcmClass[TimeDepartureCol] < BinTimeNext)
            filtered = FcmClass.filter(mask)
            OrderedClass2TimeDeparture2UserId[Class][BinTime] = filtered["id_act"].to_list()
    return OrderedClass2TimeDeparture2UserId

def GroupByClass(Fcm,ClassCol = 'class_new'):
    """
        @brief: Group the Fcm by class
        @param Fcm: DataFrame with the Fcm
    """
    logger.info("Group by class")
    OrderedClass2UserId = defaultdict()
    for Class,FcmClass in Fcm.groupby(ClassCol):
        OrderedClass2UserId[Class] = FcmClass["id_act"].to_list()
    return OrderedClass2UserId

def ComputeTransitionClassMatrix(Fcm):
    """
        @brief: Compute the DataFrame with the count of the transitions between the classes
        @param Fcm: DataFrame with the Fcm
        ["ClassBefore","ClassAfter","LengthIntersection","NumberBefore","NumberAfter"]
        ClassBefore: Counts the People in the Class before the hierarchical re-organization
        ClassAfter: Counts the People in the Class after the hierarchical re-organization
        LengthIntersection: Counts the People that pass from the Class before to after the hierarchical re-organization 
        NumberBefore: Counts the People in the Class before the hierarchical re-organization
        NumberAfter: Counts the People in the Class after the hierarchical re-organization
        NOTE: Tij 
    """
    logger.info("Compute Transition Class Matrix")
    ClassBefore = []
    ClassAfter = []
    LengthIntersection = []
    NumberBefore = []
    NumberAfter = []
    Tij = []
    TransitionUsers = defaultdict()
    Classes = sort(Fcm["class"].unique())
    for Class,ClassFcm in Fcm.groupby("class"):
        NClass = len(ClassFcm)
        for ClassNew,ClassNewFcm in Fcm.groupby("class_new"):
            NClassNew = len(ClassNewFcm)
            TransitionUsers[f"{Class},{ClassNew}"] = list(set(ClassFcm["id_act"]).intersection(set(ClassNewFcm["id_act"])))
            Intersection = len(set(ClassFcm["id_act"]).intersection(set(ClassNewFcm["id_act"])))
            # Intersection = NClass*T_{Class,ClassNew} 
            ClassBefore.append(Class)
            ClassAfter.append(ClassNew)
            LengthIntersection.append(Intersection)
            NumberBefore.append(NClass)
            NumberAfter.append(NClassNew)
            Tij.append(Intersection/NClass)
    DfComparison = pl.DataFrame({"ClassBefore":ClassBefore,"ClassAfter":ClassAfter,"LengthIntersection":LengthIntersection,"NumberBefore":NumberBefore,"NumberAfter":NumberAfter,"Tij":Tij})
    return DfComparison,TransitionUsers

def GetUsersThattransitionHighSpeed(Fcm):
    """
        @brief: Get the users that transition from one class to another
        @param Fcm: DataFrame with the Fcm
    """
    logger.info("Get Users that transition")
    UsersThatTransition = []
    for Class,ClassFcm in Fcm.groupby("class"):
        for ClassNew,ClassNewFcm in Fcm.groupby("class_new"):
            Intersection = set(ClassFcm["id_act"]).intersection(set(ClassNewFcm["id_act"]))
            UsersThatTransition.append(Intersection)

    return UsersThatTransition

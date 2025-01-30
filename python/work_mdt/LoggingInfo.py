def AddMessageToLog(Message,LogFile):
    with open(LogFile,'a') as f:
        f.write(Message+'\n')

def MessagePlotDailyDistr(CountFunctionsCalled,LogFile,Upload = False):
    if Upload:
        CountFunctionsCalled += 1
        Message = "{} Plot Daily Speed Distr: True\n".format(CountFunctionsCalled)
        Message += "\tUpload self.InfoFittedParameters,self.DictFittedData"
        AddMessageToLog(Message,LogFile)
    else:
        CountFunctionsCalled += 1
        Message = "{} Plot Daily Speed Distr: True\n".format(CountFunctionsCalled)
        Message += "\tComputed Fitted Data"
        AddMessageToLog(Message,LogFile)

def MessagePlotTimePercorrenceDistributionAllClasses(CountFunctionsCalled,LogFile,Upload):
    if Upload:
        Message = "{} Plotting TimePercorrence Distribution: True\n".format(CountFunctionsCalled)
        Message += "\tUpload Class2Time2Distr, Class2AvgTimePercorrence"
        AddMessageToLog(Message,LogFile)
    else:
        Message = "{} Plotting TimePercorrence Distribution: True\n".format(CountFunctionsCalled)
        Message += "\tComputed Class2Time2Distr, Class2AvgTimePercorrence"
        AddMessageToLog(Message,LogFile)

def MessagePlotDistrAggregatedAllDays(Upload,CountFunctions,Aggregation,LogFile):
    if Upload:
        CountFunctions += 1
        Message = "{0} Plot {1} all Day Distr: True\n".format(CountFunctions,Aggregation)
        Message += "\tUpload self.InfoFittedParameters,self.DictFittedData"
        AddMessageToLog(Message,LogFile)
    else:
        CountFunctions += 1
        Message = "{0} Plot {1} all Day Distr: True\n".format(CountFunctions,Aggregation)
        Message += "\tComputed Fitted Data"
        AddMessageToLog(Message,LogFile)

def MessagePlotSingleClass0(Upload,CountFunctionsCalled,LogFile):
    if Upload:
        CountFunctionsCalled += 1
        Message = "{} Plot Distr Per Class: True\n".format(CountFunctionsCalled)
        Message += "\tUpload Class2DictFittedData, Class2InfoFittedParameters"
        AddMessageToLog(Message,LogFile)
    else:
        CountFunctionsCalled += 1
        Message = "{} Plot Distr Per Class: True\n".format(CountFunctionsCalled)
        Message += "\tComputed Class2DictFittedData, Class2InfoFittedParameters"
        AddMessageToLog(Message,LogFile)

def MessagePlotSingleClass1(Feature,IntClass,Class2DictFittedData,Class2InfoFittedParameters,LogFile):
    Message = "\tPlot {0} Distribution Class {1}: True\n".format(Feature,IntClass)
    Message += "\t\tFitting Function {0}".format(Class2DictFittedData[IntClass][Feature]["best_fit"])
    AddMessageToLog(Message,LogFile)



###----------- ALL DAYS ANALYSIS -----------------###

def MessageConcatenateFcm(Aggregation2Fcm,Aggregation2Class2Fcm,LogFile):
    Message = "ConcatenatePerClass: True\n self.Aggregation2Fcm\n"
    for Aggregation in Aggregation2Fcm.keys():
        Message += "\tAggregation: {}".format(Aggregation)
        Message += "-> Number of Trajectories: {}".format(len(Aggregation2Fcm[Aggregation]))
        AddMessageToLog(Message,LogFile)
    Message = "ConcatenatePerClass: self.Aggregation2Class2Fcm\n"
    for Aggregation in Aggregation2Class2Fcm.keys():
        for StrClass in Aggregation2Class2Fcm[Aggregation].keys():
            Message += "\tAggregation: {}".format(Aggregation)
            Message += "\tStrClass: {}".format(StrClass)
            Message += "-> Number of Trajectories: {}".format(len(Aggregation2Class2Fcm[Aggregation][StrClass]))
        AddMessageToLog(Message,LogFile)

def MessageAveSpeedStrClass(Day2IntClass2StrClass,Day2StrClass2IntClass,DictClass2AvSpeed,LogFile):
    Message = "Associate Days Classes to common StrClass:\n"
    Message += "\tself.Day2IntClass2StrClass\n" 
    for Day in Day2IntClass2StrClass.keys():
        Message += "\t\t <{}".format(Day)
        for Class in Day2IntClass2StrClass[Day].keys():
            Message += ":< {}".format(Class)
            Message += ": {}".format(Day2IntClass2StrClass[Day][Class])
        Message += ">"
    Message += ">"
    AddMessageToLog(Message,LogFile)
    Message = "\tself.Day2StrClass2IntClass\n"
    for Day in Day2StrClass2IntClass.keys():
        Message += "\t\t <{}".format(Day)
        for Class in Day2StrClass2IntClass[Day].keys():
            Message += ":< {}".format(Class)
            Message += ": {}".format(Day2StrClass2IntClass[Day][Class])
        Message += ">"
    Message = "\tself.DictClass2AvSpeed\n"
    AddMessageToLog(Message,LogFile)
    for StrClass in DictClass2AvSpeed.keys():
        Message += "\t\t <{}".format(StrClass)
        for AvSpeed in DictClass2AvSpeed[StrClass].keys():
            Message += ":< {}".format(AvSpeed)
        Message += ">"
    AddMessageToLog(Message,LogFile)



def MessageComputeMFDAllDays(Aggregation2MFD,Aggregation2Class2MFD,LogFile):
    Message = "ComputeMFDAllDays: True\n"
    Message += "\tself.Aggregation2MFD\n"
    for Aggregation in Aggregation2MFD.keys():
        Message += "\tAggregation: {}".format(Aggregation)
        Message += "-> Number of Trajectories: {}".format(len(Aggregation2MFD[Aggregation]))
    AddMessageToLog(Message,LogFile)
    Message = "ComputeMFDAllDays: self.Aggregation2Class2MFD\n"
    for Aggregation in Aggregation2Class2MFD.keys():
        for StrClass in Aggregation2Class2MFD[Aggregation].keys():
            Message += "\tAggregation: {}".format(Aggregation)
            Message += "\tStrClass: {}".format(StrClass)
            Message += "-> Number of Trajectories: {}".format(len(Aggregation2Class2MFD[Aggregation][StrClass]))

def MessageInitInfo(Aggregation2Dict2InitialGuess,Aggregation2Class2DictFittedData,LogFile):
    Message = "InitFitInfo: True\n"
    Message += "\tself.Aggregation2Dict2InitialGuess\n"
    for Aggregation in Aggregation2Dict2InitialGuess.keys():
        Message += "\tAggregation: {}".format(Aggregation)
        Message += "-> Initial Guess: {}".format(Aggregation2Dict2InitialGuess[Aggregation])
    AddMessageToLog(Message,LogFile)
    Message = "InitFitInfo: self.Aggregation2Class2DictFittedData\n"
    for Aggregation in Aggregation2Class2DictFittedData.keys():
        for StrClass in Aggregation2Class2DictFittedData[Aggregation].keys():
            Message += "\tAggregation: {}".format(Aggregation)
            Message += "\tStrClass: {}".format(StrClass)
            Message += "-> Initial Guess: {}".format(Aggregation2Class2DictFittedData[Aggregation][StrClass])
    AddMessageToLog(Message,LogFile)


def MessageDict2AvSpeed(Day2IntClass2StrClass,Day2StrClass2IntClass,DictClass2AvSpeed,LogFile):
    Message = "GenerateDay2DictClassAvSpeed: True\n"
    for Day in Day2IntClass2StrClass.keys():
        Message += "\tDay: {}\n".format(Day)
        Message += "\t\tDay2IntClass2StrClass\n"
        for Class in Day2IntClass2StrClass[Day].keys():
            Message += "\t\t\tClass {0} -> {1}\n".format(Class,Day2IntClass2StrClass[Day][Class])
        Message += "\t\tDay2StrClass2IntClass\n"
        for Class in Day2StrClass2IntClass[Day].keys():
            Message += "\t\t\tClass {0} -> {1}\n".format(Class,Day2StrClass2IntClass[Day][Class])
        Message += "\t\tDictClass2AvSpeed\n"
        for Class in DictClass2AvSpeed.keys():
            Message += "\t\t\tClass {0} -> {1}\n".format(Class,DictClass2AvSpeed[Class])
    AddMessageToLog(Message,LogFile)
import json
import os 
from collections import defaultdict
import numpy as np

class NumpyArrayEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert ndarray to list
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj) 

def FillDictofDictsFromKeys(ListKeys,ListDicts):
    """
        Input:
            - ListKeys: list -> List of Keys
            - ListDicts: list -> List of Dictionaries
        Output:
            - DictofDicts: dict -> Dictionary of Dictionaries
        Description:
            NOTE: The ListKeys and ListDicts must have the same length and the same order. (Name of the dict in ListDicts must be the same as the key in ListKeys)
    """
    DictofDicts = defaultdict(dict)
    for Key in range(len(ListKeys)):
        DictofDicts[ListKeys[Key]] = ListDicts[Key]
    return DictofDicts

def FillDictFileNamesFromKeys(ListKeys,ListFormats,Extenstion):
    """
        Input:
            - ListKeys: list -> List of Keys
            - ListFormats: list -> List of Formats
            - Extenstion: str -> Extenstion of the File
        Output:
            - DictOfFilenames: dict -> Dictionary of File Names
        Description:
            NOTE: The end we have DictFileNames[Key] = Key_Format1_Format2_..._FormatN.Extension
            NOTE: To Use with FillDictofDictsFromKeys with same Keys.

    """
    DictOfFilenames = defaultdict(str)
    for Key in range(len(ListKeys)):
        DictOfFilenames[ListKeys[Key]] = ListKeys[Key] 
        for Format in ListFormats:
            DictOfFilenames[ListKeys[Key]] += "_{}".format(Format)
        DictOfFilenames[ListKeys[Key]] += Extenstion
    return DictOfFilenames
def UploadDictsFromListFilesJson(ListFileNames):
    ListOutDict = []
    Bool = False
    for i in range(len(ListFileNames)):
        if os.path.isfile(ListFileNames[i]):
            Bool = True
            with open(ListFileNames[i],'r') as f:
                ListOutDict.append(json.load(f))
    return ListOutDict,Bool

def SaveListDict2Json(DictOfDicts,DictFileNames):
    """
        Input:
            - ListDict: list -> List of Dictionaries
    """
    assert DictOfDicts.keys() == DictFileNames.keys()
    for Key in DictOfDicts.keys():
        with open(DictFileNames[Key],'w') as f:
            json.dump(DictOfDicts[Key],f,indent = 2)

def SaveListDict2JsonArray(DictOfDicts,DictFileNames):
    """
        Input:
            - ListDict: list -> List of Dictionaries
    """
    assert DictOfDicts.keys() == DictFileNames.keys()
    for Key in DictOfDicts.keys():
        with open(DictFileNames[Key],'w') as f:
            json.dump(DictOfDicts[Key],f,cls = NumpyArrayEncoder,indent = 2)


def GenerateListFilesCommonBaseDir(BaseDir,ListNames):
    """
        Input:
            - BaseDir: str -> Base Directory
            - ListFiles: list -> List of Files
        Output:
            - ListFiles: list -> List of Files with Common Base Directory
    """
    ListFiles = []
    for i in range(len(ListNames)):
        ListFiles.append(os.path.join(BaseDir,ListNames[i]))
    return ListFiles

# Save Procedure TimePercorrence
def SaveProcedure(BaseDir,ListKeys,ListDicts,ListFormats,Extension):
    """
        Input:
            - ListKeys: list -> List of Keys
            - ListDicts: list -> List of Dictionaries
            - ListFormats: list -> List of Formats
            - Extension: str -> Extension of the File
        Output:
            - DictFileNames: dict -> Dictionary of File Names
        Description:
            1) Fill the Dictionary of Dictionaries
            2) Fill the Dictionary of File Names
            3) Save the List of Dictionaries to Json
    """
    DictOfDicts = FillDictofDictsFromKeys(ListKeys,ListDicts)
    DictFileNames = FillDictFileNamesFromKeys(ListKeys,ListFormats,Extension)
    for Key in DictFileNames.keys():
        DictFileNames[Key] = os.path.join(BaseDir,DictFileNames[Key])
    SaveListDict2Json(DictOfDicts,DictFileNames)

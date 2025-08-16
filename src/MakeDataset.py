import os
import pandas as pd
from pathlib import Path
from object import Object


'''
@param: DirectoryPath:str -> Directory Path to detection results
@return: list[objects] -> List of Detection results
'''
def GetDetectObject(DirectoryPath, CameraID):
    ImagePaths = os.listdir(DirectoryPath)
    ObjectsList = []
    for ImagePath in ImagePaths:
        with open(os.path.join(DirectoryPath, ImagePath)) as file:
            lines = file.readlines()
            objects = []
            for line in lines:
                data = line.strip().split(',')
                if len(data) == 8:
                    obj = Object(*data)
                    # valid object class
                    if obj.category not in [0, 1, 2, 9, 11]:
                        continue
                    obj.CameraID(CameraID)
                    objects.append(obj)
        ObjectsList.append(objects)
    return ObjectsList

'''
@param: list[DirectoryPath] -> List of Path to detection results
@return: list[dataset] -> Dataset
'''
def MakeDataset(Directory_paths):
    datasets = list()
    CameraID = 0
    for Directory_path in Directory_paths:
        Objects = GetDetectObject(Directory_path, CameraID)
        datasets.append(Objects)
        CameraID += 1
    
    datasets = list(zip(*datasets))
    return datasets
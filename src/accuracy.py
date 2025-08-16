from CommonCaluculation import IoU, IoU_Threshold
from Ensemble import Ensemble
from CommonCaluculation import *

'''
@param: dataset - List of detection results from each model
@param: MV_Threshold - Threshold for how many inferences to include in the result
@return: numTP, numFP, numFN - Count of True Positives, False Positives, False Negatives after majority voting
'''
def MajorityVoting_TP(dataset, MV_Threshold) -> int:
    """
    Combine True Positive inferences using majority voting
    @param dataset: List containing inference results from each model
    @param MV_Threshold: int - Threshold for number of models required to adopt an inference
    @return numTP: int - Number of TPs meeting the threshold
    """
    # Collect all TP inferences from dataset
    TP = list()
    for model in dataset:
        for infer in model:
            if infer.isTrue is True:
                TP.append(infer)

    numTP = 0
    processed = set()
    numModel = len(dataset)

    # Count inferences that match across threshold number of models
    while TP:
        infer = TP.pop(0)
        processed_camera = set()
        processed_camera.add(infer.CameraID)
        if infer in processed:
            continue  # Skip already processed inferences

        numMatch = 1
        boxA = [infer.X_coordinate, infer.Y_coordinate, infer.width, infer.height]

        for comp in TP[:]:  # Use copy for safe deletion
            if (infer.CameraID == comp.CameraID) or (comp in processed) or (comp.CameraID in processed_camera):
                continue  # Skip inferences from same model or already processed ones

            boxB = [comp.X_coordinate, comp.Y_coordinate, comp.width, comp.height]
            if (IoU(boxA, boxB) > IoU_Threshold) and (infer.category == comp.category):
                numMatch += 1
                processed.add(comp)  # Add matching inference to processed set
                processed_camera.add(comp.CameraID)
                TP.remove(comp)  # Remove from list
                if numMatch == numModel:
                    break

        processed.add(infer)  # Add current inference to processed set

        if numMatch >= MV_Threshold:
            numTP += 1

    return numTP

def MajorityVoting_FP(dataset, MV_Threshold) -> int:
    """
    Combine False Positive inferences using majority voting
    @param dataset: List containing inference results from each model
    @param MV_Threshold: int - Threshold for number of models required to adopt an inference
    @return numFP: int - Number of FPs meeting the threshold
    """
    # Collect all FP inferences from dataset
    FP = list()
    for model in dataset:
        for infer in model:
            if infer.conf > 0 and infer.isTrue is False:
                FP.append(infer)

    numFP = 0
    processed = set()

    # Count FPs that match across threshold number of models
    while FP:
        infer = FP.pop(0)
        if infer in processed:
            continue  # Skip already processed inferences

        numMatch = 1
        boxA = [infer.X_coordinate, infer.Y_coordinate, infer.width, infer.height]

        for comp in FP[:]:  # Use copy for safe deletion
            if (infer.CameraID == comp.CameraID) or (comp in processed):
                continue  # Skip inferences from same model or already processed ones

            boxB = [comp.X_coordinate, comp.Y_coordinate, comp.width, comp.height]
            if IoU(boxA, boxB) > IoU_Threshold:
                processed.add(comp)  # Add matching inference to processed set
                numMatch += 1
                FP.remove(comp)  # Remove from list

        processed.add(infer)  # Add current inference to processed set

        if numMatch >= MV_Threshold:
            numFP += 1

    return numFP

def MajorityVoting_FN(dataset, MV_Threshold):
    """
    Determine common False Negatives across models using majority voting
    @param dataset: List of model outputs
    @param MV_Threshold: Threshold for majority voting
    @return numFN: Number of common FNs across models
    """
    numModel = len(dataset)
    FN = list()

    # Collect FNs from model outputs
    for model in dataset:
        for infer in model:
            if infer.conf == 0:
                FN.append(infer)

    numFN = 0
    processed = set()
    for infer in FN:
        if infer in processed:
            continue
        numMatch = 1  # Count the initial object itself
        for comp in FN:
            if (infer == comp) or (comp in processed) or (infer.CameraID == comp.CameraID):
                continue
            # Check match by category
            if infer.category == comp.category:
                numMatch += 1
                processed.add(comp)
        processed.add(infer)
        # Count as FN if matches meet threshold
        if numMatch >= numModel - MV_Threshold + 1:
            numFN += 1

    return numFN
 
'''
Calculate overall accuracy (acc) after majority voting on dataset
@param: datasets - List of detection results
@return: acc:float - Accuracy score
'''
def accuracy(datasets, mode):
    acc = 0
    total = {'TP':0, 'FP':0, 'FN':0}
    numModel = len(datasets[0])
    for dataset in datasets:
        if mode == 'affirmative':
            TP = MajorityVoting_TP(dataset, 1)
            FP = MajorityVoting_FP(dataset, 1)
            FN = MajorityVoting_FN(dataset, 1)
            
            sum_infer = TP + FP + FN
        elif mode == 'consensus':
            TP = MajorityVoting_TP(dataset, numModel//2 + 1 if numModel % 2 == 1 else numModel//2)  
            FP = MajorityVoting_FP(dataset, numModel//2 + 1 if numModel % 2 == 1 else numModel//2)
            FN = MajorityVoting_FN(dataset, numModel//2 + 1 if numModel % 2 == 1 else numModel//2)
            
            sum_infer = TP + FP + FN
        elif mode == 'unanimous':
            TP = TP_of_All_Model(dataset)
            FP = FP_of_All_Model(dataset)
            FN = FN_of_Each_Model(dataset)
            
            sum_infer = TP_of_All_Model(dataset) + FP_of_All_Model(dataset) + FN_of_Each_Model(dataset)
            
        try:
            acc += TP/sum_infer
        except ZeroDivisionError:
            acc += 1
    return acc / len(datasets)
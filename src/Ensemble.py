from CommonCaluculation import *

def affirmative(dataset):
    Ensembled = list()
    
    detects = list()
    detects.extend(extract_infer(dataset, 'TP'))
    detects.extend(extract_infer(dataset, 'FP'))
    
    processed = set()
    for detect in detects:
        if detect in processed:
            continue
        boxA = [detect.X_coordinate, detect.Y_coordinate, detect.width, detect.height]
        ismatch = 1
        for comp in detects:
            if (comp in processed) or (detect.CameraID == comp.CameraID):
                continue
            
            boxB = [comp.X_coordinate, comp.Y_coordinate, comp.width, comp.height]
            if (IoU(boxA, boxB) >IoU_Threshold) and (detect.category == comp.category):
                ismatch += 1
                processed.add(comp)
        if ismatch >= 1:
            Ensembled.append(detect)
        processed.add(detect)
    
    return Ensembled

def consensus(dataset):
    Ensembled = list()
    numModel = len(dataset)
    majority = numModel // 2 + 1 if numModel % 2 == 1 else numModel // 2
    
    detects = list()
    detects.extend(extract_infer(dataset, 'TP'))
    detects.extend(extract_infer(dataset, 'FP'))
    
    processed = set()
    for detect in detects:
        if detect in processed:
            continue
        boxA = [detect.X_coordinate, detect.Y_coordinate, detect.width, detect.height]
        ismatch = 1
        for comp in detects:
            if (comp in processed) or (detect.CameraID == comp.CameraID):
                continue
            
            boxB = [comp.X_coordinate, comp.Y_coordinate, comp.width, comp.height]
            if (IoU(boxA, boxB) >IoU_Threshold) and (detect.category == comp.category):
                ismatch += 1
                processed.add(comp)
        if ismatch >= majority:
            Ensembled.append(detect)
        processed.add(detect)
    
    return Ensembled

def unanimous(dataset):
    Ensembled = list()
    numModel = len(dataset)
    
    detects = list()
    detects.extend(extract_infer(dataset, 'TP'))
    detects.extend(extract_infer(dataset, 'FP'))
    
    processed = set()
    for detect in detects:
        if detect in processed:
            continue
        processed_camera = set()
        processed_camera.add(detect.CameraID)
        boxA = [detect.X_coordinate, detect.Y_coordinate, detect.width, detect.height]
        ismatch = 1
        for comp in detects:
            if (comp in processed) or (comp.CameraID in processed_camera):
                continue
            
            boxB = [comp.X_coordinate, comp.Y_coordinate, comp.width, comp.height]
            if (IoU(boxA, boxB) >IoU_Threshold) and (detect.category == comp.category):
                ismatch += 1
                processed.add(comp)
                processed_camera.add(comp.CameraID)
        if ismatch == numModel:
            Ensembled.append(detect)
        processed.add(detect)
    
    return Ensembled

def Ensemble(dataset, mode):
    '''
    データセットをアンサンブルする
    @param datasets:list
    @return list
    '''
    # 各カメラの推論結果をアンサンブル
    if mode == 'affirmative':
        Ensembled = affirmative(dataset)
    elif mode == 'consensus':
        Ensembled = consensus(dataset)
    elif mode == 'unanimous':
        Ensembled = unanimous(dataset)
    else:
        raise ValueError('modeはaffirmative, majority, unanimousのいずれかを指定してください')
        
    return Ensembled
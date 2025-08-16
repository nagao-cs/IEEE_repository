IoU_Threshold = 0.5

# boxA=[xmin, ymin, width, height]
def IoU(boxA, boxB):
    '''
    Calculate IoU between two bounding boxes
    '''
    ax_mn, ay_mn, ax_mx, ay_mx = boxA[0], boxA[1], boxA[2]+boxA[0], boxA[3]+boxA[1]
    bx_mn, by_mn, bx_mx, by_mx = boxB[0], boxB[1], boxB[2]+boxB[0], boxB[3]+boxB[1]

    a_area = (ax_mx - ax_mn + 1) * (ay_mx - ay_mn + 1)
    b_area = (bx_mx - bx_mn + 1) * (by_mx - by_mn + 1)

    abx_mn = max(ax_mn, bx_mn)
    aby_mn = max(ay_mn, by_mn)
    abx_mx = min(ax_mx, bx_mx)
    aby_mx = min(ay_mx, by_mx)
    w = max(0, abx_mx - abx_mn + 1)
    h = max(0, aby_mx - aby_mn + 1)
    intersect = w*h

    iou = intersect / (a_area + b_area - intersect)
    return iou 

def extract_infer(dataset, mode):
    '''
    Extract specific inferences from dataset
    @args: dataset, mode
    @return infer:list -> List of specific inferences
    '''
    infer = list()
    
    if mode == 'TP':
        for model_infers in dataset:
            infer.extend(list(filter(lambda infer: infer.isTrue == True and infer.conf > 0, model_infers)))
    elif mode == 'FP':
        for model_infers in dataset:
            infer.extend(list(filter(lambda infer: infer.isTrue == False and infer.conf > 0, model_infers)))
    elif mode == 'FN':
        for model_infers in dataset:
            infer.extend(list(filter(lambda infer: infer.isTrue == False and infer.conf == 0, model_infers)))
    else:
        raise ValueError('mode must be TP, FP or FN')
    
    return infer

def count_matches(infers, match_threshold):
    '''
    Count the number of matching inferences for a single inference
    @args: infer, match_threshold
    @return numMatch:int -> Number of matching inferences
    '''
    numMatch = 0
    processed = set()
    try:
        sample = infers[0]
    except IndexError:
        return 0
    if sample.isTrue == True:
        mode = 0 # TP
    elif sample.isTrue == False:
        if sample.conf > 0:
            mode = 1 # FP
        elif sample.conf == 0:
            mode = 2 # FN
    else:
        raise ValueError('sample.isTrue must be True or False')
    
    if mode == 0:
        for infer in infers:
            if infer in processed:
                continue
            boxA = [infer.X_coordinate, infer.Y_coordinate, infer.width, infer.height]
            ismatch = 1
            for comp in infers:
                if (comp in processed) or (infer.CameraID == comp.CameraID):
                    continue
                boxB = [comp.X_coordinate, comp.Y_coordinate, comp.width, comp.height]
                if (IoU(boxA, boxB) >IoU_Threshold) and (infer.category == comp.category):
                    ismatch += 1
                    processed.add(comp)
            if ismatch >= match_threshold:
                numMatch += 1
            processed.add(infer)
            
    elif mode == 1:
        for infer in infers:
            if infer in processed:
                continue
            boxA = [infer.X_coordinate, infer.Y_coordinate, infer.width, infer.height]
            ismatch = 1
            for comp in infers:
                if (comp in processed) or (infer.CameraID == comp.CameraID):
                    continue
                boxB = [comp.X_coordinate, comp.Y_coordinate, comp.width, comp.height]
                if IoU(boxA, boxB) > IoU_Threshold:
                    ismatch += 1
                    processed.add(comp)
            if ismatch >= match_threshold:
                numMatch += 1
            processed.add(infer)
    
    elif mode == 2:
        for infer in infers:
            processed_camera = set()
            processed_camera.add(infer.CameraID)
            if infer in processed:
                continue
            ismatch = 1
            for comp in infers:
                if (comp in processed) or (comp.CameraID in processed_camera):
                    continue
                if infer.category == comp.category:
                    ismatch += 1
                    processed.add(comp)
                    processed_camera.add(comp.CameraID)
            if ismatch >= match_threshold:
                numMatch += 1
            processed.add(infer)
            
    return numMatch
       
def Error_of_All_Model(dataset):
    '''
    Calculate number of inferences included in error space of all models
    @args: dataset
    @return numErr:int -> Sum of FP and FN included in error space of all models
    '''
    numModel = len(dataset)
    FP = extract_infer(dataset, 'FP')
    FN = extract_infer(dataset, 'FN')
    
    numFP = count_matches(FP, match_threshold=numModel)
    numFN = count_matches(FN, match_threshold=numModel)

    return numFP + numFN

def Error_of_Each_Model(dataset):
    '''
    Calculate number of inferences included in error space of any model
    @args: dataset
    @return numErr:int -> Sum of FP and FN included in error space of any model
    '''
    FP = extract_infer(dataset, 'FP')
    FN = extract_infer(dataset, 'FN')
    
    numModel = len(dataset)
    
    numFP = FP_of_Each_Model(dataset)
    numFN = FN_of_Each_Model(dataset)
    
    return numFP + numFN

def TP_of_All_Model(dataset):
    '''
    Calculate number of True Positives for all models
    @args: dataset
    @return numTP:int -> Number of TPs for all models
    '''
    numModel = len(dataset)

    TP = extract_infer(dataset, 'TP')

    numTP = count_matches(TP, match_threshold=numModel)

    return numTP

def TP_of_Each_Model(dataset):
    '''
    画像内のいずれかのモデルのTPの数を計算する
    @args: dataset
    @return numTP:int -> 画像内のいずれかのモデルのTPの数
    '''
    numModel = len(dataset)
    TP = extract_infer(dataset, 'TP')
    
    numTP = count_matches(TP, match_threshold=1)
    
    return numTP

def TN_of_Each_Model(dataset):
    '''
    Calculate number of True Positives for any model
    @args: dataset
    @return numTP:int -> Number of TPs for any model
    '''
    numModel = len(dataset)
    FP = extract_infer(dataset, 'FP')
    FN = extract_infer(dataset, 'FN')

    numTN = 0
    processed_FP = set() 
    for infer in FP:
        if infer in processed_FP:
            continue
        boxA = [infer.X_coordinate, infer.Y_coordinate, infer.width, infer.height]
        ismatch = 1
        for comp in FP:
            if (comp in processed_FP) or (infer.CameraID == comp.CameraID):
                continue
            boxB = [comp.X_coordinate, comp.Y_coordinate, comp.width, comp.height]
            if IoU(boxA, boxB) > IoU_Threshold:
                ismatch += 1
                processed_FP.add(comp)
        if ismatch < numModel:
            numTN += 1
        processed_FP.add(infer)
    
    processed_FN = set()
    for infer in FN:
        if infer in processed_FN:
            continue
        ismatch = 1
        ctg = infer.category
        processed_camera = set()
        processed_camera.add(infer.CameraID)
        for comp in FN:
            if (infer == comp) or (comp in processed_FN) or (comp.CameraID in processed_camera):
                continue
            if comp.category == ctg:
                ismatch += 1
                processed_FN.add(comp)
                processed_camera.add(comp.CameraID)
        if ismatch < numModel:
            numTN += 1
        processed_FN.add(infer)
    
    return numTN
 
def FP_of_All_Model(dataset):
    '''
    Calculate number of False Positives for all models
    @args: dataset
    @return numFP:int -> Number of FPs for all models
    '''
    numModel = len(dataset)
    FP = extract_infer(dataset, 'FP')
    
    numFP = count_matches(FP, match_threshold=numModel)

    return numFP     

def FP_of_Each_Model(dataset):
    '''
    Calculate number of False Positives for any model
    @args: dataset
    @return numFP:int -> Number of FPs for any model
    '''
    FP = extract_infer(dataset, 'FP')
    numFP = count_matches(FP, match_threshold=1)
    
    return numFP

def FN_of_All_Model(dataset):
    '''
    Calculate number of False Negatives for all models
    @args: dataset
    @return numFN:int -> Number of FNs for all models
    '''
    numModel = len(dataset)
    FN = extract_infer(dataset, 'FN')
    
    numFN = count_matches(FN, match_threshold=numModel)

    return numFN

def FN_of_Each_Model(dataset):
    '''
    Calculate number of False Negatives for any model
    @args: dataset
    @return numFN:int -> Number of FNs for any model
    '''
    FN = extract_infer(dataset, 'FN')
        
    numFN = count_matches(FN, match_threshold=1)

    return numFN

def Objects_of_All_Model(dataset) -> int:
    """
    Calculate total number of inferences in an image
    @args: dataset
    @return numObj:int -> Total number of inferences in an image
    """
    return TP_of_All_Model(dataset) + Error_of_Each_Model(dataset)
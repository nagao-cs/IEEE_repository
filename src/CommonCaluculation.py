IoU_Threshold = 0.5

# boxA=[xmin, ymin, width, height]
def IoU(boxA, boxB):
    '''
    二つのbboxのIoUを計算する
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
    特定の推論を抽出する
    @args: dataset, mode
    @return infer:list -> 特定の推論
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
    一つの推論に対して、一致する推論の数を数える
    @args: infer, match_threshold
    @return numMatch:int -> 一致する推論の数
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
    すべての物体検出モデルのエラー空間に含まれる推論の数
    @args: dataset
    @return numErr:int -> すべてのモデルのエラー空間に含まれるFPとFNの和
    '''
    numModel = len(dataset)
    FP = extract_infer(dataset, 'FP')
    FN = extract_infer(dataset, 'FN')
    
    numFP = count_matches(FP, match_threshold=numModel)
    numFN = count_matches(FN, match_threshold=numModel)

    return numFP + numFN

def Error_of_Each_Model(dataset):
    '''
    画像内のいずれかのモデルのエラー空間に含まれる推論の数
    @args: dataset
    @return numErr:int -> 画像内のいずれかのモデルのエラー空間に含まれるFPとFNの和
    '''
    FP = extract_infer(dataset, 'FP')
    FN = extract_infer(dataset, 'FN')
    
    numModel = len(dataset)
    
    numFP = FP_of_Each_Model(dataset)
    numFN = FN_of_Each_Model(dataset)
    
    return numFP + numFN

def TP_of_All_Model(dataset):
    '''
    すべてのモデルのTPの数を計算する
    @args: dataset
    @return numTP:int -> すべてのモデルのTPの数
    '''
    numModel = len(dataset)

    # TPに該当する推論を収集
    TP = extract_infer(dataset, 'TP')

    # すべてのモデルでTPとなった推論を数える
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
    画像内のいずれかのモデルのTNの数を計算する
    @args: dataset
    @return numTN:int -> 画像内のいずれかのモデルのTNの数
    '''
    numModel = len(dataset)
    FP = extract_infer(dataset, 'FP')
    FN = extract_infer(dataset, 'FN')

    numTN = 0
    processed_FP = set()  # 処理済みTNの追跡
    for infer in FP:
        if infer in processed_FP:
            continue  # 既に処理済みならスキップ
        boxA = [infer.X_coordinate, infer.Y_coordinate, infer.width, infer.height]
        ismatch = 1  # 同じ位置の一致数
        for comp in FP:
            if (comp in processed_FP) or (infer.CameraID == comp.CameraID):
                continue  # 同一物体または既処理物体をスキップ
            boxB = [comp.X_coordinate, comp.Y_coordinate, comp.width, comp.height]
            if IoU(boxA, boxB) > IoU_Threshold:
                ismatch += 1
                processed_FP.add(comp)
        if ismatch < numModel:
            numTN += 1
        processed_FP.add(infer)
    
    processed_FN = set()  # 処理済みFNの追跡
    for infer in FN:
        if infer in processed_FN:
            continue  # 既に処理済みならスキップ
        ismatch = 1  # 同じカテゴリの一致数
        ctg = infer.category
        processed_camera = set()
        processed_camera.add(infer.CameraID)
        for comp in FN:
            if (infer == comp) or (comp in processed_FN) or (comp.CameraID in processed_camera):
                continue  # 同一物体または既処理物体をスキップ
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
    すべてのモデルのFPの数を計算する
    @args: dataset
    @return numFP:int -> すべてのモデルのFPの数
    '''
    numModel = len(dataset)
    FP = extract_infer(dataset, 'FP')
    
    numFP = count_matches(FP, match_threshold=numModel)

    return numFP     

def FP_of_Each_Model(dataset):
    '''
    画像内のいずれかのモデルのFPの数を計算する
    @args: dataset
    @return numFP:int -> 画像内のいずれかのモデルのFPの数
    '''
    FP = extract_infer(dataset, 'FP')
    numFP = count_matches(FP, match_threshold=1)
    
    return numFP

def FN_of_All_Model(dataset):
    '''
    すべてのモデルのFNの数を計算する
    @args: dataset
    @return numFN:int -> すべてのモデルのFNの数
    '''
    numModel = len(dataset)
    FN = extract_infer(dataset, 'FN')
    
    numFN = count_matches(FN, match_threshold=numModel)

    return numFN

def FN_of_Each_Model(dataset):
    '''
    画像内のいずれかのモデルのFNの数を計算する
    @args: dataset
    @return numFN:int -> 画像内のいずれかのモデルのFNの数
    '''
    FN = extract_infer(dataset, 'FN')
        
    numFN = count_matches(FN, match_threshold=1)

    return numFN

def Objects_of_All_Model(dataset) -> int:
    """
    ある画像内のすべての推論の数を計算する
    @args: dataset
    @return numObj:int -> ある画像内のすべての推論の数
    """
    return TP_of_All_Model(dataset) + Error_of_Each_Model(dataset)
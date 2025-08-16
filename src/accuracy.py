from CommonCaluculation import IoU, IoU_Threshold
from Ensemble import Ensemble
from CommonCaluculation import *

'''
@param: dataset, MV_Threshold -> データセットといくつ以上の推論で推論結果に追加するかの閾値
@return: numTP,numFP,numFN
複数モデルの推論を多数決で統合する
'''
def MajorityVoting_TP(dataset, MV_Threshold) -> int:
    """
    TPに該当する推論を多数決で統合する。
    @param dataset: 各モデルの推論結果を格納したリスト
    @param MV_Threshold: int - 多数決で採用するモデル数の閾値
    @return numTP: int - 閾値を満たしたTPの数
    """
    # datasetからTPにあたる推論をすべて収集
    TP = list()
    for model in dataset:
        for infer in model:
            if infer.isTrue is True:
                TP.append(infer)

    numTP = 0
    processed = set()
    numModel = len(dataset)

    # TPの内、閾値以上のモデルが同じ推論をした数をカウント
    while TP:
        infer = TP.pop(0)
        processed_camera = set()
        processed_camera.add(infer.CameraID)
        if infer in processed:
            continue  # 処理済みの推論はスキップ

        numMatch = 1
        boxA = [infer.X_coordinate, infer.Y_coordinate, infer.width, infer.height]

        for comp in TP[:]:  # コピーを使うことで安全に削除可能
            if (infer.CameraID == comp.CameraID) or (comp in processed) or (comp.CameraID in processed_camera):
                continue  # 同じモデルの推論または処理済みの場合はスキップ

            boxB = [comp.X_coordinate, comp.Y_coordinate, comp.width, comp.height]
            if (IoU(boxA, boxB) > IoU_Threshold) and (infer.category == comp.category):
                numMatch += 1
                processed.add(comp)  # 同じ推論を処理済みに追加
                processed_camera.add(comp.CameraID)
                TP.remove(comp)  # リストから削除
                if numMatch == numModel:
                    break

        processed.add(infer)  # 現在の推論も処理済みに追加

        if numMatch >= MV_Threshold:
            numTP += 1

    return numTP

def MajorityVoting_FP(dataset, MV_Threshold) -> int:
    """
    FPに該当する推論を多数決で統合する。
    @param dataset: 各モデルの推論結果を格納したリスト
    @param MV_Threshold: int - 多数決で採用するモデル数の閾値
    @return numFP: int - 閾値を満たしたFPの数
    """
    # datasetからFPにあたる推論をすべて収集
    FP = list()
    for model in dataset:
        for infer in model:
            if infer.conf > 0 and infer.isTrue is False:
                FP.append(infer)

    numFP = 0
    processed = set()

    # FPの内、閾値以上のモデルが同じ推論をした数をカウント
    while FP:
        infer = FP.pop(0)
        if infer in processed:
            continue  # 処理済みの推論はスキップ

        numMatch = 1
        boxA = [infer.X_coordinate, infer.Y_coordinate, infer.width, infer.height]

        for comp in FP[:]:  # コピーを使って安全に削除
            if (infer.CameraID == comp.CameraID) or (comp in processed):
                continue  # 同じモデルの推論または処理済みの場合はスキップ

            boxB = [comp.X_coordinate, comp.Y_coordinate, comp.width, comp.height]
            if IoU(boxA, boxB) > IoU_Threshold:
                processed.add(comp)  # 同じ推論を処理済みに追加
                numMatch += 1
                FP.remove(comp)  # リストから削除

        processed.add(infer)  # 現在の推論も処理済みに追加

        if numMatch >= MV_Threshold:
            numFP += 1

    return numFP

def MajorityVoting_FN(dataset, MV_Threshold):
    """
    モデル間で共通するFNを多数決で決定。
    @param dataset: モデル出力のリスト
    @param MV_Threshold: 多数決の閾値
    @return numFN: モデル間で共通するFNの数
    """
    numModel = len(dataset)
    FN = list()

    # モデルの出力からFNを収集
    for model in dataset:
        for infer in model:
            if infer.conf == 0:
                FN.append(infer)

    numFN = 0
    processed = set()
    for infer in FN:
        if infer in processed:
            continue
        numMatch = 1  # 最初の物体自身をカウント
        for comp in FN:
            if (infer == comp) or (comp in processed) or (infer.CameraID == comp.CameraID):
                continue
            # IoUとカテゴリの一致で判断
            if infer.category == comp.category:
                numMatch += 1
                processed.add(comp)
            # if numMatch == numModel:
            #     break  # 全モデルでFNと一致した場合は打ち切り
        processed.add(infer)
        # 閾値以上のモデルで一致した場合にFNとしてカウント
        if numMatch >= numModel - MV_Threshold + 1:
            numFN += 1

    return numFN
 
'''
データセットを受け取り、mvをした後全体の正解率(acc)を返す
@param: datasets
@return: acc:float
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
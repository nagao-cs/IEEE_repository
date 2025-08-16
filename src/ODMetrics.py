from CommonCaluculation import *
from MakeDataset import *
from accuracy import *

def Cov(datasets):
    def cov_true_dataset(dataset):
        def is_true_image(image):
            for obj in image:
                if obj.isTrue == False:
                    return False
            return True
        for detectresult in dataset:
            if is_true_image(detectresult) == True:
                return True
        return False
    cov_count = 0
    for dataset in datasets:
        if cov_true_dataset(dataset) == False:
            cov_count += 1            
    return 1 - cov_count/len(datasets)

def OD_Cov(datasets):
    '''
    OD_Covを計算する
    @param datasets:list
    @return OD-Cov:float
    '''
    def ODCov_Image(dataset):
        '''
        画像内のすべての推論の数に対する、すべてのモデルでエラーとなった推論の割合
        @args dataset
        @return float
        '''
        numObj = Objects_of_All_Model(dataset) # 画像内のすべての推論の数 
        if numObj == 0:
            return 0
        numErr = Error_of_All_Model(dataset) # すべてのモデルのエラー空間に含まれる推論の数
        # print(numErr)
        return numErr/numObj
    I = 0
    num_image = len(datasets)
    for dataset in datasets:
        I += ODCov_Image(dataset)
    return 1 - (I / num_image)


def Cer(datasets):
    def cer_true_dataset(dataset):
        def is_true_image(image):
            for obj in image:
                if obj.isTrue == False:
                    return False
            return True
        for detectresult in dataset:
            if is_true_image(detectresult) == False:
                return False
        return True
    cer_count = 0
    for dataset in datasets:
        if cer_true_dataset(dataset) == False:
            cer_count += 1
    return 1 - cer_count/len(datasets)

def OD_Cer(datasets):
    '''
    OD_Cerを計算する
    @args datasets
    @return OD_Cer:float
    '''
    def ODCer_Image(dataset):
        '''
        画像内のすべての推論の数に対する、いずれかのモデルでエラーとなった推論の割合
        '''
        numObj = Objects_of_All_Model(dataset)
        if numObj == 0:
            return 0
        numErr = Error_of_Each_Model(dataset)
        return numErr/numObj
    U = 0
    for dataset in datasets:
        U += ODCer_Image(dataset)
    return 1 - (U / len(datasets))

if __name__ == "__main__":
    Common_Directory_path = r'./dataset'
    Datasets = list()
    paths = [
        [
        Common_Directory_path+'/center'
        ],
        [
        Common_Directory_path+'/center',
        Common_Directory_path+'/left_1',
        ],
        [
        Common_Directory_path+'/center',
        Common_Directory_path+'/left_1',
        Common_Directory_path+'/right_1'
        ],
        [
        Common_Directory_path+'/center', 
        Common_Directory_path+'/left_1',
        Common_Directory_path+'/right_1',
        Common_Directory_path+'/left_2',
        ],
        [
        Common_Directory_path+'/center', 
        Common_Directory_path+'/left_1',
        Common_Directory_path+'/right_1',
        Common_Directory_path+'/left_2',
        Common_Directory_path+'/right_2'
        ],
        [
        Common_Directory_path+'/center', 
        Common_Directory_path+'/left_1',
        Common_Directory_path+'/right_1',
        Common_Directory_path+'/left_2',
        Common_Directory_path+'/right_2',
        Common_Directory_path+'/left_3'
        ],
        [
        Common_Directory_path+'/center', 
        Common_Directory_path+'/left_1',
        Common_Directory_path+'/right_1',
        Common_Directory_path+'/left_2',
        Common_Directory_path+'/right_2',
        Common_Directory_path+'/left_3',
        Common_Directory_path+'/right_3'
        ]
    ]
        
    for path in paths:
        Datasets = MakeDataset(path)
        numCamera = len(Datasets[0])
        if numCamera % 2 == 1:
            MV_Threshold = numCamera // 2 + 1
        else:
            MV_Threshold = numCamera // 2 + 1
        
        print(f"{numCamera} version")
        print(f"    Cov:{Cov(Datasets)}")
        print(f"    Cer:{Cer(Datasets)}")
        print(f"    Cov_OD:{OD_Cov(Datasets)}")
        print(f"    Cer_OD:{OD_Cer(Datasets)}")
        print(f"    accuracy")
        print(f"        affirmatve:{accuracy(Datasets, 'affirmative')}")
        print(f"        consensus:{accuracy(Datasets, "consensus")}")
        print(f"        unanimous{accuracy(Datasets, "unanimous")}")
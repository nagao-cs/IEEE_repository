from MakeDataset import MakeDataset
from ODMetrics import *
from accuracy import accuracy
import itertools
import os
from matplotlib import pyplot as plt
import numpy as np

def bestCombination(paths, numConb):
    # 各指標のすべての組み合わせとスコアを記録する辞書
    rankings = {
        'Cov': [], 'Cer': [], 
        'OD-Cov': [], 'OD-Cer': [],
    }
    
    # 7つのパスからnumCombの組み合わせを生成
    for combination in itertools.combinations(paths, numConb):
        datasets = MakeDataset(combination)
        
        # 各指標の計算
        scores = {
            'Cov': Cov(datasets),
            'Cer': Cer(datasets),
            'OD-Cov': OD_Cov(datasets),
            'OD-Cer': OD_Cer(datasets),
        }
        
        # 各指標について、組み合わせとスコアを記録
        for key, score in scores.items():
            rankings[key].append({
                'combination': combination,
                'score': score
            })
    
    # 各指標についてスコアでソートし、上位10件を取得
    top_10_rankings = {}
    for key in rankings.keys():
        sorted_combinations = sorted(rankings[key], 
                                  key=lambda x: x['score'], 
                                  reverse=True)[:10]
        top_10_rankings[key] = sorted_combinations
    
    return top_10_rankings

if __name__ == '__main__':
    Common_Directory_path = r'./dataset'
    paths = [
        Common_Directory_path+'/left_3',
        Common_Directory_path+'/left_2',
        Common_Directory_path+'/left_1',
        Common_Directory_path+'/center', 
        Common_Directory_path+'/right_1',
        Common_Directory_path+'/right_2',
        Common_Directory_path+'/right_3'
    ]
    
    numConb = 3
    rankings = bestCombination(paths, numConb)
    
    # 結果の表示
    for metric, top_10 in rankings.items():
        print(f"\n{metric}のTop 10:")
        for i, result in enumerate(top_10, 1):
            combination = [os.path.basename(path) for path in result['combination']]
            print(f"{i}位: スコア: {result['score']:.3f}, 組み合わせ: {combination}")
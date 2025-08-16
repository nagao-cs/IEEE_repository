from MakeDataset import MakeDataset
from ODMetrics import *
from accuracy import accuracy
import itertools
import os
from matplotlib import pyplot as plt
import numpy as np

def bestCombination(paths, numConb):
    # Dictionary to store combinations and scores for each metric
    rankings = {
        'Cov': [], 'Cer': [], 
        'OD-Cov': [], 'OD-Cer': [],
    }
    
    # Generate combinations of numConb cameras from paths
    for combination in itertools.combinations(paths, numConb):
        datasets = MakeDataset(combination)
        
        # Calculate metrics
        scores = {
            'Cov': Cov(datasets),
            'Cer': Cer(datasets),
            'OD-Cov': OD_Cov(datasets),
            'OD-Cer': OD_Cer(datasets),
        }
        
        # Record combinations and scores for each metric
        for key, score in scores.items():
            rankings[key].append({
                'combination': combination,
                'score': score
            })
    
    # Sort by score and get top 10 for each metric
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
    
    # Display results
    for metric, top_10 in rankings.items():
        print(f"\nTop 10 for {metric}:")
        for i, result in enumerate(top_10, 1):
            combination = [os.path.basename(path) for path in result['combination']]
            print(f"Rank {i}: Score: {result['score']:.3f}, Combination: {combination}")
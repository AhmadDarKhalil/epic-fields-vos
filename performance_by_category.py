#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 17:57:48 2022

@author: ru20956
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 16:28:06 2022

@author: ru20956
"""
import os
import json
import glob
from  math import log
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt, colors
import pandas as pd

def get_scores_dict(objects_mapping_file='gt_val_data_mapping_inv.json', excel_file='outputs/fixed2d/per-sequence_results-val.csv'):
    # Load the JSON file
    with open(objects_mapping_file, 'r') as f:
        mapping = json.load(f)
        
    with open('sequences.json', 'r') as f:
        included_sequences = json.load(f)

    # Read the CSV file instead of the Excel
    df = pd.read_csv(excel_file)

    n1 = 'J-Mean'
    n2 = 'F-Mean'
    
    df = df[df['Sequence'].isin(included_sequences)]
    sequences = df['Sequence'].tolist()
    stm = df[n1].tolist()
    r3d = df[n2].tolist()

    scores = {}
    scores[n1] = {}
    scores[n2] = {}

    for i, seq in enumerate(sequences):
        obj_color = seq.split('_')[-1]
        seq_name = '_'.join(seq.split('_')[:-1])
        if not mapping[seq_name][obj_color] in scores[n1].keys():
            scores[n1][mapping[seq_name][obj_color]] = []
        if not mapping[seq_name][obj_color] in scores[n2].keys():
            scores[n2][mapping[seq_name][obj_color]] = []

        scores[n2][mapping[seq_name][obj_color]].append(r3d[i])
        scores[n1][mapping[seq_name][obj_color]].append(stm[i])

    return scores


# Opening JSON file

def preparenoun_map(noun_map):
    List = []
    for idx, each in enumerate(noun_map):
        subList = []
        tmp = each.split(',')
        for noun in tmp:
            noun = eval(noun.strip().strip('[').strip(']').strip("\\"))
            if ':' in noun:
                noun = transfer_noun(noun).strip()
            subList.append(noun)
        List.append(subList)
    return List

def transfer_noun(noun):
    List = noun.split(':')
    return ' '.join(List[1:]) + ' ' + List[0]


def get_entities_index(noun_classes, noun):
    idx = -1

    for i, each in enumerate(noun_classes):
        if noun in each:
            idx = i
            break
    return idx


static_objects_list = [0,3,12,24,36,63,42,46,70,90,110,113,124,135,159,179]
strong_static = [24,42,63,110,135,159,179]
def calculate_category_score(obj_scores,mapping_file,categories_file):
    n1 = 'J-Mean'
    n2 = 'F-Mean'
    static = [24,42,63,110,135,159,179,0,3,12,24,36,63,42,46,70,90,110,113,124,135,159,179]
    static_stm = []
    static_3d = []
    all_data = {}
    fm = open(mapping_file)
    mapping_data = json.load(fm)
    noun_classes = preparenoun_map(pd.read_csv(categories_file)['instances'])

    data = obj_scores
    count=[0 for _ in range(305)]
    j=[0 for _ in range(305)]
    f=[0 for _ in range(305)]
    j_and_f=[0 for _ in range(305)]
    for seq_object_id in data[n1].keys():
        object_id = seq_object_id.split('_')[-1]
        seq_name = '_'.join(seq_object_id.split('_')[:-1])
        noun = seq_object_id
        
        noun_idx = get_entities_index(noun_classes, noun)
        if noun_idx!= -1:

            j[noun_idx] +=sum(data[n1][seq_object_id])/len(data[n1][seq_object_id])
            f[noun_idx] +=sum(data[n2][seq_object_id])/len(data[n2][seq_object_id])
            count[noun_idx] += len(data[n1][seq_object_id])
            if  not noun_idx in static:
                for i in range(0,len(data[n1][seq_object_id])):
                    static_stm.append(sum(data[n1][seq_object_id])/len(data[n1][seq_object_id]))
                    static_3d.append(sum(data[n2][seq_object_id])/len(data[n2][seq_object_id]))

        else:
            print('Object out of EPIC classes!!!!')
            os.exit(0)
    
    print('len: ', len(static_stm))
    print('j&f: ',(sum(static_stm)/len(static_stm) + sum(static_3d)/len(static_3d)) /2)
    print('j: ', sum(static_stm)/len(static_stm))
    print('f: ', sum(static_3d)/len(static_3d))  
  

            
            
mapping_file = 'gt_val_data_mapping.json'
categories_file = 'EPIC_100_noun_classes_v2.csv'
per_object_scores = get_scores_dict(excel_file='outputs/fixed3d/per-sequence_results-val.csv')    
calculate_category_score(per_object_scores,mapping_file,categories_file) 

       
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
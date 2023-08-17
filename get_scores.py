import pandas as pd
import json
import os
import glob
from math import log
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt, colors

# Helper Functions
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

def calculate_category_score(file, mapping_file, categories_file):
    # function contents

    # Load the mapping
    json_file = 'gt_val_data_mapping_inv.json'
    with open(json_file, 'r') as f:
        mapping = json.load(f)

    # Load the Excel data
    excel_file = 'results.xlsx'
    df = pd.read_excel(excel_file)
    n1 = 'xmem_fine_j'
    n2 = 'xmem_fine_f'
    sequences = df['Sequence'].tolist()
    stm = df[n1].tolist()
    r3d = df[n2].tolist()

    scores = {}
    scores[n1] = {}
    scores[n2] = {}
    for i,seq in enumerate(sequences):
        obj_color = seq.split('_')[-1]
        seq_name = '_'.join(seq.split('_')[:-1])
        if not mapping[seq_name][obj_color] in scores[n1].keys():
            scores[n1][mapping[seq_name][obj_color]] = []
        if not mapping[seq_name][obj_color] in scores[n2].keys():
            scores[n2][mapping[seq_name][obj_color]] = []

        scores[n2][mapping[seq_name][obj_color]].append(r3d[i])
        scores[n1][mapping[seq_name][obj_color]].append(stm[i])

    with open('scores.json', 'w') as f:
        json.dump(scores, f)

    for obj in scores[n2].keys():
        stm_avg = sum(scores[n1][obj])/len(scores[n1][obj])
        r3d_avg = sum(scores[n2][obj])/len(scores[n2][obj])

        if r3d_avg + 0.2 > stm_avg:
            print(f"{obj}, len: {len(scores[n2][obj])}")
        if obj == 'door':
            print(f"stm: {stm_avg}, 3d: {r3d_avg}")

# Call the function
mapping_file = 'gt_val_data_mapping.json'
categories_file = 'EPIC_100_noun_classes_v2.csv'
file = 'scores.json'
calculate_category_score(file, mapping_file, categories_file)

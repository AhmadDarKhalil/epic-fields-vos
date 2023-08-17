#gt_val_data_mapping.json
import pandas as pd
import json

json_file = 'gt_val_data_mapping_inv.json'
with open(json_file, 'r') as f:
    mapping = json.load(f)
'''
n = {}
for k,v in mapping.items():
    kk = {}
    for kk1,vv1 in v.items():
        kk[vv1] = kk1
    n[k] = kk
with open('gt_val_data_mapping_inv.json', 'w') as f:
    json.dump(n, f)
'''

# Read the Excel file
excel_file = 'results.xlsx'
df = pd.read_excel(excel_file)
n1 = 'xmem_fine_j'
n2= 'xmem_fine_f'
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


    #print(mapping[seq_name][obj_color])
    scores[n2][mapping[seq_name][obj_color]].append(r3d[i])
    scores[n1][mapping[seq_name][obj_color]].append(stm[i])


    #print(f'{i}, {seq}')
with open('scores.json', 'w') as f:
    json.dump(scores, f)

for obj in scores[n2].keys():
    stm_avg = sum(scores[n1][obj])/len(scores[n1][obj])
    r3d_avg = sum(scores[n2][obj])/len(scores[n2][obj])

    if r3d_avg+0.2 > stm_avg:
        print(f"{obj}, len: {len(scores[n2][obj])}")
    if obj == 'door':
        print(f"stm: {stm_avg}, 3d: {r3d_avg}")
        #break

# epic-fields-vos
Baselines for semi-supervised VOS (EPIC FIELDS)



## Fixed2D Baseline

The `fixed_2d.py` runs the **Fixed2D** baseline. The script take these parameters:

### Parameters:

- `--output-folder`: Specifies the directory where the processed sequences will be saved. The default is `outputs/fixed2d`.

- `--visor-root`: Points to the root directory of the VISOR dataset. The default path is `datasets/VISOR_2022`.

The results will be stored under <output-folder>  directory. 


## Fixed3D Baseline

The `fixed_3d.py` runs the **Fixed3D** baseline. The script take these parameters:

### Parameters:

- `--output-folder`: Specifies the directory where the processed sequences will be saved. The default is `outputs/fixed3d`.

- `--visor-root`: Points to the root directory of the VISOR dataset. The default path is `datasets/VISOR_2022`.

- `--dense-models-root`: Indicates the directory containing the EPIC FIELDS dense models. 

- `--visor-to-epic`: Specifies the path to the file that maps any VISOR frame to its corresponding EPIC-KITCHENS frame. By default, it's set to `frame_mapping_train_val.json` (included as part of the repo).

- `--epic-to-visor`: Determines the path to the file that maps the EPIC-KITCHENS frames to VISOR frames. The default file is `frame_mapping_train_val_epic_to_visor.json`(included as part of the repo).

The results will be stored under <output-folder>  directory. 

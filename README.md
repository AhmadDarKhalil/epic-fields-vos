# Epic-fields-vos
This repo contains the baselines for semi-supervised VOS (EPIC FIELDS) and how to use replicate the results reported in the paper.

## Dataset Structure
To run the training or evaluation scripts, the dataset format should be as follows (following [DAVIS](https://davischallenge.org/) format), a script is given in the next step to convert VISOR to DAVIS-like dataset.
```

|- VISOR_2022
  |- val_data_mapping.json
  |- train_data_mapping.json
  |- JPEGImages
  |- Annotations
  |- ImageSets
     |- 2022
        |- train.txt
        |- val.txt
        |- val_unseen.txt
```
For more information on how to get such a format, please visit [VISOR-VOS repository](https://github.com/epic-kitchens/VISOR-VOS)

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

## Evaluation
In order to evaluate your semi-supervised method in VISOR we use [VISOR evaluation script](https://github.com/epic-kitchens/VISOR-VOS/tree/b6afe07691fad922ebacb42654046ef24a0fe3ba/evaldavis2017), execute the following command substituting `results/sample` by the folder path that contains your results:
```bash
python evaluation_method.py --task semi-supervised --results_path results/sample
```


## Citation
If you find this work useful please cite our paper:

```
    @article{EPICFIELDS2023,
           title={{EPIC-FIELDS}: {M}arrying {3D} {G}eometry and {V}ideo {U}nderstanding},
           author={Tschernezki, Vadim and Darkhalil, Ahmad and Zhu, Zhifan and Fouhey, David and Larina, Iro and Larlus, Diane and Damen, Dima and Vedaldi, Andrea},
           booktitle   = {ArXiv},
           year      = {2023}
    } 
```

Also cite the [EPIC-KITCHENS-100](https://epic-kitchens.github.io) paper where the videos originate:

```
@ARTICLE{Damen2022RESCALING,
           title={Rescaling Egocentric Vision: Collection, Pipeline and Challenges for EPIC-KITCHENS-100},
           author={Damen, Dima and Doughty, Hazel and Farinella, Giovanni Maria  and and Furnari, Antonino 
           and Ma, Jian and Kazakos, Evangelos and Moltisanti, Davide and Munro, Jonathan 
           and Perrett, Toby and Price, Will and Wray, Michael},
           journal   = {International Journal of Computer Vision (IJCV)},
           year      = {2022},
           volume = {130},
           pages = {33–55},
           Url       = {https://doi.org/10.1007/s11263-021-01531-2}
} 
```
For more information on the project and related research, please visit the [EPIC-Kitchens' EPIC Fields page](https://epic-kitchens.github.io/epic-fields/).


## License
All files in this dataset are copyright by us and published under the 
Creative Commons Attribution-NonCommerial 4.0 International License, found 
[here](https://creativecommons.org/licenses/by-nc/4.0/).
This means that you must give appropriate credit, provide a link to the license,
and indicate if changes were made. You may do so in any reasonable manner,
but not in any way that suggests the licensor endorses you or your use. You
may not use the material for commercial purposes.

## Contact

For general enquiries regarding this work or related projects, feel free to email us at [uob-epic-kitchens@bristol.ac.uk](mailto:uob-epic-kitchens@bristol.ac.uk).

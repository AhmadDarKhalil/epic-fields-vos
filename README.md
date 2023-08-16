# epic-fields-vos
Baselines for semi-supervised VOS (EPIC FIELDS)



## Fixed2D Baseline

The `fixed2d.py` script offers a method to create a static 2D baseline for video sequences. Essentially, it duplicates the first frame of each sequence from the VISOR dataset across the entire sequence.

#### Parameters:

- **--output-folder**: Specifies the directory where the processed sequences will be saved. The default is `outputs/fixed2d`.

- **--visor-root**: Points to the root directory of the VISOR dataset. The default path is `datasets/VISOR_2022`.


### Usage
Run the script using the command:
'''
python fixed2d.py --output-folder <output_folder_path> --visor-root <visor_root_path>
'''
This will generate results under <output_folder_path>  directory. 

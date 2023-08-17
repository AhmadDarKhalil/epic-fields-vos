import os
import glob
import shutil
import numpy as np
import argparse
from tqdm import tqdm

davis_palette = np.repeat(np.expand_dims(np.arange(0, 256), 1), 3, 1).astype(np.uint8)
davis_palette[:22, :] = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                            [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                            [64, 0, 0], [191, 0, 0], [64, 128, 0], [191, 128, 0],
                            [64, 0, 128], [191, 0, 128], [64, 128, 128], [191, 128, 128],
                            [0, 64, 0], [128, 64, 0], [0, 191, 0], [128, 191, 0],
                            [0, 64, 128], [128, 64, 128]]


def read_text(text_file):
    sequences = []
    with open(text_file, "r") as file:
        for line in file:
            clean_line = line.strip()  # Remove leading/trailing whitespaces and newline characters
            if clean_line:  # Skip empty lines
                sequences.append(clean_line)
    return sequences
            
def copy_first_image_to_all(root_folder,visor_root,val_sequences):
    os.makedirs(root_folder,exist_ok=True)
    sequences = read_text(val_sequences)
    for seq in tqdm(sequences):
        first = sorted(glob.glob(os.path.join(visor_root,seq,'*.png')))[0]
        os.makedirs(os.path.join(root_folder,seq),exist_ok=True)
        for image_path in glob.glob(os.path.join(visor_root,seq,'*.png')):
            shutil.copy(first,image_path.replace(visor_root,root_folder))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fixed2D script [EPIC FIELDS]")
    parser.add_argument("--output-folder", default="outputs/fixed2d", help="Output folder path")
    parser.add_argument("--visor-root", default="/home/skynet/Ahmad/datasets/VISOR_2022", help="VISOR root folder path")
    args = parser.parse_args()

    visor_masks = os.path.join(args.visor_root, 'Annotations/480p')
    val_sequences = os.path.join(args.visor_root,'ImageSets/2022/val.txt')

    # run the fixed2d
    copy_first_image_to_all(args.output_folder, visor_masks, val_sequences)

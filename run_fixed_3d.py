import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from read_write_model import read_model
from PIL import Image
import glob
from tqdm import tqdm
import shutil
import json
from scipy.ndimage.morphology import binary_dilation

class Model:
    def __init__(self):
        self.cameras = []
        self.images = []
        self.points3D = []

    def read_model(self, path, ext=""):
        self.cameras, self.images, self.points3D = read_model(path, ext)



def parse_args():
    parser = argparse.ArgumentParser(description="Fixed3D script [EPIC FIELDS]")
    parser.add_argument("--output-folder", default="outputs/fixed3d", help="Output folder path")
    parser.add_argument("--visor-root", default="/home/skynet/Ahmad/datasets/VISOR_2022", help="VISOR root folder path")
    parser.add_argument("--dense-models-root", default="/home/skynet/Ahmad/Zhifan_visualizer/build_epic_kitchens_3dmap/colmap_models_registered", help="EPIC FIELDS dense models root")
    parser.add_argument("--visor-to-epic", default="frame_mapping_train_val.json", help="path to the file that maps any VISOR frame to its EPIC-KITCHENS frame")
    parser.add_argument("--epic-to-visor", default="frame_mapping_train_val_epic_to_visor.json", help="path to the file that maps the EPIC-KITCHENS frames to VISOR frames")
    args = parser.parse_args()
    return args

    
def show_images_with_points(src_pts, dst_pts, src_image, dst_image, new_src, new_dst):
    # Create blank images for visualization
    src_vis = src_image.copy()
    dst_vis = dst_image.copy()

    # Convert points to integers
    src_pts = src_pts.astype(int)
    dst_pts = dst_pts.astype(int)

    new_src = new_src.astype(int)
    new_dst = new_dst.astype(int)

    # Draw the matched points in the source image
    #for pt in src_pts:
    #    cv2.circle(src_vis, tuple(pt), 3, (0, 0, 255), -1)
    
    for pt in new_src:
        cv2.circle(src_vis, tuple(pt), 3, (255, 0,0), -1)
    

    # Draw the matched points in the destination image
    #for pt in dst_pts:
    #    cv2.circle(dst_vis, tuple(pt), 3, (0, 0, 255), -1)
    for pt in new_dst:
        cv2.circle(dst_vis, tuple(pt), 3, (255, 0,0), -1)

    return dst_vis

def create_video_from_jpgs(folder_path, output_path, fps=30):
    # Get the list of image file names in the folder
    file_names = sorted([f for f in os.listdir(folder_path) if f.endswith('.jpg')])

    # Read the first image to get the image dimensions
    first_image = cv2.imread(os.path.join(folder_path, file_names[0]))
    height, width, _ = first_image.shape

    # Define the video writer
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Write each image to the video
    for file_name in file_names:
        image_path = os.path.join(folder_path, file_name)
        frame = cv2.imread(image_path)
        video_writer.write(frame)

    # Release the video writer
    video_writer.release()

def sort_images(images):
    sorted_imgs = {}
    #for k,v in images.items():
    m = sorted(images.items(), key=lambda x:x[1].name)
    for m1 in m:
       sorted_imgs[m1[0]] = m1[1] 
    return sorted_imgs
def swap_first_and_second(lst):
    for sublist in lst:
        sublist[0], sublist[1] = sublist[1], sublist[0]

def swap_first_and_second(lst):
    swapped_lst = []
    for sublist in lst:
        swapped_sublist = [sublist[1], sublist[0]] + sublist[2:]
        swapped_lst.append(swapped_sublist)
    return swapped_lst


def save_binary_masks(points_array, file_name):
    # Get the maximum x and y coordinates
    max_x, max_y = 456,256

    points_array = points_array.astype(int)

    # Create a black image with the same shape as the maximum coordinates
    mask = np.zeros((max_y, max_x), dtype=np.uint8)

    # Set the pixel values to 255 at the specified points
    mask[points_array[:, 1], points_array[:, 0]] = 255

    # Save the binary mask as an image file
    cv2.imwrite(file_name, mask)

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
    for seq in sequences:
        first = sorted(glob.glob(os.path.join(visor_root,seq,'*.png')))[0]
        os.makedirs(os.path.join(root_folder,seq),exist_ok=True)
        for image_path in glob.glob(os.path.join(visor_root,seq,'*.png')):
            print(f"{first} to {image_path}")
            shutil.copy(first,image_path.replace(visor_root,root_folder))

def resize_images_in_folders(folder_path):
    # Use glob to retrieve the paths of all PNG images in the subfolders
    image_paths = sorted(glob.glob(os.path.join(folder_path,'*/*png')))

    # Iterate over each image path
    for path in image_paths:
        # Open the image using PIL
        image = Image.open(path)

        # Calculate the new dimensions based on the desired scale
        width = 854
        height = 480
        
        # Resize the image
        resized_image = image.resize((width, height),resample=Image.Resampling.NEAREST)
        resized_image = interpolate_masks_in_image(resized_image)
        # Save the resized image
        resized_image.save(path)

def interpolate_masks_in_image(image, dilation_iterations=1):

    # Convert the image to a NumPy array
    image_array = np.array(image)

    # Create an empty mask array
    mask_array = np.zeros_like(image_array)

    # Get the unique object values in the image
    unique_values = np.unique(image_array)

    # Iterate over each unique object value
    for value in unique_values:
        if value == 0:
            continue

        # Create a binary mask for the current value
        binary_mask = (image_array == value).astype(np.uint8)

        # Dilate the binary mask
        dilated_mask = binary_dilation(binary_mask, iterations=dilation_iterations)

        # Update the mask array with the dilated mask
        mask_array[dilated_mask != 0] = value

    # Create a PIL image from the mask array
    interpolated_image = Image.fromarray(mask_array)
    interpolated_image.putpalette(davis_palette.ravel())
    return interpolated_image

def fill_empty_images(root_folder,visor_root):
    os.makedirs(root_folder,exist_ok=True)
    sequences = [f for f in os.listdir(root_folder) if f.startswith('P')]
    count=0
    for seq in sequences:
        first = sorted(glob.glob(os.path.join(visor_root,seq,'*.png')))[0]
        for file in sorted(glob.glob(os.path.join(visor_root,seq,'*.png'))):
            if not os.path.exists(file.replace(visor_root,root_folder)):
                #put the first frame as a prediction
                shutil.copy(first, file.replace(visor_root,root_folder)) 
                count+=1
    print('count:', count)
    
def main():
    args = parse_args()


    # Opening JSON file
    f = open(args.visor_to_epic)
    
    # returns JSON object as 
    # a dictionary
    mappings = json.load(f)


        # Opening JSON file
    f1 = open(args.epic_to_visor)
    
    # returns JSON object as 
    # a dictionary
    mappings_epic_to_visor = json.load(f1)


    #masks = './vos_codes/baseline_masks'
    root_folder = args.output_folder
    models_path  = args.dense_models_root
    visor_root = args.visor_root
    masks = os.path.join(visor_root,'Annotations/480p')
    val_sequences = os.path.join(visor_root,'ImageSets/2022/val.txt')


    #copy_first_image_to_all(root_folder,masks,val_sequences)

    sequences = read_text(val_sequences)
    
    last_vid = ''
    for seq in tqdm(sorted(sequences)):
        vid = '_'.join(seq.split('_')[:2])
        os.makedirs(os.path.join(root_folder,seq),exist_ok=True)
        if vid in mappings.keys():
            # read COLMAP model
            if vid != last_vid:
                model = Model()
                model.read_model(os.path.join(models_path,f"{vid}_low"), ext='.bin')
                model.images = sort_images(model.images)
            last_vid = vid

            
            first_mask = sorted(glob.glob(os.path.join(masks,seq,'*.png')))[0]
            im1 = Image.open(first_mask)

            first_mask_name = mappings[vid][os.path.basename(first_mask).replace('png','jpg')]
            last_mask = sorted(glob.glob(os.path.join(masks,seq,'*.png')))[-1]
            last_mask_name = mappings[vid][os.path.basename(last_mask).replace('png','jpg')]
            #print(model.images)
            model_images = set()
            

            im1 = im1.resize((456, 256), Image.NEAREST)
            im1_array = np.array(im1)
            
            for color_code in sorted(np.unique(im1_array)):
                last_v = None
                if color_code > 0:
                    
                    if color_code == 1:
                        try:
                            shutil.rmtree(os.path.join(root_folder,seq))
                            os.makedirs(os.path.join(root_folder,seq),exist_ok=True)
                        except:
                            pass
                    
                    im1_xy = np.argwhere(im1_array == color_code)
                    new_p =im1_xy.tolist() #[[372, 23],[100,121]]
                    new_p = swap_first_and_second(new_p)

                    #swap_first_and_second(new_p)
                    for k,v in model.images.items():
                            try: 
                                out_mask = Image.open(os.path.join(root_folder,seq,mappings_epic_to_visor[vid][v.name].replace('jpg','png')))
                                out_mask = np.array(out_mask)
                            except:
                                out_mask = np.zeros((256,456), dtype=np.uint8)
                            #print(v.name)
                            # Create a black image with the same shape as the maximum coordinates
                            if (int(v.name[-14:-4]) >= int(first_mask_name[-14:-4])) and (int(v.name[-14:-4]) <= int(last_mask_name[-14:-4])):
                                    points1 = []
                                    points2 = []
                                    if last_v:
                                        for i in range(0,v.point3D_ids.shape[0]):
                                            if v.point3D_ids[i] != -1:
                                                try:
                                                    index_last = int(np.where(last_v.point3D_ids == v.point3D_ids[i])[0])

                                                except:
                                                    index_last = -1

                                                if index_last > -1:
                                                    points1.append(list(last_v.xys[index_last]))
                                                    points2.append(list(v.xys[i]))

                                        src_pts = np.array(points1, dtype=np.float32)
                                        dst_pts = np.array(points2, dtype=np.float32)
                                        #print(f'shape(scr) {src_pts.shape}, shape(dst): {dst_pts.shape}')
                                        if src_pts.shape[0] >= 10:
                                            #################print(f'v(scr) {last_v.name}, v(dst): {v.name}')
                                            homography = cv2.findHomography(src_pts, dst_pts)[0]
                                            #print(homography)
                                            
                                            # New pixel coordinates in the source image

                                            new_pixel = np.array(new_p, dtype=np.float32)

                                            # Reshape the new_pixel array
                                            new_pixel = np.reshape(new_pixel, (-1, 1, 2))

                                            # Transform the new pixel to the destination image
                                            transformed_pixel = cv2.perspectiveTransform(new_pixel, homography)
                                            # Reshape the transformed_points array
                                            transformed_points = np.reshape(transformed_pixel, (-1, 2))

                                            #convert the points into integers
                                            points_array = transformed_points.astype(int)
                                            points_array = points_array[(points_array[:, 0] < 456) &(points_array[:, 0] >= 0) & (points_array[:, 1] < 256) & ((points_array[:, 1] >= 0))]

                                            
                                            # Set the pixel values to 255 at the specified points
                                            out_mask[points_array[:, 1], points_array[:, 0]] = color_code*1
                                            
                                            
                                            new_p = transformed_pixel
   

                                            model_images.add(v.name)
                                            last_v = v
                                    else:
                                        last_v = v

                                    if v.name in mappings_epic_to_visor[vid]:
                                        ##resized_mask = cv2.resize(out_mask, (854, 480), interpolation=cv2.INTER_NEAREST)
                                        im_pil = Image.fromarray(out_mask, 'P')
                                        im_pil.putpalette(davis_palette.ravel())

                                        im_pil.save(os.path.join(root_folder,seq,mappings_epic_to_visor[vid][v.name].replace('jpg','png')))

    print('Done! Now post-processing the masks (takes few mins)')
    resize_images_in_folders(root_folder)
    fill_empty_images(root_folder, masks)
    
if __name__ == "__main__":
    main()
    
    


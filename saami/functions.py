import math
import os.path
import urllib.request
from PIL import Image
import matplotlib.pyplot as plt
import random
import numpy as np
import pickle
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
from saami.utils import most_prevalent_labels, random_index_with_label

def get_volume_SAM_data(data_dict, sam_checkpoint="models/sam_vit_h_4b8939.pth", sam_model_type= "vit_h", device="cuda", main_axis='z'):

    image = data_dict["image"]
    label = data_dict["label"]
    img_shape = data_dict["image"].shape

    # Currently we use VIT-H model "models/sam_vit_h_4b8939.pth"
    vit_h_url = 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'

    if not os.path.exists(sam_checkpoint):
        print("SAM checkpoint does not exist, downloading the checkpoint under /models folder ...")
        if not os.path.exists('models'):
            os.makedirs('models')
        urllib.request.urlretrieve(vit_h_url, 'models/sam_vit_h_4b8939.pth')


    sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    # predictor = SamPredictor(sam)

    mask_generator = SamAutomaticMaskGenerator(sam)

    # mask_generator = SamAutomaticMaskGenerator(
    #     model=sam,
    #     points_per_side=32,
    #     pred_iou_thresh=0.86,
    #     stability_score_thresh=0.42,
    #     stability_score_offset=0.22,
    #     crop_n_layers=2,
    #     crop_n_points_downscale_factor=2,
    #     min_mask_region_area=00,  # Requires open-cv to run post-processing
    # )

    sam_data = {}
    sam_data["image"] = data_dict["image"]
    sam_data["gt_label"] = data_dict["label"]
    sam_data["sam_seg_x"] = np.zeros(img_shape)
    sam_data["sam_seg_y"] = np.zeros(img_shape)
    sam_data["sam_seg_z"] = np.zeros(img_shape)

    def process_slice(input_image_slice, mask_generator, axis, start_pos):


        input_shape = input_image_slice.shape

        mask = np.abs(input_image_slice) > 10
        rows, cols = np.where(mask)

        if not (rows.size > 0 and cols.size > 0):
            print('No available pixels, skipping...')
            return

        top, bottom = np.min(rows), np.max(rows)
        left, right = np.min(cols), np.max(cols)

        image_slice = input_image_slice[top:bottom + 1, left:right + 1]
        image_slice = image_slice[:, :, np.newaxis]

        image_3d = np.repeat(image_slice, 3, axis=2)
        image_3d = (image_3d / np.amax(image_3d) * 255).astype(np.uint8)

        masks = (mask_generator.generate(image_3d))
        shape = masks[0]['segmentation'].shape
        masks_label = np.zeros(shape, dtype=int)
        for index, mask in enumerate(masks):
            masks_label[mask['segmentation']] = index + 1

        if axis == 'x':
            sam_data['sam_seg_x'][start_pos, top:bottom + 1, left:right + 1] = masks_label
        elif axis == 'y':
            sam_data['sam_seg_y'][top:bottom + 1, start_pos, left:right + 1] = masks_label
        elif axis == 'z':
            sam_data['sam_seg_z'][top:bottom + 1, left:right + 1, start_pos] = masks_label


    axes = ['x', 'y', 'z'] if main_axis == 'all' else [main_axis]

    if 'x' in axes:
        # For 'x' axis
        for i in range(img_shape[0]):
            print('Processing slice {} using SAM model along x axis.'.format(i))
            process_slice(image[i, :, :], mask_generator, 'x', i)

    if 'y' in axes:
        # For 'y' axis
        for i in range(img_shape[1]):
            print('Processing slice {} using SAM model along y axis.'.format(i))
            process_slice(image[:, i, :], mask_generator, 'y', i)

    if 'z' in axes:
        # For 'z' axis
        for i in range(img_shape[2]):
            print('Processing slice {} using SAM model along z axis.'.format(i))
            process_slice(image[:, :, i], mask_generator, 'z', i)

    return sam_data

def save_volume_SAM_data(sam_data, save_path):
    # Ensure that the directory for the save path exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Open the file in binary write mode and use pickle.dump to save the dictionary
    with open(save_path, 'wb') as f:
        pickle.dump(sam_data, f)

    print('SAM data saved to {}'.format(save_path))

def load_volume_SAM_data(load_path):
    # Check if the file exists
    if not os.path.exists(load_path):
        raise FileNotFoundError('The specified file {} does not exist.'.format(load_path))

    # Open the file in binary read mode and use pickle.load to load the dictionary
    with open(load_path, 'rb') as f:
        sam_data = pickle.load(f)

    print('SAM data loaded from to {}'.format(load_path))
    return sam_data

def check_grid(data, rx, ry, rz, bs):
    if all(data[rx, ry, rz] == value for value in
           [data[rx - bs, ry, rz], data[rx + bs, ry, rz], data[rx, ry - bs, rz], data[rx, ry + bs, rz]]):
        return True, data[rx, ry, rz]
    else:
        return False, -1

def calculate_mapping(array_1, array_2, num_labels):

    if array_1.shape != array_2.shape:
        raise ValueError("The input arrays should have the same shape.")

    mapping = np.zeros((num_labels + 1, num_labels + 1), dtype=int)

    for i in range(array_1.shape[0]):
        for j in range(array_1.shape[1]):
            val_1 = array_1[i, j].astype(int)
            val_2 = array_2[i, j].astype(int)
            mapping[val_1, val_2] += 1

    return mapping

def find_largest_indices(arr):
    # Flatten the array and get the indices that would sort it in descending order
    si = np.argsort(arr.flatten())[::-1]

    # Convert the flattened indices to 2D indices
    si_2d = np.unravel_index(si, arr.shape)

    # Combine the 2D indices and return them as a list of tuples
    return list(zip(si_2d[0], si_2d[1]))


def modify_layer(array, mapping):
    # Find the majority mapping for each label in the first array
    m_array = np.full(array.shape, -1)
    ilist = find_largest_indices(mapping)

    alist = []
    while len(ilist) > 0:
        pval, cval = ilist.pop(0)
        valid = not(any(pval == t[0] for t in alist) or any(cval == t[1] for t in alist))
        if valid:
            alist.append((pval, cval))

    # For each assignment in final assignment list
    for a in alist:
        m_array[array==a[1]] = a[0]

    return m_array

def fine_tune_3d_masks(data_dict, main_axis='z'):

    data = data_dict['sam_seg_{}'.format(main_axis)]
    data_shape = data.shape

    adj_data = data.copy().astype(int)
    max_labels = np.amax(adj_data).astype(int)

    center = data_shape[2] // 2
    print('Using mask layer {} as center'.format(center))

    # First loop: from center to 0
    for rz in range(center, 0, -1):
        print('adjusting masks for layer {}'.format(rz-1))
        mapping = calculate_mapping(adj_data[:, :, rz], data[:, :, rz - 1], max_labels)
        adj_data[:, :, rz - 1] = modify_layer(adj_data[:, :, rz - 1], mapping)

    # Second loop: from center to  data_shape[2]
    for rz in range(center, data_shape[2] - 1):
        print('adjusting masks for layer {}'.format(rz+1))
        mapping = calculate_mapping(adj_data[:, :, rz], data[:, :, rz + 1], max_labels)
        adj_data[:, :, rz + 1] = modify_layer(adj_data[:, :, rz + 1], mapping)

    data_dict['sam_seg_{}'.format(main_axis)] = adj_data

    return data_dict


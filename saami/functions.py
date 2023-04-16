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


# def fine_tune_3d_masks(data_dict, main_axis='z', sample_size=10000, search_size=5):
#
#     data = data_dict['sam_seg_{}'.format(main_axis)]
#     data_shape = data.shape
#     print(data_shape)
#     bs = int((search_size-1)/2)
#
#     check_list = []
#     sid = 0
#     while sid < sample_size:
#
#         if sid % 30 == 0:
#             print('sample ID : {}'.format(sid))
#
#         if not check_list:
#             rx = random.randint(bs, data_shape[0] - bs - 1)
#             ry = random.randint(bs, data_shape[1] - bs - 1)
#             rz = random.randint(bs, data_shape[2] - bs - 1)
#             check_list, data = adjust_mask(data, rx, ry, rz, bs, check_list)
#             sid += 1
#         else:
#             rx, ry, rz = check_list.pop()
#             check_list, data = adjust_mask(data, rx, ry, rz, bs, check_list)
#             sid += 1
#
#                     #print('swapping prev value {}, cur value {}, and next value {}'.format(prev_val, cur_val, next_val))
#
#     data_dict['sam_seg_{}'.format(main_axis)] = data
#
#     return data_dict
#

#
# def fine_tune_3d_masks(data_dict, main_axis='z', sample_size=10000, search_size=5):
#
#     data = data_dict['sam_seg_{}'.format(main_axis)]
#     data_shape = data.shape
#     adj_mask = np.full(data_shape, False)
#     adj_data = np.zeros(data_shape)
#
#     bs = int((search_size-1)/2)
#
#     current_label = 0
#     for rx in range(bs, data_shape[0]-bs, 10):
#         for ry in range(bs, data_shape[1] - bs, 10):
#             print('rx {}, ry {}'.format(rx, ry))
#             check_list = []
#             rz = random.randint(bs, data_shape[2] - bs - 1)
#             check_list, data, adj_mask = adjust_mask(data, adj_mask, rx, ry, rz, bs, check_list)
#             while check_list:
#                 _, _, rz = check_list.pop()
#                 check_list, data, adj_mask = adjust_mask(data, adj_mask, rx, ry, rz, bs, check_list)
#
#     data_dict['sam_seg_{}'.format(main_axis)] = data
#
#     return data_dict

#
# def adjust_mask(data, adj_mask, rx, ry, rz, bs, check_list, main_axis='z'):
#
#     if main_axis == 'z':
#         # if adj_mask[rx, ry, rz] == False:
#         cur, cur_val = check_grid(data, rx, ry, rz, bs)
#         if cur:
#             try:
#                 prev, prev_val = check_grid(data, rx, ry, rz - bs, bs)
#                 next, next_val = check_grid(data, rx, ry, rz + bs, bs)
#
#                 if prev and next:
#
#                     if cur_val != prev_val:
#                         prev_mask = (data[:, :, rz - bs] == prev_val)
#                         cur_mask = (data[:, :, rz - bs] == cur_val)
#                         data[:, :, rz - bs][prev_mask] = cur_val
#                         data[:, :, rz - bs][cur_mask] = prev_val
#                         check_list.append((rx, ry, rz-bs))
#                         print('swapping prev value {}, cur value {}'.format(prev_val, cur_val))
#
#                     if cur_val != next_val:
#                         next_mask = (data[:, :, rz + bs] == next_val)
#                         cur_mask = (data[:, :, rz + bs] == cur_val)
#
#                         data[:, :, rz + bs][next_mask] = cur_val
#                         data[:, :, rz + bs][cur_mask] = next_val
#                         check_list.append((rx, ry, rz+bs))
#                         print('swapping cur value {}, and next value {}'.format(cur_val, next_val))
#
#
#             except:
#                 pass
#
#     return check_list, data, adj_mask



def check_grid(data, rx, ry, rz, bs):
    if all(data[rx, ry, rz] == value for value in
           [data[rx - bs, ry, rz], data[rx + bs, ry, rz], data[rx, ry - bs, rz], data[rx, ry + bs, rz]]):
        return True, data[rx, ry, rz]
    else:
        return False, -1



#
# def fine_tune_3d_masks(data_dict, main_axis='z', sample_size=1000, search_size=5):
#     data = data_dict['sam_seg_{}'.format(main_axis)]
#     data_shape = data.shape
#     adj_mask = np.full(data_shape, False)
#     adj_data = np.full(data_shape, -1)
#
#     bs = int((search_size - 1) / 2)
#
#     # Start with the most prevalent labels
#     mpl, mpl_occ = most_prevalent_labels(data)
#     # Determine the number of samples for each label
#     sample_list = (np.asarray(mpl_occ) * sample_size / np.prod(data_shape)).astype(int)
#     print(mpl)
#     print(mpl_occ)
#     print(sample_list)
#
#
#     for i in range(len(mpl)):
#         label = mpl[i]
#         print('Testing label {}, with {} samples'.format(label, sample_list[i]))
#         for s in range(sample_list[i]):
#             # Get a random starting element with specified label vale
#             rx, ry, rz = random_index_with_label(data, label)
#             print("Starting: rx, ry, rz: ({}, {}, {})".format(rx, ry, rz))
#
#             # Paint all values in adj_data
#             data_mask = (data[:, :, rz] == label) & (adj_data[:, :, rz] == -1)
#             adj_data[:, :, rz][data_mask] = label
#
#             check_list = []
#             check_list, data, adj_data = adjust_mask(data, adj_data, label, rx, ry, rz, 1, bs, check_list) # propagate along +z
#             check_list, data, adj_data = adjust_mask(data, adj_data, label, rx, ry, rz, -1, bs, check_list) # propagate along -z
#
#             while check_list:
#                 _, _, rz, dir = check_list.pop()
#                 check_list, data, adj_data = adjust_mask(data, adj_data, label, rx, ry, rz, dir, bs, check_list)
#
#
#     data_dict['sam_seg_{}'.format(main_axis)] = adj_data
#
#     return data_dict
#



def calculate_mapping(array_1, array_2, num_labels):
    if array_1.shape != array_2.shape:
        raise ValueError("The input arrays should have the same shape.")

    array_1 = array_1.astype(int)
    array_2 = array_2.astype(int)

    mapping = np.zeros((num_labels, num_labels), dtype=int)

    for i in range(array_1.shape[0]):
        for j in range(array_1.shape[1]):
            val_1 = array_1[i, j]
            val_2 = array_2[i, j]
            mapping[val_1, val_2] += 1

    return mapping

def modify_layer(array, mapping):
    # Find the majority mapping for each label in the first array
    majority_mapping = np.argmax(mapping, axis=1)

    # Create a lookup table for modifying the input array based on the majority mapping
    lookup_table = np.arange(mapping.shape[0])
    for i, majority_label in enumerate(majority_mapping):
        lookup_table[i] = majority_label


    # Modify the input array based on the lookup table
    modified_array = lookup_table[array]

    return modified_array


def fine_tune_3d_masks(data_dict, main_axis='z'):

    data = data_dict['sam_seg_{}'.format(main_axis)]
    data_shape = data.shape

    adj_data = data.copy()
    max_labels = np.amax(adj_data).astype(int)

    for rz in range(data_shape[2]):
        mapping = calculate_mapping(adj_data[:, :, rz], data[:, :, rz + 1], max_labels)
        adj_data[:, :, rz + 1] = modify_layer(adj_data[:, :, rz + 1], mapping)

    data_dict['sam_seg_{}'.format(main_axis)] = adj_data

    return data_dict





#
# def adjust_mask(data, adj_data, label, rx, ry, rz, dir, bs, check_list, main_axis='z'):
#
#     if main_axis == 'z':
#         try:
#             print('checking {},{},{}'.format(rx, ry, rz))
#             cur, cur_val = check_grid(data, rx, ry, rz, bs)
#             print(cur)
#             print(cur_val)
#
#             if cur and cur_val == label:
#                 tar, tar_val = check_grid(data, rx, ry, rz + dir * bs, bs)
#                 if tar and cur_val != tar_val:
#                     tar_mask = (data[:, :, rz + dir * bs] == tar_val)
#                     adj_data[:, :, rz + dir * bs][tar_mask] = cur_val
#                     check_list.append((rx, ry, rz + dir * bs, dir))
#                     print('swapping, adjusting the value in layer {} to {}'.format(rz + dir * bs, cur_val))
#                     print(adj_data[:, :, rz + dir * bs])
#                     print('=========')
#         except:
#             pass
#
#     return check_list, data, adj_data
#
#
# def fine_tune_3d_masks(data_dict, main_axis='z', sample_size=50, search_size=5):
#
#     data = data_dict['sam_seg_{}'.format(main_axis)]
#     data_shape = data.shape
#     adj_mask = np.full(data_shape, False)
#     adj_data = np.full(data_shape, -1)
#
#     bs = int((search_size - 1) / 2)
#     current_label = 0
#     for rx in range(bs, data_shape[0]-bs, 10):
#         for ry in range(bs, data_shape[1] - bs, 10):
#             print('rx {}, ry {}'.format(rx, ry))
#             check_list = []
#             rz = random.randint(bs, data_shape[2] - bs - 1)
#             check_list, data, adj_mask = adjust_mask(data, adj_mask, rx, ry, rz, bs, check_list)
#             while check_list:
#                 _, _, rz = check_list.pop()
#                 check_list, data, adj_mask = adjust_mask(data, adj_mask, rx, ry, rz, bs, check_list)
#
#
#     for sid in range(data_shape[2]):
#
#         s_data = data[:, :, sid]
#         mpl, mpl_occ = most_prevalent_labels(s_data)
#         sample_list = (np.asarray(mpl_occ) * sample_size / np.prod(data_shape)).astype(int)
#
#         for i in range(len(mpl)):
#             label = mpl[i]
#             print('Testing label {}, with {} samples'.format(label, sample_list[i]))
#             for s in range(sample_list[i]):
#                 # Get a random starting element with specified label vale
#                 rx, ry = random_index_with_label(s_data, label)
#                 print("Starting: rx, ry, rz: ({}, {}, {})".format(rx, ry, sid))
#
#                 # Paint all values in adj_data
#                 data_mask = (data[:, :, sid] == label) & (adj_data[:, :, sid] == -1)
#                 adj_data[:, :, sid][data_mask] = label
#                 print(adj_data[:, :, sid])
#
#
#









# def fine_tune_3d_masks(data_dict, main_axis='z', sample_size=50, search_size=5):
#
#     data = data_dict['sam_seg_{}'.format(main_axis)]
#     data_shape = data.shape
#     adj_mask = np.full(data_shape, False)
#     adj_data = np.full(data_shape, -1)
#
#     bs = int((search_size - 1) / 2)
#
#
#     for sid in range(data_shape[2]):
#
#         s_data = data[:, :, sid]
#         mpl, mpl_occ = most_prevalent_labels(s_data)
#         sample_list = (np.asarray(mpl_occ) * sample_size / np.prod(data_shape)).astype(int)
#
#         for i in range(len(mpl)):
#             label = mpl[i]
#             print('Testing label {}, with {} samples'.format(label, sample_list[i]))
#             for s in range(sample_list[i]):
#                 # Get a random starting element with specified label vale
#                 rx, ry = random_index_with_label(s_data, label)
#                 print("Starting: rx, ry, rz: ({}, {}, {})".format(rx, ry, sid))
#
#                 # Paint all values in adj_data
#                 data_mask = (data[:, :, sid] == label) & (adj_data[:, :, sid] == -1)
#                 adj_data[:, :, sid][data_mask] = label
#                 print(adj_data[:, :, sid])
#                 print('wwww')
#
#
#
#                 # check_list = []
#                 # check_list, data, adj_data = adjust_mask(data, adj_data, label, rx, ry, sid, 1, bs, check_list) # propagate along +z
#                 # check_list, data, adj_data = adjust_mask(data, adj_data, label, rx, ry, sid, -1, bs, check_list) # propagate along -z
#                 #
#                 # while check_list:
#                 #     _, _, rz, dir = check_list.pop()
#                 #     check_list, data, adj_data = adjust_mask(data, adj_data, label, rx, ry, sid, dir, bs, check_list)
#                 #
#
#
#


    #
    #
    # while sample_id < sample_size:
    #     sample_id += 1
    #
    #     most_prevalent_labels(data)
    #
    #
    #
    # for i in range(len(mpl)):
    #     label = mpl[i]
    #     print('Testing label {}, with {} samples'.format(label, sample_list[i]))
    #     for s in range(sample_list[i]):
    #         # Get a random starting element with specified label vale
    #         rx, ry, rz = random_index_with_label(data, label)
    #         print("Starting: rx, ry, rz: ({}, {}, {})".format(rx, ry, rz))
    #
    #         # Paint all values in adj_data
    #         data_mask = (data[:, :, rz] == label) & (adj_data[:, :, rz] == -1)
    #         adj_data[:, :, rz][data_mask] = label
    #
    #         check_list = []
    #         check_list, data, adj_data = adjust_mask(data, adj_data, label, rx, ry, rz, 1, bs, check_list) # propagate along +z
    #         check_list, data, adj_data = adjust_mask(data, adj_data, label, rx, ry, rz, -1, bs, check_list) # propagate along -z
    #
    #         while check_list:
    #             _, _, rz, dir = check_list.pop()
    #             check_list, data, adj_data = adjust_mask(data, adj_data, label, rx, ry, rz, dir, bs, check_list)
    #
    #
    # data_dict['sam_seg_{}'.format(main_axis)] = adj_data
    #
    # return data_dict
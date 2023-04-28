import os.path
import urllib.request
import numpy as np
from tqdm import tqdm
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor


def get_sam_mask_generator(sam_checkpoint="models/sam_vit_h_4b8939.pth", sam_model_type="vit_h", device="cuda"):
    vit_h_url = 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'

    if not os.path.exists(sam_checkpoint):
        print("SAM checkpoint does not exist, downloading the checkpoint under /models folder ...")
        if not os.path.exists('models'):
            os.makedirs('models')
        urllib.request.urlretrieve(vit_h_url, 'models/sam_vit_h_4b8939.pth')

    sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    # mask_generator = SamAutomaticMaskGenerator(sam)
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,  # Requires open-cv to run post-processing
    )

    return mask_generator


def apply_threshold_label(data, threshold=0.005):

    # Calculate the occurrence of each lable in the input data (2D array)
    ue, counts = np.unique(data, return_counts=True)
    th_val = threshold * data.shape[0] * data.shape[1] 
    mask = ue[counts < th_val]
    data[np.isin(data, mask)] = 0

    return data



def process_slice(sam_data, input_image_slice, mask_generator, axis, start_pos=0, threshold=0.0):

    mask = np.abs(input_image_slice) > 10
    rows, cols = np.where(mask)

    if not (rows.size > 0 and cols.size > 0):
        print('No available pixels, skipping...')
        return

    top, bottom = np.min(rows), np.max(rows)
    left, right = np.min(cols), np.max(cols)

    image_slice = input_image_slice[top:bottom + 1, left:right + 1]

    # Repeat dimension if input slice only has one channel (grayscale image)
    if input_image_slice.shape[2] == 1:
        image_slice = image_slice[:, :, np.newaxis]
        image_3c = np.repeat(image_slice, 3, axis=2)
        image_3c = (image_3c / np.amax(image_3c) * 255).astype(np.uint8)
    else:
        # Assumeing the input image is RGB (and with proper intensity range)
        image_3c = image_slice

    # Run SAM model
    masks = (mask_generator.generate(image_3c))
    shape = masks[0]['segmentation'].shape
    masks_label = np.zeros(shape, dtype=int)

    # Pefrom label post-processing
    for index, mask in enumerate(masks):
        masks_label[mask['segmentation']] = index + 1

    masks_label = masks_label.astype(np.int16)

    # Apply threshold to remove small regions
    if threshold > 0:
        masks_label = apply_threshold_label(masks_label, threshold)

    # Save the segmentation result to the SAM data
    if axis == 'x':
        sam_data["sam_seg"]["x"][start_pos, top:bottom + 1, left:right + 1] = masks_label
    elif axis == 'y':
        sam_data["sam_seg"]["y"][top:bottom + 1, start_pos, left:right + 1] = masks_label
    elif axis == 'z':
        sam_data["sam_seg"]["z"][top:bottom + 1, left:right + 1, start_pos] = masks_label
    elif axis == '2D':
        sam_data["sam_seg"]['2D'] = masks_label
    
    return sam_data



def get_SAM_data(data_dict, mask_generator, main_axis = '2D', threshold=0.0):
    
    image = data_dict["image"]
    label = data_dict["label"]
    img_shape = data_dict["image"].shape

    sam_data = {}
    sam_data["image"] = data_dict["image"]
    sam_data["gt_label"] = data_dict["label"]
    
    sam_data["sam_seg"] = np.zeros(img_shape)
    
    if main_axis == '2D': # 2D case

        sam_data = process_slice(sam_data, image, mask_generator, '2D', threshold=0.0)

    else: # 3D case

        axes = ['x', 'y', 'z'] if main_axis == 'all' else [main_axis]

        if 'x' in axes:
            # For 'x' axis
            total_slices = img_shape[0]
            print('Processing slice using SAM model along x axis.')
            with tqdm(total=total_slices, desc="Processing slices", unit="slice") as pbar:
                for i in range(total_slices):
                    process_slice(image[:, :, i], mask_generator, 'x', i)
                    pbar.update(1)

        if 'y' in axes:
            total_slices = img_shape[1]
            print('Processing slice using SAM model along y axis.')
            with tqdm(total=total_slices, desc="Processing slices", unit="slice") as pbar:
                for i in range(total_slices):
                    process_slice(image[:, :, i], mask_generator, 'y', i)
                    pbar.update(1)

        if 'z' in axes:
            total_slices = img_shape[2]
            print('Processing slice using SAM model along z axis.')
            with tqdm(total=total_slices, desc="Processing slices", unit="slice") as pbar:
                for i in range(total_slices):
                    process_slice(image[:, :, i], mask_generator, 'z', i)
                    pbar.update(1)

    return sam_data["sam_seg"][main_axis]



def get_volume_SAM_data(data_dict, mask_generator, main_axis='z', threshold=0.0):
    
    image = data_dict["image"]
    label = data_dict["label"]
    img_shape = data_dict["image"].shape

    sam_data = {}
    sam_data["image"] = data_dict["image"]
    sam_data["gt_label"] = data_dict["label"]
    
    sam_data["sam_seg"] = {}
    sam_data["sam_seg"]["x"] = np.zeros(img_shape)
    sam_data["sam_seg"]["y"] = np.zeros(img_shape)
    sam_data["sam_seg"]["z"] = np.zeros(img_shape)

    def process_slice(input_image_slice, mask_generator, axis, start_pos):

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

        masks_label = masks_label.astype(np.int16)
        if threshold > 0:
            masks_label = apply_threshold_label(masks_label, threshold)

        if axis == 'x':
            sam_data["sam_seg"]["x"][start_pos, top:bottom + 1, left:right + 1] = masks_label
        elif axis == 'y':
            sam_data["sam_seg"]["y"][top:bottom + 1, start_pos, left:right + 1] = masks_label
        elif axis == 'z':
            sam_data["sam_seg"]["z"][top:bottom + 1, left:right + 1, start_pos] = masks_label

    axes = ['x', 'y', 'z'] if main_axis == 'all' else [main_axis]

    if 'x' in axes:
        # For 'x' axis
        total_slices = img_shape[0]
        print('Processing slice using SAM model along x axis.')
        with tqdm(total=total_slices, desc="Processing slices", unit="slice") as pbar:
            for i in range(total_slices):
                process_slice(image[:, :, i], mask_generator, 'x', i)
                pbar.update(1)

    if 'y' in axes:
        total_slices = img_shape[1]
        print('Processing slice using SAM model along y axis.')
        with tqdm(total=total_slices, desc="Processing slices", unit="slice") as pbar:
            for i in range(total_slices):
                process_slice(image[:, :, i], mask_generator, 'y', i)
                pbar.update(1)

    if 'z' in axes:
        total_slices = img_shape[2]
        print('Processing slice using SAM model along z axis.')
        with tqdm(total=total_slices, desc="Processing slices", unit="slice") as pbar:
            for i in range(total_slices):
                process_slice(image[:, :, i], mask_generator, 'z', i)
                pbar.update(1)

    return sam_data["sam_seg"][main_axis]


def check_grid3d(data, rx, ry, rz, bs):
    if all(data[rx, ry, rz] == value for value in
           [data[rx - bs, ry, rz], data[rx + bs, ry, rz], data[rx, ry - bs, rz], data[rx, ry + bs, rz]]):
        return True, data[rx, ry, rz]
    else:
        return False, -1

def check_grid(arr, rx, ry, ns=2, method='point'):
    if method == 'point':
        if all(arr[rx, ry] == value for value in
            [arr[rx - ns, ry], arr[rx + ns, ry], arr[rx, ry - ns], arr[rx, ry + ns]]):
            return True
        else:
            return False

def check_grid(arr, rx, ry, ns=2, method='point'):
    if method == 'point':
        if all(arr[rx, ry] == value for value in
            [arr[rx - ns, ry], arr[rx + ns, ry], arr[rx, ry - ns], arr[rx, ry + ns]]):
            return True
        else:
            return False

def calculate_mapping(array_1, array_2, num_labels, neighbor_size=0):
    if array_1.shape != array_2.shape:
        raise ValueError("The input arrays should have the same shape.")

    mapping = np.zeros((num_labels + 1, num_labels + 1), dtype=int)

    for i in range(array_1.shape[0]):
        for j in range(array_1.shape[1]):
            val_1 = array_1[i, j].astype(int)
            val_2 = array_2[i, j].astype(int)
            try:
                c1 = check_grid(array_1, i, j, ns=neighbor_size)
                c2 = check_grid(array_2, i, j, ns=neighbor_size)
                if c1 and c2:
                    mapping[val_1, val_2] += 1
                else:
                    j = j + neighbor_size
            except:
                pass

    return mapping


def find_largest_indices(arr):
    # Flatten the array and get the indices that would sort it in descending order
    si = np.argsort(arr.flatten())[::-1]

    # Convert the flattened indices to 2D indices
    si_2d = np.unravel_index(si, arr.shape)

    # Initialize an empty list for the result
    result = []

    # Iterate through the sorted indices and check if the corresponding value is not 0
    for i in range(len(si)):
        if arr[si_2d[0][i], si_2d[1][i]] != 0:
            # Add the non-zero value's indices as a tuple to the result list
            result.append((si_2d[0][i], si_2d[1][i]))

    return result

# Handles the propagated information from previous layer
def modify_layer(array, mapping):
    # Find the majority mapping for each label in the first array
    # m_array = np.full(array.shape, -1)
    m_array = np.zeros(array.shape)

    ilist = find_largest_indices(mapping)

    alist = []
    vlist = []
    while len(ilist) > 0:
        pval, cval = ilist.pop(0)
        # if cval not in vlist:
        #     alist.append((pval, cval))
        #     vlist.append(cval)

        valid = not (any(pval == t[0] for t in alist) or any(cval == t[1] for t in alist))
        if valid:
            alist.append((pval, cval))

    # For each assignment in final assignment list
    for a in alist:
        m_array[array == a[1]] = a[0]

    return m_array

def fine_tune_3d_masks(data_dict, main_axis='z', neighbor_size=0):
    data = data_dict['sam_seg_{}'.format(main_axis)]
    data_shape = data.shape

    adj_data = data.copy().astype(int)
    max_labels = np.amax(adj_data).astype(int)

    def adjust_layers(data, adj_data, start, end, max_labels, neighbor_size):
        if end - start <= 1:
            return

        center = (start + end) // 2

        # Adjust layers from center to start
        for rz in range(center, start, -1):
            mapping = calculate_mapping(adj_data[:, :, rz], data[:, :, rz - 1], max_labels, neighbor_size=neighbor_size)
            adj_data[:, :, rz - 1] = modify_layer(adj_data[:, :, rz - 1], mapping)

        # Adjust layers from center to end
        for rz in range(center, end - 1):
            mapping = calculate_mapping(adj_data[:, :, rz], data[:, :, rz + 1], max_labels, neighbor_size=neighbor_size)
            adj_data[:, :, rz + 1] = modify_layer(adj_data[:, :, rz + 1], mapping)

        # Recursively adjust the layers in the left and right groups
        adjust_layers(data, adj_data, start, center, max_labels, neighbor_size)
        adjust_layers(data, adj_data, center + 1, end, max_labels, neighbor_size)

    total_iterations = 2 * (data_shape[2] - 1)

    print('Adjusting remaining layers...')
    with tqdm(total=total_iterations, desc="Adjusting masks", unit="layer") as pbar:
        adjust_layers(data, adj_data, 0, data_shape[2] - 1, max_labels, neighbor_size)

    return adj_data
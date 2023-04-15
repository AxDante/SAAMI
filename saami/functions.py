import os.path
import urllib.request
import numpy as np
import pickle
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor


def get_volume_SAM_data(data_dict, sam_checkpoint="models/sam_vit_h_4b8939.pth", sam_model_type= "vit_h", device="cuda"):

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

    sam_data['sam_seg_x'] = np.zeros(img_shape)
    sam_data['sam_seg_y'] = np.zeros(img_shape)
    sam_data['sam_seg_z'] = np.zeros(img_shape)


    def process_slice(image_slice, mask_generator, axis, pos):

        print(image_slice.shape)

        mask = np.abs(image_slice) > 10
        rows, cols = np.where(mask)

        if not (rows.size > 0 and cols.size > 0):
          return

        top, bottom = np.min(rows), np.max(rows)
        left, right = np.min(cols), np.max(cols)

        image_slice = image_slice[top:bottom + 1, left:right + 1]
        image_slice = image_slice[:, :, np.newaxis]

        image_3d = np.repeat(image_slice, 3, axis=2)
        image_3d = (image_3d / np.amax(image_3d) * 255).astype(np.uint8)

        masks = (mask_generator.generate(image_3d))
        shape = masks[0]['segmentation'].shape
        masks_label = np.zeros(shape, dtype=int)
        for index, mask in enumerate(masks):
            masks_label[mask['segmentation']] = index + 1

        if axis == 'x':
            sam_data['sam_seg_x'][pos, top:bottom + 1, left:right + 1] = masks_label
        elif axis == 'y':
            sam_data['sam_seg_y'][top:bottom + 1, pos, left:right + 1] = masks_label
        elif axis == 'z':
            sam_data['sam_seg_z'][top:bottom + 1, left:right + 1, pos] = masks_label


    for i in range(img_shape[0]):
        process_slice(image[i, :, :], mask_generator, 'x', i)

    for i in range(img_shape[1]):
        process_slice(image[:, i, :], mask_generator, 'y', i)

    for i in range(img_shape[2]):
        print('Processing slice {} using SAM model.'.format(i))
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
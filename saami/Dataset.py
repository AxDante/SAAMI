from torch.utils.data import Dataset
import os
import nibabel as nib
import numpy as np
import re
import cv2

class VolumeDataset(Dataset):
    
    """
    VolumeDataset
    """
    
    def __init__(self, input_dir, roi=None):
        self.input_dir = input_dir
        self.data = {}
        self.folder_names = os.listdir(input_dir)  # Get all folders under input_dir
        image_file_list = []

        for fid in range(len(self.folder_names)):  # for loop iterate over all folder_name under input_dir
            folder_name = self.folder_names[fid]
            image_folder = os.path.join(input_dir, folder_name, "Images")
            label_folder = os.path.join(input_dir, folder_name, "Labels")

            image_file = os.listdir(image_folder)[0]  # Currently we assume only one image file per folder, which would be the main volume for the patient
            image_path = os.path.join(image_folder, image_file)
            image_file_list.append(image_path)
            image_data = nib.load(image_path).get_fdata()

            label_data = np.zeros(image_data.shape)

            label_files = os.listdir(label_folder)

            combined_labels =  [f for f in label_files if f.endswith("combined_seg.nii.gz")]

            if len(combined_labels) > 0 :
                # Use the combined file if it exists
                label_data = nib.load(os.path.join(label_folder, combined_labels[0])).get_fdata()
            else:
                # Otherwise, combine all label files
                for label_file in os.listdir(label_folder):
                    if label_file.endswith(".nii.gz"):
                        label_path = os.path.join(label_folder, label_file)
                        label_id_str = re.findall(r'\d+', label_file)[0]
                        label_id = int(label_id_str)
                        label_image_data = nib.load(label_path).get_fdata()
                        label_data += label_id * (label_image_data > 0)

            image = image_data[:, :, :]
            label = label_data[:, :, :]

            # ROI adjustment
            if roi:
                image = image_data[roi[0][0]:roi[1][0], roi[0][1]:roi[1][1], :]
                label = label_data[roi[0][0]:roi[1][0], roi[0][1]:roi[1][1], :]

            # Check shape between image and label
            assert label.shape == image.shape, "Shape mismatch between image and label"

            self.data[fid] = {"image": image, "label": label}

        self.image_file_list = image_file_list

    def __len__(self):
        """
        Return number of folders under input_dir
        """
        return len(self.folder_names)

    def __getitem__(self, idx):
        """
        Get data_dict corresponding to idx
        """
        return self.data[idx]

    def update_label(self, idx, label, label_data):
        """
        Update data in the dataset
        """
        self.data[idx][label] = label_data

    def update_data(self, idx, data):
        """
        Update data in the dataset
        """
        self.data[idx] = data



class ImageDataset(Dataset):
    """
    ImageDataset
    """
    def __init__(self,
             input_dir, roi = None):
        self.input_dir = input_dir
        self.data = []

        image_folder = os.path.join(input_dir, "Images")
        assert os.path.exists(image_folder), "Image folder does not exist"
        image_file_list = os.listdir(image_folder)
        image_file_list = [img for img in image_file_list if img.lower().endswith(('.jpg', '.png', '.jpeg'))]
        image_file_list.sort()


        label_folder = os.path.join(input_dir, "Labels")
        gt_label = True
        if not os.path.exists(label_folder):
            print("Processing SAAMI without labels....")
            gt_label = False
            label_file_list = [''] * len(image_file_list)
        else:
            label_file_list = os.listdir(label_folder)
            label_file_list = [lbl for lbl in label_file_list if lbl.lower().endswith(('.jpg', '.png', '.jpeg'))]
            label_file_list.sort()
            assert len(image_file_list) == len(label_file_list), "Number of images and labels do not match"

        # Iterate through each pair and load image/label for each pair
        for img_file, lbl_file in zip(image_file_list, label_file_list):

            img_path = os.path.join(image_folder, img_file)
            lbl_path = os.path.join(label_folder, lbl_file)

            image = cv2.imread(img_path)
            label = None
            if gt_label: 
                label = cv2.imread(lbl_path)

            # ROI adjustment
            if roi:
                image = image[roi[0][0]:roi[1][0], roi[0][1]:roi[1][1]]
                if gt_label:
                    label = label[roi[0][0]:roi[1][0], roi[0][1]:roi[1][1]]

            data = {"image": image, "label": label}
            self.data.append(data)

        self.image_file_list = image_file_list

    def __len__(self):
        """
        Return the number of data items in the dataset
        """
        return len(self.data)

    def __getitem__(self,
                    idx):
        """
        Get data corresponding to idx
        """
        return self.data[idx]
    
    
    def update_data(self, idx, data):
        """
        Update data in the dataset
        """
        self.data[idx] = data

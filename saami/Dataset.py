from torch.utils.data import Dataset
import os
import nibabel as nib
import numpy as np
import re

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

            image_file = os.listdir(image_folder)[0]
            image_path = os.path.join(image_folder, image_file)
            image_file_list.append(image_path)
            image_data = nib.load(image_path).get_fdata()

            label_data = np.zeros(image_data.shape)

            label_files = os.listdir(label_folder)

            combined_label_file =  [f for f in label_files if f.endswith("combined_seg.nii.gz")][0] 

            if combined_label_file:
                # Use the combined file if it exists
                label_data = nib.load(os.path.join(label_folder, combined_label_file)).get_fdata()
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

    def update_data(self, data, label, idx):
        """
        Update data in the dataset
        """
        self.data[idx][label] = data


class ImageDataset(Dataset):
    """
    ImageDataset
    """

    def __init__(self,
                 input_dir):
        self.input_dir = input_dir
        self.data = {}
        self.folder_names = os.listdir(input_dir)  # Get all folders under input_dir

        for folder_name in self.folder_names:  # for loop iterate over all folder_name under input_dir
            image_folder = os.path.join(input_dir, folder_name, "Images")
            label_folder = os.path.join(input_dir, folder_name, "Labels")

            image_file = os.listdir(image_folder)[0]
            image_path = os.path.join(image_folder, image_file)
            image_data = nib.load(image_path).get_fdata()

            label_data = np.zeros(image_data.shape)

            label_files = os.listdir(label_folder)

            combined_label_file =  [f for f in label_files if f.endswith("combined_seg.nii.gz")][0] 

            print(combined_label_file)
            print(image_folder)
            print(label_folder)

            if combined_label_file:
                print('combined label file exists')
                label_data = nib.load(os.path.join(label_folder, combined_label_file)).get_fdata()
            else:
                for label_file in os.listdir(label_folder):
                    if label_file.endswith(".nii.gz"):
                        label_path = os.path.join(label_folder, label_file)
                        label_id_str = re.findall(r'\d+', label_file)[0]
                        label_id = int(label_id_str)
                        label_image_data = nib.load(label_path).get_fdata()
                        label_data += label_id * (label_image_data > 0)

            image = image_data[:, :, :]
            label = label_data[:, :, :]

            self.data[folder_name] = {"image": image, "label": label}

    def __len__(self):
        """
        Return number of folders under input_dir
        """
        return len(self.folder_names)

    def __getitem__(self,
                    idx):
        """
        Get data_dict corresponding to idx
        """
        folder_name = self.folder_names[idx]
        return self.data[folder_name]
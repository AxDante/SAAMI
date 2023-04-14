from torch.utils.data import Dataset
import os
import nibabel as nib
import numpy as np
import re

class VolumeDataset(Dataset):
    
    """
    VolumeDataset
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
"Utility file for saami package"
import os
import numpy as np
import nibabel as nib
import pickle

def download_progress_hook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    progress = min(100, (downloaded / total_size) * 100)
    print(f"\rDownload progress: {progress:.2f}%", end='')

def most_prevalent_labels(data):
    unique, counts = np.unique(data, return_counts=True)
    label_count = sorted(zip(unique, counts), key=lambda x: x[1], reverse=True)

    labels = [int(label) for label, count in label_count]
    occurrences = np.asarray([count for label, count in label_count])
    return labels, occurrences

def random_index_with_label(data, label):

    if len(data.shape) == 3:
        indices = np.argwhere(data == label)
        if indices.size == 0:
            return (-1, -1, -1)
        random_index = indices[np.random.choice(indices.shape[0])]
        return tuple(random_index)

    elif len(data.shape) == 2:
        indices = np.argwhere(data == label)
        if indices.size == 0:
            return (-1, -1)
        random_index = indices[np.random.choice(indices.shape[0])]
        return tuple(random_index)

def convert_to_nifti(data_dict, save_path='outputs/test.nii', main_axis='z', affine=np.eye(4)):

    data = data_dict['sam_seg'][main_axis]

    # Convert data to nifti image
    data_array = np.array(data, dtype=np.int16)  # Convert the data type to int16
    nifti_img = nib.Nifti1Image(data_array, affine)

    # Check if the saving folder exists
    folder_path = os.path.dirname(save_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Save data
    nib.save(nifti_img, save_path)
    print('Nifti data saved to {}'.format(save_path))


def save_npz_data(sam_data, save_path):
    
    # The npz data contains the original image, ground truth label (if any), and SAM masks
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Open the file in binary write mode and use pickle.dump to save the dictionary
    with open(save_path, 'wb') as f:
        pickle.dump(sam_data, f)

    print('SAM data saved to {}'.format(save_path))

def flatten_dict(data_dict, parent_key='', sep='_'):
    """
    Flatten a nested dictionary.

    Args:
        data_dict (dict): The nested dictionary to flatten.
        parent_key (str, optional): The key for the current level. Default is an empty string.
        sep (str, optional): The separator to use between keys. Default is '_'.

    Returns:
        dict: A flattened dictionary.
    """
    flattened = {}
    for key, value in data_dict.items():
        new_key = "{}{}{}".format(parent_key, sep, key) if parent_key else key
        if isinstance(value, dict):
            flattened.update(flatten_dict(value, new_key, sep=sep))
        else:
            flattened[new_key] = value
    return flattened



def save_data_to_npz_file(data_dict, keys, file_name="outputs/data.npz"):
    """
    Save data from a dictionary to a single npz file.

    Args:
        data_dict (dict): Dictionary containing the data.
        keys (list): List of keys to save from the dictionary.
        file_name (str, optional): Name of the saved file. Default is 'data.npz'.
    """

    data_to_save = {}
    for key in keys:
        if key in data_dict:
            value = data_dict[key]
            if isinstance(value, dict):
                flat_value = flatten_dict(value, parent_key=key)
                data_to_save.update(flat_value)
            else:
                data_to_save[key] = value
        else:
            print("Key {} not found in the data dictionary".format(key))
    
    if data_to_save:
        np.savez(file_name, **data_to_save)
        print("Data saved to :{}".format(file_name))
    else:
        print("No data to save to file")


def save_SAM_data(sam_data, save_path):
    # Ensure that the directory for the save path exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Open the file in binary write mode and use pickle.dump to save the dictionary
    with open(save_path, 'wb') as f:
        pickle.dump(sam_data, f)

    print('SAM data saved to {}'.format(save_path))

def load_SAM_data(load_path):
    # Check if the file exists
    if not os.path.exists(load_path):
        raise FileNotFoundError('The specified file {} does not exist.'.format(load_path))

    # Open the file in binary read mode and use pickle.load to load the dictionary
    with open(load_path, 'rb') as f:
        sam_data = pickle.load(f)

    print('SAM data loaded from to {}'.format(load_path))
    return sam_data
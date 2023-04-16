"Utility file for saami package"
import os
import numpy as np
import nibabel as nib

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

    data = data_dict['sam_seg_{}'.format(main_axis)]

    # Convert data to nifti image
    data_array = np.array(data)
    nifti_img = nib.Nifti1Image(data_array, affine)

    # Check if the saving folder exists
    folder_path = os.path.dirname(save_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Save data
    nib.save(nifti_img, save_path)
    print('Nifti data saved to {}'.format(save_path))
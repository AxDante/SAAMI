import nibabel as nib
from typing import NamedTuple
from saami.Dataset import VolumeDataset, ImageDataset
from saami.utils import save_SAM_data, convert_to_nifti
from saami.functions import get_SAM_data, fine_tune_3d_masks, get_sam_mask_generator
from saami.visualizer import visualize_volume_SAM


class DatasetInfo(NamedTuple):
    dataset_class: type
    main_axis: str

    
class SAAMI:
    def __init__(self, data_dir, mask_generator=None, main_axis='z', roi=None, dataset_type='Volume'):
        self.data_dir = data_dir
        self.roi = roi
        self.mask_generator = mask_generator if mask_generator else get_sam_mask_generator()
        self.dataset_type = dataset_type
        self.main_axis = main_axis if main_axis else '2D' if dataset_type == 'Image' else 'z'
        # Dataset type mapping
        dataset_class_map = {
            "Volume": DatasetInfo(VolumeDataset, main_axis if main_axis else 'z'),
            "Image": DatasetInfo(ImageDataset, '2D'),
        }

        # Create an instance of the specified dataset class
        self.dataset = dataset_class_map[dataset_type].dataset_class(data_dir, roi=self.roi)

    def calculate_3d_mask(self, idx, threshold=0.0):

        assert self.dataset_type == 'VolumeDataset', "calculate_3d_mask is only available for VolumeDataset"
        data = self.dataset[idx]
        mask_data = get_SAM_data(data, self.mask_generator, main_axis=self.main_axis, threshold=threshold)
        data['sam_seg'][self.main_axis] = mask_data
        self.dataset.update_data(idx, mask_data)
        return mask_data

    def get_mask(self, idx):
        data = self.dataset[idx]
        if self.dataset_type == 'VolumeDataset':
            return data['sam_seg_{}'.format(self.main_axis)]
        elif self.dataset_type == 'ImageDataset':
            return data['sam_seg'.format(self.main_axis)]

    def finetune_3d_mask(self, idx, neighbor_size=3):
        assert self.dataset_type == 'VolumeDataset', "finetune_3d_mask is only available for VolumeDataset"
        data = self.dataset[idx]
        mask_data = fine_tune_3d_masks(data, main_axis=self.main_axis, neighbor_size=neighbor_size)
        data['sam_seg'][self.main_axis] = mask_data
        self.dataset.update_data(idx, mask_data)
        return mask_data

    def save_data(self, idx, save_path='outputs/saved_data.pkl'):
        data = self.dataset[idx]
        save_SAM_data(data, save_path)

    def save_all_data(self, save_path=None):
        for i in range(len(self.dataset)):
            save_path = save_path if save_path else 'outputs/saved_data_{}.pkl'.format(i)
            self.save_data(i, save_path=save_path)

    def save_mask(self, idx, save_path='outputs/saved_mask.nii'):
        data = self.dataset[idx]
        orig_file_path = self.dataset.image_file_list[idx]
        orig_nifti = nib.load(orig_file_path)
        convert_to_nifti(data, save_path=save_path, affine=orig_nifti.affine)

    def save_masks(self, save_path=None):
        for i in range(len(self.dataset)):
            save_path = save_path if save_path else 'outputs/saved_mask_{}.nii'.format(i)
            self.save_mask(i, save_path=save_path)

    def visualize(self, idx, save_folder="outputs/example_images"):
        data = self.dataset[idx]
        visualize_volume_SAM(data, save_path=save_folder, show_tkinter = True)
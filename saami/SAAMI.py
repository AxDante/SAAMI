import nibabel as nib
from saami.Dataset import VolumeDataset
from saami.utils import save_volume_SAM_data, convert_to_nifti
from saami.functions import get_volume_SAM_data, fine_tune_3d_masks, get_sam_mask_generator
from saami.visualizer import visualize_volume_SAM

class SAAMI:
    def __init__(self, data_dir, mask_generator=None, main_axis='z', roi=None):
        self.data_dir = data_dir
        self.roi = roi
        self.mask_generator = mask_generator if mask_generator else get_sam_mask_generator()
        self.vol_dataset = VolumeDataset(data_dir, roi=self.roi)
        self.main_axis = main_axis

    def calculate_3d_mask(self, idx):
        data = self.vol_dataset[idx]
        mask_data = get_volume_SAM_data(data, self.mask_generator, main_axis=self.main_axis)
        self.vol_dataset.update_data(mask_data, 'sam_seg_{}'.format(self.main_axis), idx)
        return mask_data

    def get_mask(self, idx):
        data = self.vol_dataset[idx]
        return data['sam_seg_{}'.format(self.main_axis)]

    def finetune_3d_mask(self, idx):
        data = self.vol_dataset[idx]
        mask_data = fine_tune_3d_masks(data, main_axis=self.main_axis)
        self.vol_dataset.update_data(mask_data, 'sam_seg_{}'.format(self.main_axis), idx)
        return mask_data

    def save_data(self, idx, save_path='outputs/saved_data.pkl'):
        data = self.vol_dataset[idx]
        save_volume_SAM_data(data, save_path)

    def save_mask(self, idx, save_path='outputs/saved_mask.nii'):
        data = self.vol_dataset[idx]
        orig_file_path = self.vol_dataset.image_file_list[idx]
        orig_nifti = nib.load(orig_file_path)
        convert_to_nifti(data, save_path=save_path, affine=orig_nifti.affine)

    def visualize(self, idx, save_folder="outputs/example_images"):
        data = self.vol_dataset[idx]
        visualize_volume_SAM(data, save_path=save_folder, show_tkinter = True)
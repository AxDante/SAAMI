import cv2
import os
from tqdm import tqdm
import nibabel as nib
from typing import NamedTuple
from saami.Dataset import VolumeDataset, ImageDataset
from saami.utils import save_SAM_data, convert_to_nifti, save_data_to_npz_file
from saami.functions import get_SAM_data, fine_tune_3d_masks, get_sam_mask_generator
from saami.visualizer import visualize_volume_SAM
from typing import Any, Callable, Optional, Union


class DatasetInfo(NamedTuple):
    dataset_class: type
    main_axis: str

class SAAMI:
    def __init__(self, data_dir: str, mask_generator: Optional[Callable] = None, main_axis: str = '', roi: Optional[Any] = None, dataset_type: str = 'Volume'):
        """
        Initialize SAAMI class.

        Args:
            data_dir (str): The directory containing the dataset.
            mask_generator (Optional[Callable]): The function for generating the mask.
            main_axis (str): The main axis of the dataset.
            roi (Optional[Any]): The region of interest.
            dataset_type (str): The type of dataset being used (Volume or Image).
        """
        self.data_dir = data_dir
        self.roi = roi
        self.mask_generator = mask_generator if mask_generator else get_sam_mask_generator(sam_model_type="vit_b")
        self.main_axis = main_axis if main_axis else '2D' if dataset_type == 'Image' else 'z'
        
        # Dataset type mapping
        dataset_class_map = {
            "Volume": DatasetInfo(VolumeDataset, self.main_axis),
            "Image": DatasetInfo(ImageDataset, '2D'),
        }
        self.dataset_type = dataset_type
        # Create an instance of the specified dataset class
        self.dataset = dataset_class_map[dataset_type].dataset_class(data_dir, roi=self.roi)

    def calculate_3d_mask(self, idx: int, threshold: float = 0.0, img_normalize: bool = False) -> Any:
        """
        Calculate 3D mask for the VolumeDataset.

        Args:
            idx (int): The index of the volume in the dataset.
            threshold (float): Threshold value for mask calculation.
            img_normalize (bool): Whether to normalize the image data.

        Returns:
            Any: The mask data.
        """

        assert self.dataset_type == 'Volume', "calculate_3d_mask is only available for VolumeDataset"
        data = self.dataset[idx]
        mask_data = get_SAM_data(data, self.mask_generator, main_axis=self.main_axis, threshold=threshold, img_normalize=img_normalize)
        data['sam_seg'][self.main_axis] = mask_data
        self.dataset.update_data(idx, mask_data)
        return mask_data
    
    def calculate_mask(self, idx: int, threshold: float = 0.0, update: bool = False, save_path: str = '', img_normalize: bool = False, img_operation: Optional[Callable] = None) -> Any:
        """
        Calculate the mask for the given index.

        Args:
            idx (int): The index of the item in the dataset.
            threshold (float): Threshold value for mask calculation.
            update (bool): Whether to update the dataset with the calculated mask.
            save_path (str): The path to save the mask.
            img_normalize (bool): Whether to normalize the image data.
            img_operation (Optional[Callable]): External image operation function.

        Returns:
            Any: The mask data.
        """
        data = self.dataset[idx]
        sam_data, pp_data = get_SAM_data(data, self.mask_generator, main_axis=self.main_axis, threshold=threshold, img_normalize=img_normalize, img_operation=img_operation)
        if update:
            self.dataset.update_data(idx, sam_data)
        if save_path:
            # print('saving data to {}'.format(save_path))
            self.save_mask(sam_data, save_path=save_path, idx=idx)
            if pp_data is not None:
                pp_save_path = "{}_preprocessed.jpg".format((os.path.splitext(save_path))[0])
                # print('saving PP data to {}'.format(pp_save_path))
                cv2.imwrite(pp_save_path, pp_data)
        return sam_data['sam_seg'][self.main_axis]


    def calculate_masks(self, threshold: float = 0.0, update: bool = False, save_data: bool = False, save_dir: str = 'outputs', img_normalize: bool = False , img_operation: Optional[Callable] = None) -> None:
        """
        Calculate masks for all items in the dataset.

        Args:
            threshold (float): Threshold value for mask calculation.
            update (bool): Whether to update the dataset with the calculated masks.
            save_data (bool): Whether to save the calculated masks.
            save_dir (str): The directory to save the masks.
            img_normalize (bool): Whether to normalize the image data.
            img_operation (Optional[Callable]): External image operation function.

        """
        if save_data:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)

        for idx in tqdm(range(len(self.dataset)), desc="Calculating masks ... "):
            if save_data:
                if self.dataset_type == 'Image':
                    save_path = os.path.join(save_dir, "{}_sam_mask.jpg".format(os.path.splitext(self.dataset.image_file_list[idx])[0]))
                    self.calculate_mask(idx, threshold=threshold, update=update, save_path=save_path, img_normalize=img_normalize, img_operation=img_operation)  
                    # Save pre-processed images if available

                elif self.dataset_type == 'Volume':    
                    save_path = os.path.join(save_dir, "{}_sam_mask.nii.gz".format(os.path.splitext(self.dataset.image_file_list[idx])[0]))
                    self.calculate_mask(idx, threshold=threshold, update=update, save_path=save_path, img_normalize=img_normalize, img_operation=img_operation)  

    def get_mask(self, idx: int) -> Any:
        """
        Get the mask for the given index.

        Args:
            idx (int): The index of the item in the dataset.

        Returns:
            Any: The mask data.
        """
        data = self.dataset[idx]
        return data['sam_seg'][self.main_axis]

    def finetune_3d_mask(self, idx: int, neighbor_size: int = 3) -> Any:
        """
        Fine-tune the 3D mask for the given index.

        Args:
            idx (int): The index of the volume in the dataset.
            neighbor_size (int): The size of the neighborhood for fine-tuning.

        Returns:
            Any: The fine-tuned mask data.
        """
        assert self.dataset_type == 'Volume', "finetune_3d_mask is only available for VolumeDataset"
        sam_data = self.dataset[idx]
        mask_data = fine_tune_3d_masks(sam_data, main_axis=self.main_axis, neighbor_size=neighbor_size)
        sam_data['sam_seg'][self.main_axis] = mask_data
        self.dataset.update_data(idx, sam_data)
        return mask_data

    def save_numpy_data(self, idx: int, save_path: str = '') -> None:
        """
        Save the numpy data for the given index.

        Args:
            idx (int): The index of the item in the dataset.
            save_path (str): The path to save the numpy data.
        """
        data = self.dataset[idx]
        save_path = save_path if save_path else 'outputs/saved_data_{}.npz'.format(idx)
        keys = ['image', 'label', 'sam_seg']
        save_data_to_npz_file(data, keys, save_path)

    def save_all_numpy_data(self, save_folder: str = 'outputs') -> None:
        """
        Save all numpy data from the dataset.

        Args:
            save_folder (str): The folder to save the numpy data.
        """
        for i in range(len(self.dataset)):
            save_path = '{}/{:05d}.npz'.format(save_folder, i)
            self.save_numpy_data(i, save_path=save_path)

    def save_all_data(self, save_path: Optional[str] = None) -> None:
        """
        Save all data from the dataset.

        Args:
            save_path (str, optional): The path to save the data.
        """        
        for i in range(len(self.dataset)):
            save_path = save_path if save_path else 'outputs/saved_data_{}.pkl'.format(i)
            self.save_data(i, save_path=save_path)

    def save_mask(self, data: Any, save_path: str = 'outputs/saved_mask.nii', idx: int = 0) -> None:
        """
        Save the mask data.

        Args:
            data (Any): The mask data.
            save_path (str): The path to save the mask.
            idx (int): The index of the item in the dataset.
        """

        if save_path.endswith('.nii'):
            # Index is needed to find the original nifti file
            orig_file_path = self.dataset.image_file_list[idx]
            assert os.path.exists(orig_file_path), "Original nifti does not exist"
            orig_nifti = nib.load(orig_file_path)
            convert_to_nifti(data, save_path=save_path, affine=orig_nifti.affine)
        elif save_path.endswith('.pkl'):
            save_SAM_data(data, save_path)
        elif save_path.endswith('jpg') or save_path.endswith('png'):
            cv2.imwrite(save_path, data['sam_seg'][self.main_axis])

    def save_mask_by_index(self, idx: int, save_path: str = 'outputs/saved_mask.nii') -> None:
        """
        Save the mask data for the given index.

        Args:
            idx (int): The index of the item in the dataset.
            save_path (str): The path to save the mask.
        """
        data = self.dataset[idx]
        if save_path.endswith('.nii'):
            orig_file_path = self.dataset.image_file_list[idx]
            assert os.path.exists(orig_file_path), "Original nifti does not exist"
            orig_nifti = nib.load(orig_file_path)
            convert_to_nifti(data, save_path=save_path, affine=orig_nifti.affine)
        elif save_path.endswith('.pkl'):
            save_SAM_data(data, save_path)
        elif save_path.endswith('jpg') or save_path.endswith('png'):
            cv2.imwrite(save_path, data['sam_seg'][self.main_axis])

    def save_masks(self, save_path: Optional[str] = None) -> None:
        """
        Save masks for all items in the dataset.

        Args:
            save_path (str, optional): The path to save the masks.
        """
        for i in range(len(self.dataset)):
            save_path = save_path if save_path else 'outputs/saved_mask_{}.nii'.format(i)
            self.save_mask_by_index(i, save_path=save_path)

    def visualize(self, idx: int, save_folder: str = "outputs/example_images") -> None:
        """
        Visualize the data for the given index.

        Args:
            idx (int): The index of the item in the dataset.
            save_folder (str): The folder to save the visualization.
        """
        data = self.dataset[idx]
        visualize_volume_SAM(data, save_path=save_folder, show_tkinter = True)
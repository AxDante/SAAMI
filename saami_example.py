from saami.Dataset import VolumeDataset
from saami.functions import get_volume_SAM_data, save_volume_SAM_data, load_volume_SAM_data, fine_tune_3d_masks
from saami.visualizer import visualize_volume_SAM
from saami.utils import convert_to_nifti
import os
import nibabel as nib

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE' # Walkaround
 
spine_data = VolumeDataset(os.path.join(os.getcwd(), "data/MRI_example"))[0]

example_spine_data_path = 'outputs/example_spine_sam_data.pkl'
if not os.path.exists(example_spine_data_path):
    spine_sam_data = get_volume_SAM_data(spine_data)
    save_volume_SAM_data(spine_sam_data, example_spine_data_path)
else:
    spine_sam_data = load_volume_SAM_data(example_spine_data_path)

example_ft_spine_data_path = 'outputs/ft_example_spine_sam_data.pkl'
if not os.path.exists(example_ft_spine_data_path):
    ft_spine_sam_data = fine_tune_3d_masks(spine_sam_data)
    save_volume_SAM_data(ft_spine_sam_data, example_ft_spine_data_path)
else:
    ft_spine_sam_data = load_volume_SAM_data(example_ft_spine_data_path)

orig_nifti = nib.load('data/MRI_example/ID02/Images/T1_contrast.nii.gz')
print(orig_nifti)
print(orig_nifti.affine)

convert_to_nifti(ft_spine_sam_data, save_path='outputs/example_spine_3d_mask.nii', affine=orig_nifti.affine)
visualize_volume_SAM(ft_spine_sam_data, save_path=os.path.join(os.getcwd(), "outputs/example_images"), show_tkinter=True)
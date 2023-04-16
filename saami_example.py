from saami.Dataset import VolumeDataset
from saami.functions import get_volume_SAM_data, save_volume_SAM_data, load_volume_SAM_data, fine_tune_3d_masks
from saami.visualizer import visualize_volume_SAM
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE' # Walkaround
 
spine_data = VolumeDataset(os.path.join(os.getcwd(), "data/MRI_example"))[0]

example_spine_data_path = 'outputs/example_spine_sam_data.pkl'
if not os.path.exists(example_spine_data_path):
    spine_sam_data = get_volume_SAM_data(spine_data)
    save_volume_SAM_data(spine_sam_data, example_spine_data_path)
else:
    spine_sam_data = load_volume_SAM_data(example_spine_data_path)

spine_sam_data = fine_tune_3d_masks(spine_sam_data)


visualize_volume_SAM(spine_sam_data, save_path=os.path.join(os.getcwd(), "outputs/example_images"), show_tkinter=True)
from saami.VolumeData import VolumeDataset
from saami.functions import get_volume_SAM
from saami.visualizer import visualize_volume_SAM
import os

spine_data = VolumeDataset(os.path.join(os.getcwd(), "data/MRI_dataset"))
spine_sam_data = get_volume_SAM(spine_data)
visualize_volume_SAM(spine_sam_data)
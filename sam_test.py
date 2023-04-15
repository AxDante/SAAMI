from saami.VolumeData import VolumeDataset
from saami.functions import get_volume_SAM
from saami.visualizer import visualize_volume_SAM
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE' #Walkaround
 
spine_data = VolumeDataset(os.path.join(os.getcwd(), "data/MRI_sample"))[0]
print(spine_data)
spine_sam_data = get_volume_SAM(spine_data)
print(spine_sam_data)

visualize_volume_SAM(spine_sam_data, save_path=os.path.join(os.getcwd(), "outputs"))
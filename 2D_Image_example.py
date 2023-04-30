# Import sammi module
from saami.SAAMI import SAAMI

# Data2D = SAAMI('data/Chest_X-ray_example_processed', dataset_type='Image')

# print('dataset loaded')
# mask = Data2D.calculate_mask(0, threshold= 0.003)

# Data2D.save_mask(0, save_path='outputs/saved_2D_sam_mask.png')
# Data2D.save_numpy_data(0, save_path='outputs/saved_2D_data.npz')


ISIC_data = SAAMI('data/ISIC2018_segmentation_example', dataset_type='Image')
ISIC_data.calculate_masks()
ISIC_data.save_all_numpy_data()

# Import sammi module
from saami.SAAMI import SAAMI

Data2D = SAAMI('data/Chest_X-ray_example_processed', roi=None, dataset_type='Image')

print('dataset loaded')
mask = Data2D.calculate_mask(0, threshold= 0.003)

Data2D.save_mask(0, save_path='outputs/saved_2D_sam_mask.png')


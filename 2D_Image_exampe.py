# Import sammi module
from saami.SAAMI import SAAMI

dataset_2d = SAAMI('data/Chest_X-ray_example_processed', roi=None, dataset_type='Image')

print('dataset loaded')
mask = dataset_2d.calculate_3d_mask(0, threshold= 0.003)

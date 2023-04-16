from saami.Dataset import VolumeDataset
from saami.functions import get_volume_SAM_data, save_volume_SAM_data, load_volume_SAM_data, fine_tune_3d_masks
from saami.visualizer import visualize_volume_SAM
import numpy as np

data = np.array([[[1, 1, 1, 1, 2, 2, 2,],
                  [1, 1, 1, 1, 2, 2, 2,],
                  [1, 1, 1, 1, 2, 3, 2,],
                  [1, 1, 1, 1, 3, 3, 3,],
                  [1, 1, 1, 1, 3, 3, 3,],
                  [2, 2, 2, 2, 3, 3, 3,],
                  [2, 2, 2, 2, 3, 3, 0]],
                 [[4, 4, 4, 4, 2, 2, 2,],
                  [4, 4, 4, 4, 2, 2, 2,],
                  [4, 4, 4, 4, 2, 3, 2,],
                  [4, 4, 4, 4, 1, 1, 1,],
                  [4, 4, 4, 4, 1, 1, 1,],
                  [2, 2, 2, 2, 1, 1, 1,],
                  [2, 2, 2, 2, 1, 1, 3]],
                 [[1, 1, 1, 1, 2, 2, 2,],
                  [1, 1, 1, 1, 2, 2, 2,],
                  [1, 1, 1, 1, 2, 3, 2,],
                  [1, 1, 1, 1, 3, 3, 3,],
                  [1, 1, 1, 1, 3, 3, 3,],
                  [2, 2, 2, 2, 3, 3, 3,],
                  [2, 2, 2, 2, 3, 3, 4]],
                 [[0, 0, 0, 1, 2, 2, 2, ],
                  [0, 0, 0, 1, 2, 2, 2, ],
                  [0, 0, 0, 1, 2, 3, 2, ],
                  [0, 0, 0, 1, 1, 1, 1, ],
                  [1, 0, 0, 1, 1, 1, 1, ],
                  [2, 2, 2, 2, 1, 1, 3, ],
                  [2, 2, 2, 2, 3, 3, 4]]
                 ])


reshaped_data = np.transpose(data, (1, 2, 0))

layer_0 = reshaped_data[:, :, 0]



test_sam_data = {}
test_sam_data['sam_seg_z'] = reshaped_data

test_sam_data = fine_tune_3d_masks(test_sam_data, search_size=3)
out = test_sam_data['sam_seg_z']
layer_0 = out[:, :, 0]
print(out[:, :, 0])
print(out[:, :, 1])
print(out[:, :, 2])
print(out[:, :, 3])
# Import sammi module
from saami.SAAMI import SAAMI

# roi = ((70, 150), (700, 600))

SAAMIdata = SAAMI('data/MRI_example', roi=None)

# Calculates 3D mask for the first volume (idx = 0)
mask = SAAMIdata.calculate_3d_mask(0, threshold= 0.003)

SAAMIdata.save_mask(0, save_path='outputs/saved_sam_mask.nii')

# Fine-tune 3D mask for the first volume
new_mask = SAAMIdata.finetune_3d_mask(0, neighbor_size=3)

# # Save 3D mask for the first volume
SAAMIdata.save_mask(0, save_path='outputs/saved_sam_mask_adjusted.nii')

# visualization for the first volume
SAAMIdata.visualize(0)


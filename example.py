from saami.SAAMI import SAAMI

SAAMIdata = SAAMI('data/MRI_example')

# Calculates 3D mask for the first volume (idx = 0)
mask = SAAMIdata.calculate_3d_mask(0)

# Fine-tune 3D mask for the first volume
new_mask = SAAMIdata.finetune_3d_mask(0)

# Save 3D mask for the first volume
SAAMIdata.save_mask(0, save_path='outputs/saved_mask.nii')

# visualization for the first volume
SAAMIdata.visualize(0)

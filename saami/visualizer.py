import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
import os  # Import the os module to handle file paths

def visualize_volume_SAM(data_dict, show_widget=False, save_image=True, save_path=""):
    
    images = data_dict["image"]
    labels = data_dict["gt_label"]
    max_slice = images.shape[2] - 1

    masks_z = data_dict['sam_seg_z']

    # Function to update the plot based on the slider value
    def get_plot(slice_idx):

        image = images[:, :, slice_idx]
        gt_label = labels[:, :, slice_idx]
        sam_label = masks_z[:, :, slice_idx]

        fig, axes = plt.subplots(2, 3, figsize=(15, 15))
        (ax1, ax2, ax3), (ax4, ax5, ax6) = axes

        ax1.imshow(image, cmap="gray", aspect="equal")
        ax1.set_title("Original Image")
        ax1.axis("off")

        label_img = ax2.imshow(gt_label, cmap="jet", aspect="equal")
        ax2.set_title("Ground Truth Label")
        ax2.axis("off")
        
        ax3.imshow(image, cmap="gray", aspect="equal")
        ax3.imshow(gt_label, cmap="jet", alpha=0.5, aspect="equal")
        ax3.set_title("Ground Truth Overlay")
        ax3.axis("off")

        ax4.imshow(image, cmap="gray", aspect="equal")
        ax4.set_title("Original Image")
        ax4.axis("off")

        label_img = ax5.imshow(sam_label, cmap="jet", aspect="equal")
        ax5.set_title("SAM-Mask Label")
        ax5.axis("off")
        
        ax6.imshow(image, cmap="gray", aspect="equal")
        ax6.imshow(sam_label, cmap="jet", alpha=0.5, aspect="equal")
        ax6.set_title("SAM-Mask Overlay")
        ax6.axis("off")

        fig.subplots_adjust(wspace=0.2, hspace=-0.4)
        
        if show_widget:
            plt.show()

        # Save the plot if save_image is True
        if save_image:
            # Create the save_path directory if it does not exist
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            # Construct the filename for the current slice
            filename = os.path.join(save_path, f"Slice_{slice_idx}_visualization.jpg")
            # Save the plot to the specified file
            fig.savefig(filename)

    if show_widget:
        slider = widgets.IntSlider(min=0, max=max_slice, step=1, value=0)
        widgets.interact(get_plot, slice_idx=slider)
    else:
        for i in range(max_slice):
            get_plot(i) 
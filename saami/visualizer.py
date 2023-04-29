import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
import os
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import Normalize


def visualize_volume_SAM(data_dict, show_widget=False, show_tkinter=False, save_path="", axis='z'):

    images = data_dict["image"]
    labels = data_dict["label"]

    masks = data_dict['sam_seg'][axis]
    if axis == 'x':
        max_slice = images.shape[0] - 1

    elif axis == 'y':
        max_slice = images.shape[1] - 1

    elif axis == 'z':
        max_slice = images.shape[2] - 1

    max_mask_value = np.amax(masks)


    # Function to update the plot based on the slider value
    def get_plot(fig, slice_idx):

        # Clear previous image for the GUI
        fig.clear()

        if axis == 'x':
            image = images[slice_idx, :, :]
            gt_label = labels[slice_idx, :, :]
            sam_label = masks[slice_idx, :, :]
        elif axis == 'y':
            image = images[:, slice_idx, :]
            gt_label = labels[:, slice_idx, :]
            sam_label = masks[:, slice_idx, :]
        elif axis == 'z':
            image = images[:, :, slice_idx]
            gt_label = labels[:, :, slice_idx]
            sam_label = masks[:, :, slice_idx]

        aspect='auto'
        # Set fixed color map range for the labels
        vmin, vmax = 0, max_mask_value
        norm = Normalize(vmin=vmin, vmax=vmax)

        axes = fig.subplots(2, 3)
        (ax1, ax2, ax3), (ax4, ax5, ax6) = axes

        ax1.imshow(image, cmap="gray", aspect=aspect)
        ax1.set_title("Original Image")
        ax1.axis("off")

        label_img = ax2.imshow(gt_label, cmap="jet", aspect=aspect, norm=norm)
        ax2.set_title("Ground Truth Label")
        ax2.axis("off")

        ax3.imshow(image, cmap="gray", aspect="equal")
        ax3.imshow(gt_label, cmap="jet", alpha=0.5, aspect=aspect, norm=norm)
        ax3.set_title("Ground Truth Overlay")
        ax3.axis("off")

        ax4.imshow(image, cmap="gray", aspect=aspect)
        ax4.set_title("Original Image")
        ax4.axis("off")

        label_img = ax5.imshow(sam_label, cmap="jet", aspect=aspect, norm=norm)
        ax5.set_title("SAM-Mask Label")
        ax5.axis("off")

        ax6.imshow(image, cmap="gray", aspect=aspect)
        ax6.imshow(sam_label, cmap="jet", alpha=0.5, aspect=aspect, norm=norm)
        ax6.set_title("SAM-Mask Overlay")
        ax6.axis("off")

        fig.subplots_adjust(wspace=0.2, hspace=0.2)

        if show_tkinter:
            canvas.draw()

    # Show the ipy widget (which works in notebook environemnt
    if show_widget:
        fig = plt.figure(figsize=(15, 15))
        slider = widgets.IntSlider(min=0, max=max_slice, step=1, value=0)
        widgets.interact(lambda slice_idx: get_plot(fig, slice_idx), slice_idx=slider)

    # Show the tinker GUI
    if show_tkinter:
        window = tk.Tk()
        window.title("Volume Visualization")

        fig = plt.figure(figsize=(15, 15))
        canvas = FigureCanvasTkAgg(fig, master=window)
        canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

        # Create a frame for the slider and button
        control_frame = tk.Frame(window)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y)

        # Slider
        slider = tk.Scale(control_frame, from_=0, to=max_slice, orient=tk.VERTICAL,
                          command=lambda s: get_plot(fig, int(s)))
        slider.pack(side=tk.TOP, pady=10)

        # Save Image button
        def save_image():
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            filename = os.path.join(save_path, f"Slice_{slider.get()}_visualization.jpg")
            print('saving file to {}'.format(filename))
            fig.savefig(filename)

        save_button = tk.Button(control_frame, text="Save Image", command=save_image)

        save_button.pack(side=tk.TOP, pady=10)

        # Callback function to handle window close event
        def on_close():
            window.destroy()
            plt.close(fig)

        # Bind the callback function to the window close event
        window.protocol("WM_DELETE_WINDOW", on_close)

        get_plot(fig, 0)
        window.mainloop()

    # Otherwise just run through the volume and save the images
    if not show_tkinter and not show_widget:
        fig = plt.figure(figsize=(15, 15))
        for i in range(max_slice):
            get_plot(fig, i)
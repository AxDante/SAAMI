
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.widgets import Button

def np_array_to_image(np_array):
    np_array = np_array.astype(np.uint8)
    return np_array

def get_npz_files(folder_path):
    npz_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.npz')]
    return npz_files

def update_visualization(npz_file, img_axes):
    data = np.load(npz_file)

    image = np_array_to_image(data['image'])
    label = np_array_to_image(data['label'])
    sam_seg_2D_data = data['sam_seg_2D']
    sam_seg_2D_data = (sam_seg_2D_data - sam_seg_2D_data.min()) / (sam_seg_2D_data.max() - sam_seg_2D_data.min()) * 255
    sam_seg_2D = np_array_to_image(np.stack((sam_seg_2D_data,) * 3, axis=-1))
    

    img_axes[0].set_data(image)
    img_axes[1].set_data(label)
    img_axes[2].set_data(sam_seg_2D)

def visualize_npz_files(npz_files, index):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    ax1.set_title("Image")
    ax1.axis("off")

    ax2.set_title("Label")
    ax2.axis("off")

    ax3.set_title("SAM Seg 2D")
    ax3.axis("off")

    img1 = ax1.imshow(np.zeros((10, 10, 3), dtype=np.uint8))
    img2 = ax2.imshow(np.zeros((10, 10, 3), dtype=np.uint8))
    img3 = ax3.imshow(np.zeros((10, 10, 3), dtype=np.uint8))

    update_visualization(npz_files[index], (img1, img2, img3))

    ax_prev = plt.axes([0.4, 0.05, 0.1, 0.075])
    ax_next = plt.axes([0.5, 0.05, 0.1, 0.075])

    button_prev = Button(ax_prev, 'Previous')
    button_next = Button(ax_next, 'Next')

    def on_prev_clicked(event):
        nonlocal index
        index -= 1
        index = max(0, index)
        update_visualization(npz_files[index], (img1, img2, img3))
        fig.canvas.draw_idle()

    def on_next_clicked(event):
        nonlocal index
        index += 1
        index = min(len(npz_files) - 1, index)
        update_visualization(npz_files[index], (img1, img2, img3))
        fig.canvas.draw_idle()

    button_prev.on_clicked(on_prev_clicked)
    button_next.on_clicked(on_next_clicked)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    folder_path = "outputs"
    npz_files = get_npz_files(folder_path)
    if len(npz_files) > 0:
        visualize_npz_files(npz_files, 0)
    else:
        print("No npz files found in the specified folder.")







"Utility file for saami package"
import numpy as np

def download_progress_hook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    progress = min(100, (downloaded / total_size) * 100)
    print(f"\rDownload progress: {progress:.2f}%", end='')

def most_prevalent_labels(data):
    unique, counts = np.unique(data, return_counts=True)
    label_count = sorted(zip(unique, counts), key=lambda x: x[1], reverse=True)

    labels = [int(label) for label, count in label_count]
    occurrences = np.asarray([count for label, count in label_count])
    return labels, occurrences

def random_index_with_label(data, label):

    if len(data.shape) == 3:
        indices = np.argwhere(data == label)
        if indices.size == 0:
            return (-1, -1, -1)
        random_index = indices[np.random.choice(indices.shape[0])]
        return tuple(random_index)

    elif len(data.shape) == 2:
        indices = np.argwhere(data == label)
        if indices.size == 0:
            return (-1, -1)
        random_index = indices[np.random.choice(indices.shape[0])]
        return tuple(random_index)
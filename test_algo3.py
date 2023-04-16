import numpy as np

import numpy as np



test_arr = np.array([
    [
        [1, 1, 1, 1, 2, 2, 2, ],
        [1, 1, 1, 1, 2, 2, 2, ],
        [1, 1, 1, 1, 2, 3, 2, ],
        [1, 1, 1, 1, 3, 3, 3, ],
        [1, 1, 1, 1, 3, 3, 3, ],
        [2, 2, 2, 2, 3, 3, 3, ],
        [2, 2, 2, 2, 3, 4, 0, ]
    ],
    [
        [4, 4, 4, 4, 2, 2, 2, ],
        [4, 4, 4, 4, 2, 2, 2, ],
        [4, 4, 4, 4, 2, 3, 2, ],
        [4, 4, 4, 4, 1, 1, 1, ],
        [4, 4, 4, 4, 1, 1, 1, ],
        [2, 2, 2, 2, 1, 1, 1, ],
        [2, 2, 2, 2, 1, 1, 0, ]
    ],
    [
        [1, 1, 1, 1, 2, 2, 2, ],
        [1, 1, 1, 1, 2, 2, 2, ],
        [1, 1, 1, 1, 2, 3, 2, ],
        [1, 1, 1, 1, 3, 3, 3, ],
        [1, 1, 1, 1, 3, 3, 3, ],
        [2, 2, 2, 2, 3, 3, 3, ],
        [2, 2, 2, 2, 3, 3, 4, ]
    ],
    [
        [0, 0, 0, 1, 2, 2, 2, ],
        [0, 0, 0, 1, 2, 2, 2, ],
        [0, 0, 0, 1, 2, 3, 2, ],
        [0, 0, 0, 1, 1, 1, 1, ],
        [1, 0, 0, 1, 1, 1, 1, ],
        [2, 2, 2, 2, 1, 1, 3, ],
        [2, 2, 2, 2, 3, 3, 4]
    ]
])

def calculate_mapping(array_1, array_2, num_labels):
    if array_1.shape != array_2.shape:
        raise ValueError("The input arrays should have the same shape.")

    mapping = np.zeros((num_labels, num_labels), dtype=int)

    for i in range(array_1.shape[0]):
        for j in range(array_1.shape[1]):
            val_1 = array_1[i, j]
            val_2 = array_2[i, j]
            mapping[val_1, val_2] += 1

    return mapping


def find_largest_indices(arr):
    # Flatten the array and get the indices that would sort it in descending order
    sorted_indices = np.argsort(arr.flatten())[::-1]

    # Convert the flattened indices to 2D indices
    sorted_2d_indices = np.unravel_index(sorted_indices, arr.shape)

    # Combine the 2D indices and return them as a list of tuples
    return list(zip(sorted_2d_indices[0], sorted_2d_indices[1]))


def modify_layer(array, mapping, max_labels):
    # Find the majority mapping for each label in the first array
    print(mapping)
    m_array = np.full(array.shape, -1)
    ilist = find_largest_indices(mapping)
    print('indices_list')
    print(ilist)

    assign_queue = [i for i in range(max_labels)]
    alist = []
    while len(ilist) > 0:
        pval, cval = ilist.pop(0)
        valid = not(any(pval == t[0] for t in alist) or any(cval == t[1] for t in alist))
        if valid:
            alist.append((pval, cval))

    # For each assignment in final assignment list
    for a in alist:
        m_array[array==a[1]] = a[0]

    return m_array


data = test_arr
num_labels = np.amax(data) + 1

for i in range(data.shape[0] - 1):
    mapping = calculate_mapping(data[i, :, :], data[i + 1, :, :], num_labels)
    data[i + 1, :, :] = modify_layer(data[i + 1, :, :], mapping, num_labels)
print('-------------------')
for i in range(data.shape[0]):
    print(data[i, :, :])

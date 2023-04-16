import numpy as np

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

def modify_layer(array, mapping):
    # Find the majority mapping for each label in the first array
    majority_mapping = np.argmax(mapping, axis=1)

    # Create a lookup table for modifying the input array based on the majority mapping
    lookup_table = np.arange(mapping.shape[0])
    for i, majority_label in enumerate(majority_mapping):
        lookup_table[i] = majority_label

    # Modify the input array based on the lookup table
    modified_array = lookup_table[array]

    return modified_array



# Example usage:
array_1 = np.array([[0, 1, 2], [3, 4, 0], [1, 2, 3]])
array_2 = np.array([[0, 1, 1], [2, 4, 0], [0, 2, 3]])
num_labels = 5



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
          [2, 2, 2, 2, 3, 3, 4 ]
    ]
])


#
# print(test_arr.shape)
# mapping = calculate_mapping(test_arr[0, :, :], test_arr[1, :, :], num_labels)
# print(mapping)
# modified_array = modify_layer(test_arr[1, :, :], mapping)
# print(modified_array)

#data = test_arr.copy()




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
    modified_array = array.copy()

    indices_list = find_largest_indices(array)
    print(indices_list)

    assign_queue = [i for i in range(max_labels)]
    print(assign_queue)

    while len(assign_queue) > 0:
        print('Current assign queue')
        print(assign_queue)

        prev_label, cur_label = indices_list.pop(0)
        modified_array[array==cur_label] = prev_label


        print('Current indices_list')
        print(indices_list)

        print('Current cur_label')
        print(cur_label)
        assign_queue = np.delete(assign_queue, cur_label)


    return modified_array


def fine_tune_3d_masks(data_dict, main_axis='z'):

    data = data_dict['sam_seg_{}'.format(main_axis)]
    data_shape = data.shape

    adj_data = data.copy()
    max_labels = np.amax(adj_data).astype(int)

    for rz in range(data_shape[2]):
        mapping = calculate_mapping(adj_data[:, :, rz], data[:, :, rz + 1], max_labels)

        adj_data[:, :, rz + 1] = modify_layer(adj_data[:, :, rz + 1], mapping, max_labels)

    data_dict['sam_seg_{}'.format(main_axis)] = adj_data

    return data_dict


data = np.random.randint(5, size=(3, 10, 10))
data = test_arr
num_labels = np.amax(data) + 1

for i in range(data.shape[0]):
    print(data[i, :, :])

for i in range(data.shape[0] - 1):
    mapping = calculate_mapping(data[i, :, :], data[i+1, :, :], num_labels)
    print(mapping.shape)
    data[i + 1, :, :] = modify_layer(data[i + 1, :, :], mapping, num_labels)
print('-------------------')
for i in range(data.shape[0]):
    print(data[i, :, :])

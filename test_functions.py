import numpy as np


def check_grid(arr, rx, ry, ns=2, method='point'):
    if method == 'point':
        if all(arr[rx, ry] == value for value in
            [arr[rx - ns, ry], arr[rx + ns, ry], arr[rx, ry - ns], arr[rx, ry + ns]]):
            return True
        else:
            return False

def calculate_mapping(array_1, array_2, num_labels, neighbor_size=0):
    if array_1.shape != array_2.shape:
        raise ValueError("The input arrays should have the same shape.")

    mapping = np.zeros((num_labels + 1, num_labels + 1), dtype=int)

    for i in range(array_1.shape[0]):
        for j in range(array_1.shape[1]):
            val_1 = array_1[i, j].astype(int)
            val_2 = array_2[i, j].astype(int)
            try:
                c1 = check_grid(array_1, i, j, ns=neighbor_size)
                c2 = check_grid(array_2, i, j, ns=neighbor_size)
                if c1 and c2:
                    mapping[val_1, val_2] += 1
                else:
                    j = j + neighbor_size
            except:
                pass

    return mapping


def find_largest_indices(arr):
    # Flatten the array and get the indices that would sort it in descending order
    si = np.argsort(arr.flatten())[::-1]

    # Convert the flattened indices to 2D indices
    si_2d = np.unravel_index(si, arr.shape)

    # Initialize an empty list for the result
    result = []

    # Iterate through the sorted indices and check if the corresponding value is not 0
    for i in range(len(si)):
        if arr[si_2d[0][i], si_2d[1][i]] != 0:
            # Add the non-zero value's indices as a tuple to the result list
            result.append((si_2d[0][i], si_2d[1][i]))

    return result

# Handles the propagated information from previous layer
def modify_layer(array, mapping):
    # Find the majority mapping for each label in the first array
    m_array = np.full(array.shape, -1)
    ilist = find_largest_indices(mapping)

    alist = []
    vlist = []
    while len(ilist) > 0:
        pval, cval = ilist.pop(0)
        if cval not in vlist:
            alist.append((pval, cval))
            vlist.append(cval)

        # valid = not (any(pval == t[0] for t in alist) or any(cval == t[1] for t in alist))
        # if valid:
        #     alist.append((pval, cval))

    # For each assignment in final assignment list
    for a in alist:
        m_array[array == a[1]] = a[0]

    return m_array


arr_1 = np.array([[0, 0, 0, 0, 0], 
                  [0, 1, 3, 3, 0],
                  [0, 1, 3, 4, 4],
                  [0, 1, 2, 2, 4],
                  [0, 0, 0, 0, 0]])
arr_2 = np.array([[0, 0, 0, 0, 0], 
                  [0, 4, 2, 2, 2],
                  [0, 4, 2, 3, 3],
                  [1, 1, 2, 2, 2],
                  [1, 1, 1, 1, 1]])

mapping = calculate_mapping(arr_1, arr_2, 4, neighbor_size=0)
print(mapping)
ilist = find_largest_indices(mapping)
print(ilist)
adj_data = modify_layer(arr_2, mapping)
print(adj_data)

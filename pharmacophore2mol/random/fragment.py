import numpy as np

def fragment_3d_array(array, block_size, stride):
    """
    Fragments a 3D numpy array into overlapping sub-arrays.

    Parameters:
    - array (np.ndarray): Input 3D numpy array.
    - block_size (int): The size of each sub-array block (assumed cubic, e.g., 3 for 3x3x3).
    - stride (int): The step size for moving the window.

    Returns:
    - np.ndarray: A 4D array containing the fragmented blocks. The shape will be
                  (num_blocks, block_size, block_size, block_size).
    """
    z_max, y_max, x_max = array.shape
    blocks = []

    for z in range(0, z_max - block_size + 1, stride):
        for y in range(0, y_max - block_size + 1, stride):
            for x in range(0, x_max - block_size + 1, stride):
                block = array[z:z + block_size, y:y + block_size, x:x + block_size]
                blocks.append(block)

    return np.array(blocks)


if __name__ == "__main__":
    arr = np.random.rand(5, 5, 5)
    blocks = fragment_3d_array(arr, block_size=3, stride=1)
    print(blocks.shape)
    print(blocks[0])
    print(blocks[1])
    print(blocks[2])
    # print(blocks[3])

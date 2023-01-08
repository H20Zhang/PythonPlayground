import torch
import numpy as np


if __name__ == "__main__":

    data = [[1, 2], [3, 4]]
    np_array = np.array(data)
    print(f"data:{data}")
    print(f"np_array:{np_array}")

    # create tensor from array
    x_data = torch.tensor(data)
    print(f"x_data:{x_data}")

    # create tensor from numpy array
    x_np = torch.from_numpy(np_array)
    print(f"x_np:{x_np}")

    # create tensor from another tensor
    x_ones = torch.ones_like(x_data)
    x_rand = torch.rand_like(x_data, dtype=torch.float)
    print(f"x_ones:{x_ones}")
    print(f"x_rand:{x_rand}")

    # create tensor with given shape
    shape = (2, 3, )
    rand_tensor = torch.rand(shape)
    ones_tensor = torch.ones(shape)
    zeros_tensor = torch.zeros(shape)
    print(f"shape:{shape}")
    print(f"rand_tensor:{rand_tensor}")
    print(f"ones_tensor:{ones_tensor}")
    print(f"zeros_tensor:{zeros_tensor}")

    # get attributes of a tensor
    tensor = torch.rand(3, 4)
    print(f"shape:{tensor.shape}")
    print(f"datatype:{tensor.dtype}")
    print(f"device:{tensor.device}")

    # tensor operation - indexing, modification, and slicing
    tensor = torch.rand(4, 4)
    first_row = tensor[0]
    first_col = tensor[..., 0]
    last_col = tensor[:, -1]

    print(f"tensor:{tensor}")
    print(f"first_row:{first_row}")
    print(f"first_col:{first_col}")
    print(f"last_col:{last_col}")

    mod_tensor = tensor.clone().detach()
    mod_tensor[..., 1] = 1
    print(f"tensor:{tensor} \n modified_tensor:{mod_tensor}")

    cat_tensor_wide = torch.cat([tensor, tensor, tensor], dim=1)
    cat_tensor_tall = torch.cat([tensor, tensor, tensor], dim=0)
    print(f"cat_tensor_wide:{cat_tensor_wide} \n cat_tensor_tall:{cat_tensor_tall}")

    # tensor operation - arithmetic operations
    tensor = torch.rand(4, 2)
    t1 = tensor.T  # transpose
    t2 = tensor.matmul(t1)  # matrix multiplication
    t3 = tensor.mul(tensor)  # element-wise multiplication
    t4 = tensor.sum()  # tensor-wise sum
    t5 = t4.item()  # collect single value from the tensor

    print(f"tensor:{tensor}")
    print(f"t1:{t1}")
    print(f"t2:{t2}")
    print(f"t3:{t3}")
    print(f"t4:{t4}")
    print(f"t5:{t5}")

    tensor.add_(5)  # in-place add, inplace operations are denoted by a _ suffix
    print(f"tensor after in-place add:{tensor}")


    # Bridge with NumPy
    t1 = torch.ones(5)
    n1 = t1.numpy()  # from Tensor to NumPy array
    print(f"t1:{t1} \n n1:{n1}")

    n2 = np.ones(6)
    t2 = torch.from_numpy(n2)  # from NumPy array to Tensor
    print(f"n2:{n2} \n t2:{t2}")

    # the numpy array's memory address is shared with Tensor
    t1.add_(1)
    print(f"t1:{t1} \n n1:{n1}")

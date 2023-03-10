import torch
import numpy as np

if __name__ == '__main__':
    print(torch.has_cuda)
    x = torch.rand(10000, 10000).cuda()
    y = torch.rand(10000, 10000).cuda()

    torch.cuda.is_available()
    # convert tensor x to numpy array and print out the shape of the array
    print(x.cpu().numpy().shape)
    # convert tensor y to numpy array and print out the shape of the array
    print(y.cpu().numpy().shape)

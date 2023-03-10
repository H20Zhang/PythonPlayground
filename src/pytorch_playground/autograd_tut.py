import torch

if __name__ == '__main__':
    x = torch.ones(5)
    y = torch.zeros(3)
    w = torch.randn(5, 3, requires_grad=True)
    b = torch.randn(3, requires_grad=True)
    z = torch.matmul(x, w) + b
    loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

    print(f"Gradient function for z = {z.grad_fn}")
    print(f"Gradient function for loss = {loss.grad_fn}")

    #  Perform the backward propagation.
    #  We can only obtain the grad properties for the leaf nodes of the computation graph, which have requires_grad property
    #  set to true.
    #  We can only perform gradient calculations using backward once on a given graph.
    loss.backward()
    print(w.grad)
    print(b.grad)

    #  We can disable the tracking of gradient
    #  The tracking of gradient needs to be disabled
    #       1. inference
    #       2. mark some parameters in the neural network as frozen parameters (i.e., pretrained network)
    #       3. speed up computation when only doing forward pass
    with torch.no_grad():
        z = torch.matmul(x, w) + b
        print(z.requires_grad)

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# get the device for training
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    r"""
    Example neural network layer
    """

    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        #  call flatten(), instead of flatten.forward, as flatten() is equivalent to __call__, which also invokes hooks
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1).item()
print(f"Predicated class: {y_pred}")

#  Print the parameter of the model
print(f"Model structure:{model}")
for name, param in model.named_parameters():
    print(f"Layer:{name} | Size:{param.size()} | Values:{param[:2]} \n")
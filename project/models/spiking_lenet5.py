import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .base_layers import ConvBnSpike, ConvSpike, LinearSpike
from spikingjelly.clock_driven import surrogate, neuron, functional, layer


class SpikingLeNet5(nn.Module):
    """LeNet5-like model using Spiking Neurons. Warning: original MaxPool2D are replaced by Strided Convolution in this implementation"""

    def __init__(self, in_channels: int, num_classes: int, neuron_model: str = "LIF", bias=True):
        super(SpikingLeNet5, self).__init__()
        self.conv1 = ConvBnSpike(in_channels, 20, kernel_size=5, neuron_model=neuron_model)
        self.stride1 = ConvSpike(20, 20, kernel_size=3, stride=2, bias=bias, neuron_model=neuron_model)

        self.conv2 = ConvBnSpike(20, 50, kernel_size=5, neuron_model=neuron_model)
        self.stride2 = ConvSpike(50, 50, kernel_size=3, stride=2, bias=bias, neuron_model=neuron_model)

        # flatten at 2nd dimension because dim0 and dim1 are Timesteps and Batch, respectively
        self.flat = nn.Flatten(start_dim=2)

        self.fc1 = LinearSpike(
            in_channels=4050,  # number of channels for 34x34 event frames from N-MNIST
            out_channels=120,
            bias=bias,
            neuron_model=neuron_model
        )

        self.fc_final = nn.Linear(120, num_classes, bias=True)

    def forward(self, x):
        """

        Args:
            x (torch.Tensor): input tensor of dimension (T, B, C, H, W).

        Returns:
            torch.Tensor: Tensor (of logits) of dimension (B, num_classes)
        """

        x = self.conv1(x)
        x = self.stride1(x)
        x = self.conv2(x)
        x = self.stride2(x)
        x = self.flat(x)

        x = self.fc1(x)

        # .mean(0) is used to convert the Spiking Feature Maps (dim=[T, B, C]) to numerical values (dim=[B, C])
        # so that we can obtain some prediction
        x = x.mean(0)

        x = self.fc_final(x)
        return x

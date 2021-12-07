import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from spikingjelly.clock_driven import surrogate, layer, neuron


class ConvBnSpike(nn.Sequential):
    """Convolution + BatchNorm + spiking neuron activation. Accepts input of dimension (T, B, C, H, W)"""

    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, neuron_model="LIF"):
        super(ConvBnSpike, self).__init__()
        padding = kernel_size // 2 + dilation - 1

        self.add_module('conv_bn', layer.SeqToANNContainer(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=kernel_size,
                      padding=padding,
                      # no bias if we use BatchNorm (because it has a BN itself)
                      bias=False,
                      dilation=dilation,
                      stride=stride),
            nn.BatchNorm2d(out_channels)
        ))

        # surrogate gradient function to use during the backward pass.
        # it is fixed here.
        surr_func = surrogate.ATan(alpha=2.0, spiking=True)

        # The spiking neuron's hyperparameters are fixed
        if neuron_model == "PLIF":
            self.add_module('spike', neuron.MultiStepParametricLIFNode(detach_reset=True, surrogate_function=surr_func))
        elif neuron_model == "IF":
            self.add_module('spike', neuron.MultiStepIFNode(detach_reset=True, surrogate_function=surr_func))
        else:
            self.add_module('spike', neuron.MultiStepLIFNode(detach_reset=True, surrogate_function=surr_func))


class ConvSpike(nn.Sequential):
    """Convolution + BatchNorm + spiking neuron activation. Accepts input of dimension (T, B, C, H, W)"""

    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=True, neuron_model="LIF"):
        super(ConvSpike, self).__init__()
        padding = kernel_size // 2 + dilation - 1

        self.add_module('conv_bn', layer.SeqToANNContainer(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=kernel_size,
                      padding=padding,
                      bias=bias,  # REMEMBER : bias is not bio-plausible and hard to implement on neuromorphic hardware
                      dilation=dilation,
                      stride=stride),
            nn.BatchNorm2d(out_channels)
        ))

        # surrogate gradient function to use during the backward pass.
        # it is fixed here.
        surr_func = surrogate.ATan(alpha=2.0, spiking=True)

        # The spiking neuron's hyperparameters are fixed
        if neuron_model == "PLIF":
            self.add_module('spike', neuron.MultiStepParametricLIFNode(detach_reset=True, surrogate_function=surr_func))
        elif neuron_model == "IF":
            self.add_module('spike', neuron.MultiStepIFNode(detach_reset=True, surrogate_function=surr_func))
        else:
            self.add_module('spike', neuron.MultiStepLIFNode(detach_reset=True, surrogate_function=surr_func))


class LinearSpike(nn.Sequential):
    """FC layer + spiking neuron activation. Accepts input of dimension (T, B, C)"""

    def __init__(self, in_channels, out_channels, bias=True, neuron_model="LIF"):
        super(LinearSpike, self).__init__()

        self.add_module('fc', layer.SeqToANNContainer(
            nn.Linear(in_channels, out_channels, bias=bias)
        ))

        # surrogate gradient function to use during the backward pass.
        # it is fixed here.
        surr_func = surrogate.ATan(alpha=2.0, spiking=True)

        # The spiking neuron's hyperparameters are fixed
        if neuron_model == "PLIF":
            self.add_module('spike', neuron.MultiStepParametricLIFNode(detach_reset=True, surrogate_function=surr_func))
        elif neuron_model == "IF":
            self.add_module('spike', neuron.MultiStepIFNode(detach_reset=True, surrogate_function=surr_func))
        else:
            self.add_module('spike', neuron.MultiStepLIFNode(detach_reset=True, surrogate_function=surr_func))

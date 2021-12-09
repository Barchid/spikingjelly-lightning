""" Implementation of spiking activation maps (SAM). Original paper: https://arxiv.org/pdf/2103.14441.pdf"""

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import cv2
from spikingjelly.clock_driven import neuron
import math
import matplotlib.pyplot as plt
from celluloid import Camera
import uuid


class SAM(object):
    """Computes the Spiking activation map (SAM) for one layer"""

    def __init__(self, layer, name: str, gamma=0.4):
        """[summary]

        Args:
            layer: Layer of spiking neurons that will be registered
            name (str): Name of the layer
            input_height (int): Height of the original input image.
            input_width (int): Width of the original input image.
            gamma (float, optional): GAMMA hyperparameter (see paper). Defaults to 0.4.
        """
        super(SAM, self).__init__()
        self.name = name
        self.hook = layer.register_forward_hook(self.hook_save_spikes)
        self.gamma = gamma
        self.spikes = None

    def hook_save_spikes(self, module, input, output):
        self.spikes = output.detach().cpu().numpy()

    def get_sam(self, input: torch.Tensor):
        """Gets the Spiking Activation Map for the input in parameters.

        Args:
            input (torch.Tensor): input tensor.Shapes can be (T, B, C, H, W) for an input spike train or (B, C, H, W) for an input static image

        Returns:
            [type]: [description]
        """
        assert self.spikes is not None
        assert len(input.shape) == 4 or len(input.shape) == 5  # input must be (B, C, H, W) or (T, B, C, H, W)

        # get height and width of original input images
        height = input.shape[-2]
        width = input.shape[-1]

        # preprocess input
        input = input.detach().cpu().numpy()
        if len(input.shape) == 4:
            input = [input] * self.spikes.shape[0]

        # Compute the SAM for each layer and each timesteps
        heatmaps = []

        # FOR EACH timesteps
        for t in range(1, len(self.spikes)):
            NCS = np.zeros_like(self.spikes[0])  # shape=(B, C, H, W)

            # previous timesteps (i.e. t_p < t)
            for t_p in range(0, t):
                mask = self.spikes[t_p] == 1.
                NCS[mask] += math.exp(-self.gamma * abs(t - t_p))

            M = np.sum(NCS * self.spikes[t], axis=1)
            heatmap = self._format_heatmap(M, height, width)
            heatmaps.append(heatmap)

        self.heatmap_video(input, heatmaps)

        # Resets the spikes record for another forward pass
        self.spikes = None

        return heatmaps

    def _format_heatmap(self, M: np.ndarray, height: int, width: int):
        batch = []

        # for each heatmap in the batch
        for i in range(M.shape[0]):

            # normalize between 0 and 1
            max = np.max(M[i])
            min = np.min(M[i])
            heatmap = (M[i] - min) / (max - min + 1e-7)

            # resize the heatmap
            heatmap = cv2.resize(heatmap, (width, height))

            batch.append(heatmap)

        return np.array(batch)

    def heatmap_video(self, input: List[np.ndarray], heatmaps: List[np.ndarray]):
        for j in range(len(heatmaps[0])):  # FOR EACH input IN BATCH
            fig, ax = plt.subplots()
            camera = Camera(fig)
            plt.axis("off")
            for i in range(len(heatmaps)):  # FOR EACH timestep
                # input[i+1] because the heatmaps list doesn't take the timestep=0 into account
                img_heatmap = show_cam_on_image(input[i + 1][j].transpose(1, 2, 0),
                                                heatmaps[i][j], use_rgb=input.shape[2] == 3)

                ax.imshow(img_heatmap)
                camera.snap()
            filename = f"batch{j}_{str(uuid.uuid4())}.mp4"
            anim = camera.animate(interval=40)
            anim.save(filename)


def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.

    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    # print(np.unique(np.uint8(255 * mask)))
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        print(np.unique(img))
        raise Exception("The input image should np.float32 in the range [0, 1]")

    img = np.squeeze(img)  # squeeze if needed
    if img.shape[-1] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[-1] == 2:
        img = np.stack((img[:, :, 0], img[:, :, 1], np.zeros_like(img[:, :, 0])), axis=2)

    cam = heatmap + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

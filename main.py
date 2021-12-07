from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl


def main():
    pass


if __name__ == "__main__":
    # seeds the random from numpy, pytorch, etc to obtain reproductibility
    pl.seed_everything(1234)

    # Program args
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--timesteps', type=int, required=True)
    
    # Args for model
    

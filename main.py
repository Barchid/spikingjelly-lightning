from argparse import ArgumentParser
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from project.datamodules.nmnist import NMNISTDataModule

from project.lenet5_module import Lenet5Module


def main():
    # seeds the random from numpy, pytorch, etc for reproductibility
    pl.seed_everything(1234)

    args = get_args()

    module = create_module(args)

    datamodule = create_datamodule(args)

    trainer = create_trainer(args)

    # Launch training/validation
    if args.mode == "train":
        trainer.fit(module, datamodule=datamodule, ckpt_path=args.ckpt_path)

        # report results in a txt file
        report_path = os.path.join(args.default_root_dir, 'train_report.txt')
        report = open(report_path, 'a')

        # TODO: add any data you want to report here
        # here, we put the model's hyperparameters and the resulting val accuracy
        report.write(
            f"SpikingLeNet5 {args.timesteps} {args.bias} {args.neuron_model} {args.learning_rate}  {trainer.checkpoint_callback.best_model_score}\n")
    elif args.mode == "lr_find":
        lr_finder = trainer.tuner.lr_find(module, datamodule=datamodule)
        fig = lr_finder.plot(suggest=True)
        fig.show()
        print(f'SUGGESTION IS :', lr_finder.suggestion())
    else:
        trainer.validate(module, datamodule=datamodule, ckpt_path=args.ckpt_path)


def create_module(args) -> pl.LightningModule:
    # vars() is required to pass the arguments as parameters for the LightningModule
    dict_args = vars(args)

    # TODO: you can change the module class here
    module = Lenet5Module(**dict_args)

    return module


def create_datamodule(args) -> pl.LightningDataModule:
    # vars() is required to pass the arguments as parameters for the LightningDataModule
    dict_args = vars(args)

    # TODO: you can change the datamodule here
    datamodule = NMNISTDataModule(**dict_args)

    return datamodule


def create_trainer(args) -> pl.Trainer:
    # saves the best model checkpoint based on the accuracy in the validation set
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_acc",  # TODO: select the logged metric to monitor the checkpoint saving
        filename="model-{epoch:03d}-{val_acc:.4f}",
        save_top_k=1,
        mode="max",
    )

    # create trainer
    trainer = pl.Trainer.from_argparse_args(args, callbacks=[checkpoint_callback])
    return trainer


def get_args():
    # Program args
    # TODO: you can add program-specific arguments here
    parser = ArgumentParser()
    parser.add_argument('--mode', type=str, choices=["train", "validate", "lr_find"], default="train")
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help="Path of a checkpoint file. Defaults to None, meaning the training/testing will start from scratch.")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--timesteps', type=int, required=True)

    # Args for model
    parser = Lenet5Module.add_model_specific_args(parser)

    # Args for Trainer
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()

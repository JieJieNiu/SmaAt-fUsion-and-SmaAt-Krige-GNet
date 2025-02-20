import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)
from lightning.pytorch import loggers
import argparse
from models import unet_precip_regression_lightning as unet_regr
from lightning.pytorch.tuner import Tuner
from pathlib import Path
from root import ROOT_DIR
from torchinfo import summary
import torch

torch.set_float32_matmul_precision('medium')

def train_regression(hparams, find_batch_size_automatically: bool = False):

    if hparams.model == "Node_SmaAt_bridge":
        net = unet_regr.Node_SmaAt_bridge(hparams = hparams)
    elif hparams.model == "Krige_GNet":
        net = unet_regr.Krige_GNet(hparams = hparams)
    else:
        raise NotImplementedError(f"Model '{hparams.model}' not implemented")

    default_save_path = ROOT_DIR / "lightning_results"
    tb_logger = loggers.TensorBoardLogger(save_dir=default_save_path, name=net.__class__.__name__)
    
    
    checkpoint_callback = ModelCheckpoint(
        dirpath= Path(tb_logger.log_dir),
        filename=net.__class__.__name__ + "_rain_threshold_50_{epoch}-{val_loss:.6f}",
        save_top_k=-1,
        verbose=False,
        monitor="val_loss",
        mode="min",
    )
    
    lr_monitor = LearningRateMonitor()
    earlystopping_callback = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=hparams.es_patience,
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        fast_dev_run=hparams.fast_dev_run,
        max_epochs=hparams.epochs,
        default_root_dir=default_save_path,
        logger=tb_logger,
        callbacks=[checkpoint_callback, earlystopping_callback, lr_monitor],
        val_check_interval=hparams.val_check_interval,
    )

    if find_batch_size_automatically:
        tuner = Tuner(trainer)

        # Auto-scale batch size by growing it exponentially (default)
        tuner.scale_batch_size(net, mode="binsearch")

    # This can be used to speed up training with newer GPUs:
    # https://lightning.ai/docs/pytorch/stable/advanced/speed.html#low-precision-matrix-multiplication
    # torch.set_float32_matmul_precision('medium')
    #summary(net, input_size=[(32, 12, 64, 64),(32, 12, 22, 8)])

    trainer.fit(model=net, ckpt_path=hparams.resume_from_checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    #parser = unet_regr.Kriging_regression_base.add_model_specific_args(parser)
    parser = unet_regr.node_regression_base.add_model_specific_args(parser)
    
    parser.add_argument(
        "--dataset_folder",
        #default = ROOT_DIR / "data" / "precipitation" / "Krige_2014-2023.h5"
        default = ROOT_DIR / "data" / "precipitation" / "Node_2014-2023.h5",
        type=str,
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--fast_dev_run", type=bool, default=False)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--val_check_interval", type=float, default=None)

    args = parser.parse_args()

    # args.fast_dev_run = True
    args.n_channels = 12
    # args.gpus = 1
    args.model = "Node_SmaAt_bridge"
    #args.model = "Krige_GNet"
    #args.model = "SmaAt_UNet"
    args.lr_patience = 8
    args.es_patience = 12
    # args.val_check_interval = 0.25
    args.kernels_per_layer = 2
    args.use_oversampled_dataset = True
    args.dropout=0.5
    args.dataset_folder = (
        #ROOT_DIR / "data" / "precipitation" / "Krige_2016-2019.h5"
        ROOT_DIR / "data" / "precipitation" / "Node_2016-2019.h5"

    )

    train_regression(args, find_batch_size_automatically=False)


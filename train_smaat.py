import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)
import torch
from lightning.pytorch import loggers
from pytorch_lightning.profilers import SimpleProfiler
import argparse
from models import unet_precip_regression_lightning as unet_regr
from lightning.pytorch.tuner import Tuner
from pathlib import Path
from root import ROOT_DIR
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

torch.set_float32_matmul_precision('medium')

def train_regression(hparams, find_batch_size_automatically: bool = False):

    if hparams.model == "UNetDS_Attention":
        net = unet_regr.UNetDS_Attention(hparams=hparams)
    elif hparams.model == "UNet_Attention":
        net = unet_regr.UNet_Attention(hparams=hparams)
    elif hparams.model == "UNet":
        net = unet_regr.UNet(hparams=hparams)
    elif hparams.model == "UNetDS":
        net = unet_regr.UNetDS(hparams=hparams)
    elif hparams.model == "SmaAt_UNet":
        net = unet_regr.SmaAt_UNet(hparams=hparams)
    elif hparams.model == "Node_SmaAt_bridge":
        net = unet_regr.Node_SmaAt_bridge(hparams = hparams)
    elif hparams.model == "Krige_GNet":
        net = unet_regr.Krige_GNet(hparams = hparams)
    elif hparams.model == "Hybrid_UNet_2":
        net = unet_regr.Hybrid_SmaAt_2(hparams = hparams)
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

    profiler = SimpleProfiler()
    
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        profiler=profiler,
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

    trainer.fit(model=net, ckpt_path=hparams.resume_from_checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    #parser = unet_regr.Kriging_regression_base.add_model_specific_args(parser)
    parser = unet_regr.Precip_regression_base.add_model_specific_args(parser)
    
    parser.add_argument(
        "--dataset_folder",
        default= ROOT_DIR / "data" / "precipitation" / "Node_2014-2023.h5",
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
    #args.model = "Hybrid_UNet"
    #args.model = "Krige_GNet"
    args.model = "SmaAt_UNet"
    args.lr_patience = 8
    args.es_patience = 12
    # args.val_check_interval = 0.25
    args.kernels_per_layer = 2
    args.use_oversampled_dataset = True
    args.dropout=0.5
    args.dataset_folder = (
        ROOT_DIR / "data" / "precipitation" / "Node_2016-2019.h5"
    )
    # args.resume_from_checkpoint = f"lightning/precip_regression/{args.model}/UNetDS_Attention.ckpt"
    print(torch.cuda.memory_summary())
    torch.cuda.empty_cache()
    train_regression(args, find_batch_size_automatically=False)

    # All the models below will be trained
    # for m in ["SmaAt_UNet","UNet", "UNetDS", "UNet_Attention", "UNetDS_Attention"]:
    #     args.model = m
    #     print(f"Start training model: {m}")
    #     train_regression(args, find_batch_size_automatically=False)

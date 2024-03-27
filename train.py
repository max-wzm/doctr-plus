import argparse
import gc
import os
from datetime import datetime

import torch
from torch.utils.data import DataLoader

import wandb
from data_process.mixed_dataset import MixedDataset
from data_process.qb_dataset import QbDataset
from data_process.utils import get_unwarp, tensor_unwarping
from data_process.uvdoc_dataset import UVDocDataset
from network.core_net import GeoTr

os.environ["WANDB_API_KEY"] = "7974567f16cc72e73931e4bfb12c157156ab7109"
gamma_w = 0.0

import logging

logger: logging.Logger = None


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def log(content):
    logger.info(content)


def setup_data(args):
    t_UVDoc_data = QbDataset(
        appearance_augmentation=args.appearance_augmentation,
        geometric_augmentations=args.geometric_augmentationsUVDoc,
        split="train"
    )
    v_UVDoc_data = QbDataset(split="val")
    train_loader = DataLoader(
        dataset=t_UVDoc_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        dataset=v_UVDoc_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_lr_scheduler(optimizer, args, epoch_start):
    """Return a learning rate scheduler
    Parameters:
        optimizer          -- the optimizer of the network
        args               -- stores all the experiment flags
        epoch_start        -- the epoch number we started/continued from
    We keep the same learning rate for the first <args.n_epochs> epochs
    and linearly decay the rate to zero over the next <args.n_epochs_decay> epochs.
    """

    def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch + epoch_start - args.n_epochs) / float(
            args.n_epochs_decay + 1
        )
        return lr_l

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

    return scheduler


def update_learning_rate(scheduler, optimizer):
    """Update learning rates; called at the end of every epoch"""
    old_lr = optimizer.param_groups[0]["lr"]
    scheduler.step()
    lr = optimizer.param_groups[0]["lr"]
    log("learning rate update from %.7f -> %.7f" % (old_lr, lr))
    return lr


def main_worker(args):
    train_loader, val_loader = setup_data(args)
    device = torch.device("cuda:0")

    net = GeoTr()
    net.to(device)

    l1_loss = torch.nn.L1Loss()
    mse_loss = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999))

    global gamma_w
    epoch_start = 0

    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)

    current_time = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    experiment_name = (
        "time_"
        + current_time
        + "params"
        + str(args.batch_size)
        + "_lr="
        + str(args.lr)
        + "_nepochs"
        + str(args.n_epochs)
        + "_nepochsdecay"
        + str(args.n_epochs_decay)
        + "_alpha"
        + str(args.alpha_w)
        + "_gamma="
        + str(args.gamma_w)
        + "_gammastartep"
        + str(args.ep_gamma_start)
        + "_data"
        + args.data_to_use
    )

    if args.resume:
        experiment_name = "RESUME" + experiment_name
    log_file_name = os.path.join(args.logdir, experiment_name + ".log")
    global logger
    logger = get_logger(log_file_name)

    if args.resume is not None:
        if os.path.isfile(args.resume):
            log("Loading model and optimizer from checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)

            net.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            log(
                "Loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
            epoch_start = checkpoint["epoch"]
            if epoch_start >= args.ep_gamma_start:
                gamma_w = args.gamma_w
        else:
            log("No checkpoint found at '{}'".format(args.resume))

    lr_scheduler = get_lr_scheduler(optimizer, args, epoch_start)

    for epoch in range(epoch_start, args.n_epochs + args.n_epochs_decay + 1):
        log(f"---- EPOCH {epoch} ----")
        if epoch >= args.ep_gamma_start:
            gamma_w = args.gamma_w
            log(f"at epoch {epoch}, gamma_w changed to {gamma_w}")

        train_mse = 0.0
        best_val_mse = 99999.0
        losscount = 0

        net.train()

        for batch in train_loader:
            img_uv, img_uv_dw, bm = batch

            img_uv_c = img_uv.to(device, non_blocking=True)
            img_uv_dw_c = img_uv_dw.to(device, non_blocking=True)
            bm_c = bm.to(device, non_blocking=True)

            # net input size is (b, 3, 288, 288)
            # net output size is (b, 2, 288, 288)
            pred_bm = net(img_uv_c)
            if pred_bm.max() > 2:
                # log(f"warning here: {pred_bm.max()}")
                pred_bm = ((pred_bm / 288.0) - 0.5) * 2
            pred_img_dw = tensor_unwarping(img_uv_c, pred_bm)

            optimizer.zero_grad(set_to_none=True)

            recon_loss = l1_loss(pred_img_dw, img_uv_dw_c)
            bm_loss = l1_loss(pred_bm, bm_c)
            net_loss = args.alpha_w * bm_loss + gamma_w * recon_loss
            net_loss.backward()
            optimizer.step()

            tmp_mse = mse_loss(pred_img_dw, img_uv_dw_c)
            wandb.log({"train_loss": net_loss})
            if losscount % 50 == 0:
                img_sample = img_uv_c.detach().cpu().numpy()[0].transpose(1, 2, 0)
                img_dw_sample = img_uv_dw_c.detach().cpu().numpy()[0].transpose(1, 2, 0)
                img_pred_sample = (
                    pred_img_dw.detach().cpu().numpy()[0].transpose(1, 2, 0)
                )
                image = wandb.Image(img_sample, caption=f"epoch{epoch}_before")
                image_dw = wandb.Image(img_dw_sample, caption=f"after_gt")
                image_pred = wandb.Image(img_pred_sample, caption=f"after_pred")
                wandb.log({"examples": [image, image_dw, image_pred]})

            train_mse += float(tmp_mse)
            losscount += 1

            gc.collect()

        train_mse = train_mse / max(1, losscount)
        curr_lr = update_learning_rate(lr_scheduler, optimizer)
        log(f"TRAIN at EPOCH {epoch} ENS: train_mse = {train_mse}, curr_lr = {curr_lr}")
        wandb.log({"train_mse": train_mse})

        net.eval()
        with torch.no_grad():
            mse_loss_val = 0.0
            for batch in val_loader:
                img, img_dw, bm = batch
                img = img.to(device)
                img_dw = img_dw.to(device)

                pred_bm = net(img)
                if pred_bm.max() > 2:
                    pred_bm = ((pred_bm / 288.0) - 0.5) * 2
                pred_img_dw = tensor_unwarping(img, pred_bm)

                loss_img_val = mse_loss(img_dw, pred_img_dw)
                mse_loss_val += float(loss_img_val)

            val_mse = mse_loss_val / len(val_loader)
            log(f"EVAL at EPOCH {epoch} ENS: val_mse = {val_mse}, curr_lr = {curr_lr}")
        wandb.log({"val_mse": val_mse})

        # save best model
        if val_mse < best_val_mse or epoch == args.n_epochs + args.n_epochs_decay:
            best_val_mse = val_mse
            # save
            state = {
                "epoch": epoch + 1,
                "model_state": net.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }
            model_path = (
                "models/"
                + f"{current_time}_ep_{epoch + 1}_{val_mse:.5f}_{train_mse:.5f}_best_model.pkl"
            )
            torch.save(state, model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparams")

    parser.add_argument(
        "--data_path_doc3D",
        nargs="?",
        type=str,
        default="./data/doc3D/",
        help="Data path to load Doc3D data.",
    )
    parser.add_argument(
        "--data_path_UVDoc",
        nargs="?",
        type=str,
        default="./data/UVdoc/",
        help="Data path to load UVDoc data.",
    )
    parser.add_argument(
        "--data_to_use",
        type=str,
        default="both",
        choices=["both", "doc3d"],
        help="Dataset to use for training, either 'both' for Doc3D and UVDoc, or 'doc3d' for Doc3D only.",
    )
    parser.add_argument(
        "--batch_size", nargs="?", type=int, default=64, help="Batch size."
    )
    parser.add_argument(
        "--n_epochs",
        nargs="?",
        type=int,
        default=40,
        help="Number of epochs with initial (constant) learning rate.",
    )
    parser.add_argument(
        "--n_epochs_decay",
        nargs="?",
        type=int,
        default=25,
        help="Number of epochs to linearly decay learning rate to zero.",
    )
    parser.add_argument(
        "--lr", nargs="?", type=float, default=1e-4, help="Initial learning rate."
    )
    parser.add_argument(
        "--alpha_w",
        nargs="?",
        type=float,
        default=5.0,
        help="Weight for the 2D grid L1 loss.",
    )
    parser.add_argument(
        "--beta_w",
        nargs="?",
        type=float,
        default=5.0,
        help="Weight for the 3D grid L1 loss.",
    )
    parser.add_argument(
        "--gamma_w",
        nargs="?",
        type=float,
        default=1.0,
        help="Weight for the image reconstruction loss.",
    )
    parser.add_argument(
        "--ep_gamma_start",
        nargs="?",
        type=int,
        default=30,
        help="Epoch from which to start using image reconstruction loss.",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        type=str,
        default="models/init_model.pkl",
        help="Path to previous saved model to restart from.",
    )
    parser.add_argument(
        "--logdir",
        nargs="?",
        type=str,
        default="./log/default",
        help="Path to store the logs.",
    )
    parser.add_argument(
        "-a",
        "--appearance_augmentation",
        nargs="*",
        type=str,
        default=["shadow", "visual", "noise", "color"],
        choices=["shadow", "blur", "visual", "noise", "color"],
        help="Appearance augmentations to use.",
    )
    parser.add_argument(
        "-gUVDoc",
        "--geometric_augmentationsUVDoc",
        nargs="*",
        type=str,
        default=["rotate", "flip", "perspective"],
        choices=["rotate", "flip", "perspective"],
        help="Geometric augmentations to use for the UVDoc dataset.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of workers to use for the dataloaders.",
    )
    parser.add_argument(
        "--project",
        type=int,
        default="test_doctr++",
        help="Number of workers to use for the dataloaders.",
    )

    args = parser.parse_args()

    wandb.init(
        project=args.project,
        config={
            "learning_rate": args.lr,
            "epochs": args.n_epochs + args.n_epochs_decay,
        },
    )
    main_worker(args)

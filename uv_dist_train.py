import gc
import os
from datetime import datetime

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import wandb
from config import TrainConfig
from data_process.mixed_dataset import MixedDataset, MixedSeperateDataset
from data_process.qb_dataset import QbDataset
from data_process.utils import UV_IMG_SIZE, get_unwarp, tensor_unwarping
from data_process.uvdoc_dataset import UVDocDataset
from network.netloss import LocalLoss, WarperUtil
from torch.cuda.amp import GradScaler as GradScaler

from uvnet.model import UVDocnet
os.environ["WANDB_API_KEY"] = "7974567f16cc72e73931e4bfb12c157156ab7109"
gamma_w = 0.0

import logging

logger: logging.Logger = None
warper_util: WarperUtil = None


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
    dataset_cls_dict = {
        "mixed": MixedDataset,
        "mixed_sep": MixedSeperateDataset,
        "uvdoc": UVDocDataset,
        "qbdoc": QbDataset,
    }
    data_cls = dataset_cls_dict[args.data_to_use]
    t_UVDoc_data = data_cls(
        appearance_augmentation=args.appearance_augmentation,
        geometric_augmentations=args.geometric_augmentationsUVDoc,
        split="train",
    )
    v_UVDoc_data = data_cls(split="val")

    train_sampler = DistributedSampler(t_UVDoc_data)
    val_sampler = DistributedSampler(v_UVDoc_data)

    train_loader = DataLoader(
        dataset=t_UVDoc_data,
        sampler=train_sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        dataset=v_UVDoc_data,
        sampler=val_sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, train_sampler, val_sampler


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


def setup_dist(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"

    dist.init_process_group(backend="nccl")


def cleanup():
    dist.destroy_process_group()


def train_epoch(
    args,
    epoch,
    train_loader,
    net,
    optimizer,
    lr_scheduler,
    device,
    l1_loss,
    mse_loss,
    local_loss: LocalLoss,
    scaler,
):
    net.train()
    alpha_w = args.alpha_w
    # ppedge_enabled = epoch > args.ep_beta_start
    ppedge_enabled = False
    recon_enabled = epoch > args.ep_gamma_start
    beta_w = args.beta_w if epoch > args.ep_beta_start else 0
    gamma_w = args.gamma_w if epoch > args.ep_gamma_start else 0
    losscount = 0
    train_mse = 0.0
    for batch in train_loader:
        if args.data_to_use == "mixed_sep":
            img_uv, img_uv_dw, bm = batch[0]
            img_qb, img_qb_dw, _ = batch[1]
        else:
            img_uv, img_uv_dw, bm = batch

        # train with 3d data
        img_uv_c = img_uv.to(device, non_blocking=True)
        img_uv_dw_c = img_uv_dw.to(device, non_blocking=True)
        bm_c = bm.to(device, non_blocking=True)

        pred_bm = net(img_uv_c)
        pred_img_dw = tensor_unwarping(img_uv_c, pred_bm, (712, 488))
        optimizer.zero_grad(set_to_none=True)

        mask = torch.logical_and(pred_img_dw, img_uv_dw_c)
        recon_loss = l1_loss(mask * pred_img_dw, mask * img_uv_dw_c) if recon_enabled else 0
        bm_loss = l1_loss(pred_bm, bm_c)
        net_loss = alpha_w * bm_loss + gamma_w * recon_loss
        net_loss.backward()
        optimizer.step()

        tmp_mse = mse_loss(pred_img_dw, img_uv_dw_c)
        train_mse += float(tmp_mse)
        losscount += 1
        wandb.log(
            {
                "train_loss": net_loss,
                "recon_loss": recon_loss,
                "bm_loss": bm_loss,
            }
        )
        if losscount % 50 == 0:
            img_sample = img_uv_c.detach().cpu().numpy()[0].transpose(1, 2, 0)
            img_dw_sample = img_uv_dw_c.detach().cpu().numpy()[0].transpose(1, 2, 0)
            img_pred_sample = pred_img_dw.detach().cpu().numpy()[0].transpose(1, 2, 0)
            image = wandb.Image(img_sample, caption=f"ep{epoch}_before")
            image_dw = wandb.Image(img_dw_sample, caption=f"after_gt")
            image_pred = wandb.Image(img_pred_sample, caption=f"after_pred")
            wandb.log({"examples": [image, image_dw, image_pred]})

        gc.collect()

    train_mse = train_mse / max(1, losscount)
    curr_lr = update_learning_rate(lr_scheduler, optimizer)
    return train_mse, curr_lr
    # log(f"TRAIN at EPOCH {epoch} ENS: train_mse = {train_mse}, curr_lr = {curr_lr}")
    # wandb.log({"train_mse": train_mse})


def eval_epoch(epoch, val_loader, net, device, mse_loss):
    net.eval()
    with torch.no_grad():
        mse_loss_val = 0.0
        for batch in val_loader:
            img, img_dw, bm = batch[0]
            img = img.to(device)
            img_dw = img_dw.to(device)

            pred_bm = net(img)
            # if pred_bm.max() > 2:
            #     pred_bm = ((pred_bm / 288.0) - 0.5) * 2
            pred_img_dw = tensor_unwarping(img, pred_bm)

            loss_img_val = mse_loss(img_dw, pred_img_dw)
            mse_loss_val += float(loss_img_val)

        val_mse = mse_loss_val / len(val_loader)

    return val_mse
    #     log(f"EVAL at EPOCH {epoch} ENS: val_mse = {val_mse}, curr_lr = {curr_lr}")
    # wandb.log({"val_mse": val_mse})


def load_model(path, net, optimizer, device):
    log("Loading model and optimizer from checkpoint '{}'".format(path))
    checkpoint = torch.load(path, map_location=device)
    # print(checkpoint["model_state"])
    model_state = checkpoint["model_state"]
    if not list(model_state.keys())[0].startswith("module."):
        model_state = {"module." + k: v for k, v in model_state.items() if not "out_point_positions3D" in k}
    net.load_state_dict(model_state)
    # optimizer.load_state_dict(checkpoint["optimizer_state"])
    log("Loaded checkpoint '{}' (epoch {})".format(path, checkpoint["epoch"]))
    epoch_start = checkpoint["epoch"]
    return net, optimizer, epoch_start


def basic_worker(args):
    current_time = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    experiment_name = "DIST-" + current_time
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    log_file_name = os.path.join(args.logdir, experiment_name + ".log")
    global logger
    logger = get_logger(log_file_name)

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    logger.info(f"Start running dist_worker on rank {rank}")
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(rank)

    global warper_util
    warper_util = WarperUtil(64).to(device)

    train_loader, val_loader, train_sampler, val_sampler = setup_data(args)
    # net = GeoTr().to(device)
    net = UVDocnet(num_filter=32, kernel_size=5).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999))
    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = torch.nn.parallel.DistributedDataParallel(
        net, device_ids=[rank]
    )
    epoch_start = 0
    global gamma_w
    if args.resume:
        if os.path.isfile(args.resume):
            net, optimizer, epoch_start = load_model(
                args.resume, net, optimizer, device
            )
            epoch_start = 0
            if epoch_start >= args.ep_gamma_start:
                gamma_w = args.gamma_w
        else:
            log("No checkpoint found at '{}'".format(args.resume))

    l1_loss = torch.nn.L1Loss().to(device)
    mse_loss = torch.nn.MSELoss().to(device)
    self_loss = LocalLoss().to(device)

    lr_scheduler = get_lr_scheduler(optimizer, args, epoch_start)
    scaler = GradScaler()
    dist.barrier()
    best_val_mse = 9999.0
    for epoch in range(epoch_start, args.n_epochs + args.n_epochs_decay + 1):

        log(f"---- EPOCH {epoch} for rank {rank} ----")
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)

        train_mse, curr_lr = train_epoch(
            args,
            epoch,
            train_loader,
            net,
            optimizer,
            lr_scheduler,
            device,
            l1_loss,
            mse_loss,
            self_loss,
            scaler,
        )
        log(f"TRAIN at EPOCH {epoch} ENS: train_mse = {train_mse}, curr_lr = {curr_lr}")

        val_mse = eval_epoch(epoch, val_loader, net, device, mse_loss)
        log(f"EVAL at EPOCH {epoch} ENS: val_mse = {val_mse}, curr_lr = {curr_lr}")

        wandb.log({"train_mse": train_mse})
        wandb.log({"val_mse": val_mse})

        if rank == 0:
            # save
            state = {
                "epoch": epoch + 1,
                "model_state": net.state_dict(),
            }
            dir = "models/" + current_time + "/"
            if not os.path.exists(dir):
                os.makedirs(dir)
            name = f"dist_ep_{epoch + 1}_{val_mse:.5f}_{train_mse:.5f}.pkl"
            if val_mse < best_val_mse:
                best_val_mse = val_mse
                name = "best_model_" + name
            model_path = dir + name
            torch.save(state, model_path)
        dist.barrier()

    cleanup()


if __name__ == "__main__":
    args = TrainConfig()
    print(args.config_dict)

    wandb.init(
        project=args.project,
        group=args.group,
        config=args.config_dict,
    )
    basic_worker(args)

import os
import shutil
import argparse
from tqdm import tqdm
import yaml
from easydict import EasyDict
from glob import glob
import torch
import torch.utils.tensorboard
from torch.nn.utils import clip_grad_norm_
#from torch_geometric.data import DataLoader
from torch_geometric.loader import DataLoader
import pickle
from pathlib import Path

from models.epsnet import get_model
from utils.datasets import ConformationDataset, TSDataset
from utils.transforms import CountNodesPerGraph
from utils.misc import (
    seed_all,
    get_new_log_dir,
    get_logger,
    get_checkpoint_path,
    inf_iterator,
)
from utils.common import get_optimizer, get_scheduler

import wandb

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--split_file", type=str, default="random_split.pkl")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resume_iter", type=int, default=None)
    parser.add_argument("--logdir", type=str, default="./logs")
    parser.add_argument("--pretrain", type=str, default="")
    parser.add_argument("--project", type=str, default="")
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--fn", type=str, default=None)
    args = parser.parse_args()
    torch.randn(1).to(args.device)

    resume = os.path.isdir(args.config)
    if resume:
        config_path = glob(os.path.join(args.config, "*.yml"))[0]
        resume_from = args.config
    else:
        config_path = args.config
    with open(config_path, "r") as f:
        config = EasyDict(yaml.safe_load(f))
    config_name = os.path.basename(config_path)[
        : os.path.basename(config_path).rfind(".")
    ]
    seed_all(config.train.seed)

    if args.tag is None:
        args.tag = args.name

    # Logging
    if resume:
        log_dir = get_new_log_dir(
            args.logdir, prefix=config_name, tag=f"{args.tag}_resume", fn=args.fn
        )
        os.symlink(
            os.path.realpath(resume_from),
            os.path.join(log_dir, os.path.basename(resume_from.rstrip("/"))),
        )
    else:
        log_dir = get_new_log_dir(
            args.logdir, prefix=config_name, tag=f"{args.tag}", fn=args.fn
        )
        shutil.copytree("./models", os.path.join(log_dir, "models"))

    ckpt_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = get_logger("train", log_dir)
    logger.info(args)
    logger.info(config)

    use_wandb = False
    if args.name and args.project:
        use_wandb = True
        wandb.init(project=args.project, name=args.name)
        wandb.config = config

    shutil.copyfile(config_path, os.path.join(log_dir, os.path.basename(config_path)))

    # Datasets and loaders
    logger.info("Loading datasets...")
    transforms = CountNodesPerGraph()
    if hasattr(config.model, "TS") and config.model.TS:
        train_set = TSDataset(config.dataset.train, transform=transforms)
        val_set = TSDataset(config.dataset.val, transform=transforms)
    else:
        with open(args.split_file, "rb") as f:
            split_dict = pickle.load(f)
        train_indices = split_dict["train"]
        val_indices = split_dict["val"]

        train_set = ConformationDataset(config.dataset.data, train_indices, transform=transforms)
        val_set = ConformationDataset(config.dataset.data, val_indices, transform=transforms)
    train_iterator = inf_iterator(
        DataLoader(train_set, config.train.batch_size, shuffle=True)
    )
    val_loader = DataLoader(val_set, config.train.batch_size, shuffle=False)

    # Model
    logger.info("Building model...")
    model = get_model(config.model).to(args.device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total number of learnable parameters: {total_params}")
    # Optimizer
    optimizer = get_optimizer(config.train.optimizer, model)
    scheduler = get_scheduler(config.train.scheduler, optimizer)
    start_iter = 1

    # Resume from checkpoint
    if resume:
        ckpt_path, start_iter = get_checkpoint_path(
            os.path.join(resume_from, "checkpoints"), it=args.resume_iter
        )
        logger.info("Resuming from: %s" % ckpt_path)
        logger.info("Iteration: %d" % start_iter)
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])

    if args.pretrain:
        logger.info(f"pretraining model checkpoint load : {args.pretrain}")
        ckpt = torch.load(args.pretrain, map_location=args.device)
        model.load_state_dict(ckpt["model"], strict=False)

    def train(it):
        model.train()
        optimizer.zero_grad()
        batch = next(train_iterator).to(args.device)
        loss_list_r_ts_p_3 = model.get_loss(
            atom_type=batch.atom_type,
            r_feat=batch.r_feat,
            p_feat=batch.p_feat,
            pos_N_3_3=batch.pos,
            bond_index=batch.edge_index,
            bond_type=batch.edge_type,
            batch=batch.batch,
            num_nodes_per_graph=batch.num_nodes_per_graph,
            num_graphs=batch.num_graphs,
            anneal_power=config.train.anneal_power,
        )
        loss = sum(loss_list_r_ts_p_3)
        n = loss.size(0)
        loss_sum = loss.sum()
        loss = loss.mean()
        loss.backward()
        orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
        optimizer.step()

        return (
            loss_sum,
            [l.sum() for l in loss_list_r_ts_p_3],
            optimizer.param_groups[0]["lr"],
            orig_grad_norm,
            n
        )

    def validate(it):
        sum_loss, sum_n = 0, 0
        # sum_loss_r, sum_loss_ts, sum_loss_p = 0, 0, 0
        sum_loss_ts = 0
        
        with torch.no_grad():
            model.eval()
            for batch in val_loader:
                batch = batch.to(args.device)
                loss_list_r_ts_p_3 = model.get_loss(
                    atom_type=batch.atom_type,
                    r_feat=batch.r_feat,
                    p_feat=batch.p_feat,
                    pos_N_3_3=batch.pos,
                    bond_index=batch.edge_index,
                    bond_type=batch.edge_type,
                    batch=batch.batch,
                    num_nodes_per_graph=batch.num_nodes_per_graph,
                    num_graphs=batch.num_graphs,
                    anneal_power=config.train.anneal_power,
                )
                loss = sum(loss_list_r_ts_p_3)
                sum_loss += loss.sum().item()
                #sum_loss_r += loss_list_r_ts_p_3[0].sum().item()
                sum_loss_ts += loss_list_r_ts_p_3[0].sum().item()
                #sum_loss_p += loss_list_r_ts_p_3[2].sum().item()
                sum_n += loss.size(0)
        
        avg_loss = sum_loss / sum_n
        #avg_loss_r = sum_loss_r / sum_n
        avg_loss_ts = sum_loss_ts / sum_n
        #avg_loss_p = sum_loss_p / sum_n

        if config.train.scheduler.type == "plateau":
            scheduler.step(avg_loss)
        else:
            scheduler.step()

        logger.info("[Validate] Iter %05d | Loss %.6f " % (it, avg_loss))
        if use_wandb:
            wandb.log({
                "val/loss": avg_loss,
                #"val/loss_reactants": avg_loss_r,
                "val/loss_TS": avg_loss_ts,
                #"val/loss_products": avg_loss_p,
            })
        return avg_loss

    try:
        loss_sum, loss_r_sum, loss_ts_sum, loss_p_sum = 0, 0, 0, 0
        n_sum = 0
        grad_norm_sum = 0
        best_loss = 10000
        for it in tqdm(range(start_iter, config.train.max_iters + 1)):
            loss, loss_list, lr, grad_norm, n = train(it)
            loss_sum += loss
            #loss_r_sum += loss_list[0]
            loss_ts_sum += loss_list[0]
            #loss_p_sum += loss_list[2]
            n_sum += n
            grad_norm_sum += grad_norm
            if (it-1) % config.train.val_freq == 0 or it == config.train.max_iters:
                if use_wandb:
                    wandb.log(
                        {
                            "train/loss": loss_sum / n_sum,
                            #"train/loss_reactants": loss_r_sum / n_sum,
                            "train/loss_TS": loss_ts_sum / n_sum,
                            #"train/loss_products": loss_p_sum / n_sum,
                            "train/lr": lr,
                            "train/grad_norm": grad_norm_sum / config.train.val_freq,
                        }
                    )
                logger.info(
                    "[Train] Iter %05d | Loss %.2f | Grad %.2f | LR %.6f"
                    % (
                        it,
                        loss_sum / n_sum,
                        grad_norm_sum / config.train.val_freq,
                        lr,
                    )
                )
                loss_sum, loss_r_sum, loss_ts_sum, loss_p_sum = 0, 0, 0, 0
                n_sum = 0
                grad_norm_sum = 0
                avg_val_loss = validate(it)
                if avg_val_loss < best_loss:
                    best_loss = avg_val_loss
                    ckpt_path = os.path.join(ckpt_dir, "%d.pt" % it)
                    torch.save(
                        {
                            "config": config,
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "iteration": it,
                            "avg_val_loss": avg_val_loss,
                        },
                        ckpt_path,
                    )

    except KeyboardInterrupt:
        logger.info("Terminating...")

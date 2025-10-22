import json
import os
import time

import torch
import torch.nn.functional as F
from loguru import logger
from timm.utils import AverageMeter

from _data_cm import build_loaders, get_topk, get_class_num
from _utils import (
    build_optimizer,
    calc_learnable_params,
    calc_map_eval,
    EarlyStopping,
    init,
    print_in_md,
    save_checkpoint,
    seed_everything,
    validate_smart,
    rename_output,
)
from _utils_cm import validate
from config import get_config
from loss import ListwiseLoss
from network import build_model


def train_epoch(args, dataloader, net, criterion, optimizer, epoch):
    tic = time.time()

    stat_meters = {}
    for x in ["t_loss", "s_loss", "loss", "img_mAP", "txt_mAP"]:
        stat_meters[x] = AverageMeter()

    net.train()
    for images, texts, labels, _ in dataloader:
        images, texts, labels = images.to(args.device), texts.to(args.device), labels.to(args.device)

        img_embs, txt_embs = net(images, texts)

        img_embs = F.normalize(img_embs)
        txt_embs = F.normalize(txt_embs)

        all_mats = {
            "img_img": img_embs @ img_embs.T,  # image-image similarity
            "txt_txt": txt_embs @ txt_embs.T,  # text-text similarity
            "img_txt": img_embs @ txt_embs.T,  # image-text similarity
            "txt_img": txt_embs @ img_embs.T,  # text-image similarity
        }

        loss1, loss2 = 0, 0
        for sims in all_mats.values():
            loss1 += criterion.fwd_t_loss(sims, labels)[0]
            loss2 += criterion.fwd_s_ndcg(sims, labels)

        stat_meters["t_loss"].update(loss1)
        stat_meters["s_loss"].update(loss2)

        loss = args.lambda1 * loss1 + args.lambda2 * loss2
        stat_meters["loss"].update(loss)

        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(net.parameters(), 2)
        optimizer.step()

        # to check overfitting
        map1 = calc_map_eval(img_embs.sign(), labels)
        stat_meters["img_mAP"].update(map1)

        map2 = calc_map_eval(txt_embs.sign(), labels)
        stat_meters["txt_mAP"].update(map2)

        torch.cuda.empty_cache()

    toc = time.time()
    sm_str = ""
    for x in stat_meters.keys():
        sm_str += f"[{x}:{stat_meters[x].avg:.1f}]" if "n_" in x else f"[{x}:{stat_meters[x].avg:.4f}]"
    logger.info(
        f"[Training][dataset:{args.dataset}][bits:{args.n_bits}][epoch:{epoch}/{args.n_epochs - 1}][time:{(toc - tic):.3f}]{sm_str}"
    )


def train_init(args):
    # setup net
    net = build_model(args)

    # setup criterion
    criterion = ListwiseLoss()

    logger.info(f"number of learnable params: {calc_learnable_params(net)}")

    # setup optimizer
    optimizer = build_optimizer(args.optimizer, net.parameters(), lr=args.lr, weight_decay=args.wd)

    return net, criterion, optimizer


def train(args, train_loader, query_loader, dbase_loader):
    net, criterion, optimizer = train_init(args)

    early_stopping = EarlyStopping()

    for epoch in range(args.n_epochs):
        train_epoch(args, train_loader, net, criterion, optimizer, epoch)

        # we monitor mAP@topk validation accuracy every 5 epochs
        if (epoch + 1) % 5 == 0 or (epoch + 1) == args.n_epochs:
            early_stop = validate_smart(
                args,
                query_loader,
                dbase_loader,
                early_stopping,
                epoch,
                model=net,
                multi_thread=args.multi_thread,
                validate_fnc=validate,
            )
            if early_stop:
                break

    if early_stopping.counter == early_stopping.patience:
        logger.info(
            f"without improvement, will save & exit, best mAP: {early_stopping.best_map:.3f}, best epoch: {early_stopping.best_epoch}"
        )
    else:
        logger.info(
            f"reach epoch limit, will save & exit, best mAP: {early_stopping.best_map:.3f}, best epoch: {early_stopping.best_epoch}"
        )

    save_checkpoint(args, early_stopping.best_checkpoint)

    return early_stopping.best_epoch, early_stopping.best_map


def main():
    init()
    args = get_config()

    if "rename" in args and args.rename:
        rename_output(args)

    dummy_logger_id = None
    rst = []
    # for dataset in ["nuswide", "flickr", "coco"]:
    for dataset in ["nuswide"]:
        print(f"processing dataset: {dataset}")
        args.dataset = dataset
        args.n_classes = get_class_num(dataset)
        args.topk = get_topk(dataset)

        train_loader, query_loader, dbase_loader = build_loaders(
            dataset, args.data_dir, batch_size=args.batch_size, num_workers=args.n_workers
        )

        # for hash_bit in [16, 32, 64, 128]:
        for hash_bit in [32, 128]:
            print(f"processing hash-bit: {hash_bit}")
            seed_everything(args.seed)
            args.n_bits = hash_bit

            args.save_dir = f"./output/{args.backbone}/{dataset}/{hash_bit}"
            os.makedirs(args.save_dir, exist_ok=True)
            if any(x.endswith(".pth") for x in os.listdir(args.save_dir)):
                print(f"*.pth exists in {args.save_dir}, will pass")
                continue

            if dummy_logger_id is not None:
                logger.remove(dummy_logger_id)
            dummy_logger_id = logger.add(f"{args.save_dir}/train.log", mode="w", level="INFO")

            with open(f"{args.save_dir}/config.json", "w") as f:
                json.dump(
                    vars(args),
                    f,
                    indent=4,
                    sort_keys=True,
                    default=lambda o: o if type(o) in [bool, int, float, str] else str(type(o)),
                )

            best_epoch, best_map = train(args, train_loader, query_loader, dbase_loader)
            rst.append({"dataset": dataset, "hash_bit": hash_bit, "best_epoch": best_epoch, "best_map": best_map})

    print_in_md(rst)


if __name__ == "__main__":
    main()

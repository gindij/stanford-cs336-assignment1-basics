import argparse
import json
import os
import random
import pprint

import numpy as np
import torch
import tqdm
import wandb

from cs336_basics.data import get_batch
from cs336_basics.optimizer.adamw import AdamW
from cs336_basics.optimizer.learning_rate_scheduler import get_cosine_annealing_lr
from cs336_basics.transformer.model import TransformerLM
from cs336_basics.transformer.functional import cross_entropy
from cs336_basics.utils import save_checkpoint, load_checkpoint, clip_gradients


def main():

    random.seed(1)

    parser = argparse.ArgumentParser(description="Train a model")

    # transformer model arguments
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=2048)
    parser.add_argument("--attn-pdrop", type=float, default=0.1)
    parser.add_argument("--residual-pdrop", type=float, default=0.1)
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--ep-norm", type=float, default=1e-5)

    parser.add_argument("--clip-max-l2", type=float, default=1.0)

    # training parameters
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-train-batches-per-epoch", type=int, default=5)
    parser.add_argument("--num-valid-batches-per-epoch", type=int, default=2)

    # learning rate adjustment parameters
    parser.add_argument("--lr-min", type=float, default=1e-6)
    parser.add_argument("--lr-max", type=float, default=1e-3)
    parser.add_argument("--lr-fixed", type=float, default=1e-4)
    parser.add_argument("--use-cos-annealing", action="store_true")

    # other parameters
    parser.add_argument("--random-state", type=int, default=1)
    parser.add_argument("--token-path-train", type=str)
    parser.add_argument("--token-path-valid", type=str)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--epochs-before-persist", type=int, default=5)

    args = parser.parse_args()

    lr_warmup_batches = int(0.1 * args.num_train_batches_per_epoch)
    lr_cosine_cycle_batches = int(0.9 * args.num_train_batches_per_epoch)

    params = {
        **vars(args),
        "lr_warmup_batches": lr_warmup_batches,
        "lr_cosine_cycle_batches": lr_cosine_cycle_batches,
    }

    pprint.pprint(params)

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    elif torch.backends.mps.is_available():
        device = "mps"

    model = TransformerLM(
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        attn_pdrop=args.attn_pdrop,
        residual_pdrop=args.residual_pdrop,
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        ep_norm=args.ep_norm,
    ).to(device)

    # dummy learning rate if using cosine annealing
    optimizer = AdamW(
        model.parameters(),
        lr=0.0 if args.use_cos_annealing else args.lr_fixed,
        device=device,
    )

    train_token_array = np.load(args.token_path_train, mmap_mode="r")
    train_batches = [
        get_batch(
            train_token_array,
            args.batch_size,
            args.context_length,
            device,
        )
        for _ in range(args.num_train_batches_per_epoch)
    ]
    valid_token_array = np.load(args.token_path_valid, mmap_mode="r")
    valid_batches = [
        get_batch(
            valid_token_array,
            args.batch_size,
            args.context_length,
            device,
        )
        for _ in range(args.num_valid_batches_per_epoch)
    ]

    train_losses = []
    valid_losses = []

    total_batches_processed = 0
    for epoch in range(args.epochs):
        with tqdm.tqdm(
            range(args.num_train_batches_per_epoch),
            desc=f"Epoch {epoch + 1}/{args.epochs}",
        ) as pbar:
            total_epoch_loss = 0.0
            for ix_batch in pbar:

                total_batches_processed += 1
                if args.use_cos_annealing:
                    new_lr = get_cosine_annealing_lr(
                        t=total_batches_processed,
                        lr_min=args.lr_min,
                        lr_max=args.lr_max,
                        warmup_iters=lr_warmup_batches,
                        cosine_cycle_iters=lr_cosine_cycle_batches,
                    )
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = new_lr

                data, labels = train_batches[ix_batch]
                logits = model(data)

                optimizer.zero_grad()
                loss = cross_entropy(
                    logits.view(-1, args.vocab_size),
                    labels.view(-1),
                )
                loss.backward()
                clip_gradients(model.parameters(), args.clip_max_l2)
                optimizer.step()

                total_epoch_loss += loss.item()
                pbar.set_postfix(loss=total_epoch_loss / (ix_batch + 1))

        train_losses.append(total_epoch_loss / args.num_train_batches_per_epoch)

        with tqdm.tqdm(
            range(args.num_valid_batches_per_epoch), desc="Validation step:"
        ) as pbar:
            with torch.no_grad():
                total_val_loss = 0.0
                for ix_batch in pbar:
                    data, labels = valid_batches[ix_batch]
                    logits = model(data)
                    loss = cross_entropy(
                        logits.view(-1, args.vocab_size), labels.view(-1)
                    )
                    total_val_loss += loss.item()
                    pbar.set_postfix(val_loss=total_epoch_loss / (ix_batch + 1))

        valid_losses.append(total_val_loss / args.num_valid_batches_per_epoch)

        if (epoch + 1) % args.epochs_before_persist == 0 or epoch == args.epochs - 1:
            batches_processed = (epoch + 1) * args.num_train_batches_per_epoch
            save_checkpoint(
                model,
                optimizer,
                iteration=batches_processed,
                out=os.path.join(
                    args.checkpoint_dir, f"checkpoint{batches_processed}.pt"
                ),
            )

    metrics_dict = {
        "params": params,
        "metrics": {"train_loss": train_losses, "valid_loss": valid_losses},
    }
    with open(
        os.path.join(args.checkpoint_dir, "metrics.json"), "w", encoding="utf-8"
    ) as metrics_file:
        json.dump(metrics_dict, metrics_file)


if __name__ == "__main__":
    main()

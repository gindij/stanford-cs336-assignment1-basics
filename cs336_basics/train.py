import numpy as np
import random
import torch
import wandb
import argparse
import os
import tqdm

from cs336_basics.transformer.model import TransformerLM
from cs336_basics.data import get_batch
from cs336_basics.utils import save_checkpoint, load_checkpoint
from cs336_basics.optimizer.adamw import AdamW
from cs336_basics.transformer.functional import cross_entropy
from cs336_basics.optimizer.learning_rate_scheduler import get_cosine_annealing_lr


parser = argparse.ArgumentParser()
parser.add_argument("--lr-min", type=float, default=1e-5)
parser.add_argument("--lr-max", type=float, default=1e-3)
parser.add_argument("--lr-warmup-iters", type=int, default=50000)
parser.add_argument("--lr-cosine-cycle-iters", type=int, default=800000)

parser.add_argument("--vocab-size", type=int, default=50257)
parser.add_argument("--context-length", type=int, default=256)
parser.add_argument("--num-layers", type=int, default=3)
parser.add_argument("--num-heads", type=int, default=4)
parser.add_argument("--d-model", type=int, default=384)
parser.add_argument("--d-ff", type=int, default=1536)
parser.add_argument("--attn-pdrop", type=float, default=0.3)
parser.add_argument("--residual-pdrop", type=float, default=0.3)

parser.add_argument("--epochs", type=int, default=250)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--num-train-batches", type=int, default=400)
parser.add_argument("--eval-every", type=int, default=10)
parser.add_argument("--num-eval-batches", type=int, default=100)

parser.add_argument("--save-every", type=int, default=10)
parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
parser.add_argument("--start-from", type=str, default="new")

parser.add_argument("--dataset", type=str, default="ts")
args = parser.parse_args()

# Start a new wandb run to track this script.
run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="gindi",
    # Set the wandb project where this run will be logged.
    project="cs336-hw1",
    # Track hyperparameters and run metadata.
    config=vars(args),
)

device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"
elif torch.backends.mps.is_available():
    device = "mps"

print(f"Using device: {device}")

model = TransformerLM(
    d_model=args.d_model,
    num_heads=args.num_heads,
    d_ff=args.d_ff,
    attn_pdrop=args.attn_pdrop,
    residual_pdrop=args.residual_pdrop,
    vocab_size=args.vocab_size,
    context_length=args.context_length,
    num_layers=args.num_layers,
).to(device)

lr_init = get_cosine_annealing_lr(
    0,
    args.lr_min,
    args.lr_max,
    args.lr_warmup_iters,
    args.lr_cosine_cycle_iters,
)
optimizer = AdamW(model.parameters(), lr_init, device=device)
iteration = 0
if args.start_from == "latest" and os.path.exists(args.checkpoint_dir):
    ckpt = max(os.listdir(args.checkpoint_dir))
    iteration = load_checkpoint(ckpt, model, optimizer)

if not os.path.exists(args.checkpoint_dir):
    os.makedirs(args.checkpoint_dir)

assert args.dataset in {"ts", "owt"}
train_data = np.load(f"tokens/{args.dataset}_train.npy", mmap_mode="r")
valid_data = np.load(f"tokens/{args.dataset}_valid.npy", mmap_mode="r")

epoch_pbar = tqdm.tqdm(range(iteration, args.epochs), desc="Epochs")
for epoch in epoch_pbar:

    model.train()

    if epoch % args.save_every == 0:
        save_checkpoint(
            model,
            optimizer,
            epoch,
            os.path.join(args.checkpoint_dir, f"{str(epoch).zfill(5)}.pt"),
        )

    if epoch % args.eval_every == 0:
        model.eval()
        with torch.no_grad():
            total_eval_loss = 0.0
            for i in tqdm.tqdm(range(args.num_eval_batches), desc="Eval batches", leave=False):
                x, y = get_batch(
                    valid_data,
                    batch_size=args.batch_size,
                    context_length=args.context_length,
                    device=device,
                )
                yhat = model(x).view(-1, args.vocab_size)
                loss = cross_entropy(yhat, y.flatten())
                total_eval_loss += loss.item()

            run.log(
                {"mean_epoch_eval_loss": total_eval_loss / args.num_eval_batches},
                step=epoch,
            )

    total_train_loss = 0.0
    batch_pbar = tqdm.tqdm(range(args.num_train_batches), desc="Train batches", leave=False)
    for i in batch_pbar:

        x, y = get_batch(
            train_data,
            batch_size=args.batch_size,
            context_length=args.context_length,
            device=device,
        )

        yhat = model(x).view(-1, args.vocab_size)
        loss = cross_entropy(yhat, y.flatten())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for param_group in optimizer.param_groups:
            param_group["lr"] = get_cosine_annealing_lr(
                epoch * args.num_train_batches + i,
                args.lr_min,
                args.lr_max,
                args.lr_warmup_iters,
                args.lr_cosine_cycle_iters,
            )

        total_train_loss += loss.item()

        batch_pbar.set_postfix({"mean_loss": total_train_loss / (i + 1)})

    # Log metrics to wandb.
    run.log({"mean_epoch_train_loss": total_train_loss / args.num_train_batches}, step=epoch)

    epoch_pbar.set_postfix({"mean_epoch_train_loss": total_train_loss / args.num_train_batches})

# Finish the run and upload any remaining data.
run.finish()

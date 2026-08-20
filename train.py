"""Train and evaluate STCRN on one traffic dataset."""

import argparse
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch import optim

import data_loader
import metric
from logger import get_logger
from model import STCRN


class Trainer:
    def __init__(self, args, scaler, device):
        self.model = STCRN(args).to(device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.learning_rate)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=args.scheduler_period,
            eta_min=args.minimum_learning_rate,
        )
        self.loss_function = nn.HuberLoss()
        self.scaler = scaler

    def train_batch(self, history, target, timestamps):
        self.optimizer.zero_grad()
        prediction = self.model(history, timestamps)
        prediction = self.scaler.inverse_transform(prediction)
        loss = self.loss_function(prediction, target)
        loss.backward()
        self.optimizer.step()
        return _batch_metrics(loss, prediction, target)

    def evaluate_batch(self, history, target, timestamps):
        prediction = self.model(history, timestamps)
        prediction = self.scaler.inverse_transform(prediction)
        loss = self.loss_function(prediction, target)
        return _batch_metrics(loss, prediction, target), prediction


def _batch_metrics(loss, prediction, target):
    return (
        loss.item(),
        metric.mae(prediction, target).item(),
        metric.rmse(prediction, target).item(),
        metric.mape(prediction, target, mask_value=0.0).item(),
    )


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_device(requested_device):
    if requested_device != "auto":
        return torch.device(requested_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _move_batch(batch, device):
    return tuple(tensor.to(device=device, dtype=torch.float32) for tensor in batch)


def _mean_metrics(metric_rows):
    return tuple(np.mean(metric_column) for metric_column in zip(*metric_rows))


def evaluate(data_loader_, trainer, device, keep_predictions=False):
    metric_rows = []
    predictions, targets = [], []
    trainer.model.eval()
    with torch.no_grad():
        for batch in data_loader_:
            history, target, timestamps = _move_batch(batch, device)
            batch_metrics, prediction = trainer.evaluate_batch(
                history, target, timestamps
            )
            metric_rows.append(batch_metrics)
            if keep_predictions:
                predictions.append(prediction.cpu())
                targets.append(target.cpu())

    mean_metrics = _mean_metrics(metric_rows)
    if keep_predictions:
        return mean_metrics, torch.cat(predictions), torch.cat(targets)
    return mean_metrics


def log_metrics(logger, split, epoch, values):
    loss, mae_value, rmse_value, mape_value = values
    logger.info(
        "Epoch %d | %s loss %.4f | MAE %.4f | RMSE %.4f | MAPE %.6f",
        epoch,
        split,
        loss,
        mae_value,
        rmse_value,
        mape_value,
    )


def run(args):
    set_seed(args.seed)
    device = select_device(args.device)
    run_name = f"{time.strftime('%Y%m%d-%H%M%S')}-{args.dataset}"
    output_dir = Path(args.output_dir) / run_name
    logger = get_logger(output_dir)
    logger.info("Arguments: %s", args)
    logger.info("Device: %s", device)

    train_loader, validation_loader, test_loader, scaler = data_loader.load_dataset(args)
    trainer = Trainer(args, scaler, device)
    logger.info("Trainable parameters: %d", trainer.model.parameter_count())
    logger.info("%s", trainer.model)

    best_validation_loss = math.inf
    best_epoch = 0
    early_stop_counter = 0
    checkpoint_path = output_dir / "best_model.pth"

    for epoch in range(1, args.epochs + 1):
        trainer.model.train()
        training_rows = []
        start_time = time.time()
        for batch in train_loader:
            history, target, timestamps = _move_batch(batch, device)
            training_rows.append(
                trainer.train_batch(history, target, timestamps)
            )
        logger.info(
            "Epoch %d | training time %.4f seconds",
            epoch,
            time.time() - start_time,
        )
        log_metrics(logger, "train", epoch, _mean_metrics(training_rows))
        trainer.scheduler.step()

        start_time = time.time()
        validation_metrics = evaluate(validation_loader, trainer, device)
        logger.info(
            "Epoch %d | validation time %.4f seconds",
            epoch,
            time.time() - start_time,
        )
        log_metrics(logger, "validation", epoch, validation_metrics)

        validation_loss = validation_metrics[0]
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            early_stop_counter = 0
            torch.save(trainer.model.state_dict(), checkpoint_path)
            log_metrics(
                logger,
                "test",
                epoch,
                evaluate(test_loader, trainer, device),
            )
        else:
            early_stop_counter += 1
            logger.info(
                "Early stopping counter: %d/%d",
                early_stop_counter,
                args.patience,
            )
            if early_stop_counter >= args.patience:
                logger.info("Early stopped; best epoch: %d", best_epoch)
                break

    trainer.model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    test_metrics, test_prediction, test_target = evaluate(
        test_loader, trainer, device, keep_predictions=True
    )
    log_metrics(logger, "final test", best_epoch, test_metrics)

    for horizon in range(args.output_len):
        horizon_prediction = test_prediction[:, horizon]
        horizon_target = test_target[:, horizon]
        logger.info(
            "Horizon %d | MAE %.4f | RMSE %.4f | MAPE %.6f",
            horizon + 1,
            metric.mae(horizon_prediction, horizon_target),
            metric.rmse(horizon_prediction, horizon_target),
            metric.mape(horizon_prediction, horizon_target, mask_value=0.0),
        )

    if args.save_predictions:
        np.savez(output_dir / "test_prediction.npz", data=test_prediction.numpy())
        np.savez(output_dir / "test_target.npz", data=test_target.numpy())


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_path", default="data")
    parser.add_argument("--dataset", default="PEMS08")
    parser.add_argument(
        "--split_ratio", type=float, nargs=3, default=[0.6, 0.2, 0.2]
    )
    parser.add_argument("--input_len", type=int, default=12)
    parser.add_argument("--output_len", type=int, default=12)
    parser.add_argument("--num_nodes", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--scheduler_period", type=int, default=100)
    parser.add_argument("--minimum_learning_rate", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--output_dir", default="checkpoints")
    parser.add_argument("--save_predictions", action="store_true")
    parser.add_argument("--spatial_hidden_dim", type=int, default=32)
    parser.add_argument("--temporal_hidden_dim", type=int, default=64)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())

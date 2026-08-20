"""Dataset preparation for the STCRN main experiments."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


TIMESTAMP_CONFIG = {
    "PEMS04": ("2018-01-01 00:00:00", "5min"),
    "PEMS07": ("2017-05-01 00:00:00", "5min"),
    "PEMS08": ("2016-07-01 00:00:00", "5min"),
}
TIMESTAMP_FILE_DATASETS = {"XMBRT", "HZMETRO", "BJMETRO"}


def generate_timestamps(start: str, periods: int, frequency: str):
    time_index = pd.date_range(start=start, periods=periods, freq=frequency)
    timestamp = np.stack(
        [
            time_index.month,
            time_index.day,
            time_index.weekday,
            time_index.hour,
            time_index.minute,
        ],
        axis=-1,
    )
    return torch.from_numpy(timestamp)


def split_chronologically(data, split_ratio):
    total_steps = len(data)
    train_end = int(total_steps * split_ratio[0])
    validation_end = train_end + int(total_steps * split_ratio[1])
    return data[:train_end], data[train_end:validation_end], data[validation_end:]


def create_windows(data, input_length: int, output_length: int):
    histories, targets = [], []
    window_length = input_length + output_length
    for start in range(len(data) - window_length + 1):
        window = data[start : start + window_length]
        histories.append(window[:input_length])
        targets.append(window[input_length:])
    if not histories:
        raise ValueError(
            "The dataset is shorter than input_len + output_len."
        )
    return torch.stack(histories), torch.stack(targets)


class StandardScaler:
    def __init__(self, mean: float, standard_deviation: float):
        if standard_deviation == 0:
            raise ValueError("Training data has zero standard deviation.")
        self.mean = mean
        self.standard_deviation = standard_deviation

    def transform(self, data):
        return (data - self.mean) / self.standard_deviation

    def inverse_transform(self, data):
        return data * self.standard_deviation + self.mean


class TrafficDataset(Dataset):
    def __init__(self, history, target, timestamp):
        self.history = history
        self.target = target
        self.timestamp = timestamp

    def __len__(self):
        return len(self.history)

    def __getitem__(self, index):
        return self.history[index], self.target[index], self.timestamp[index]


def _load_timestamps(dataset_dir: Path, dataset: str, periods: int):
    if dataset in TIMESTAMP_CONFIG:
        start, frequency = TIMESTAMP_CONFIG[dataset]
        return generate_timestamps(start, periods, frequency)
    if dataset in TIMESTAMP_FILE_DATASETS:
        timestamp_path = dataset_dir / f"{dataset}.npy"
        return torch.from_numpy(np.load(timestamp_path))
    raise ValueError(f"Unsupported dataset: {dataset}")


def load_dataset(args):
    """Return train/validation/test loaders and the training-data scaler."""

    if len(args.split_ratio) != 3 or not np.isclose(sum(args.split_ratio), 1.0):
        raise ValueError("split_ratio must contain three values that sum to 1.")

    dataset_dir = Path(args.data_path) / args.dataset
    data_path = dataset_dir / f"{args.dataset}.npz"
    traffic_data = torch.from_numpy(np.load(data_path)["data"][..., 0]).float()
    num_steps, num_nodes = traffic_data.shape

    if args.num_nodes is None:
        args.num_nodes = num_nodes
    elif args.num_nodes != num_nodes:
        raise ValueError(
            f"num_nodes={args.num_nodes}, but {args.dataset} contains {num_nodes} nodes."
        )

    timestamps = _load_timestamps(
        dataset_dir, args.dataset, periods=num_steps
    )
    histories, targets = create_windows(
        traffic_data, args.input_len, args.output_len
    )
    timestamp_history, _ = create_windows(
        timestamps, args.input_len, args.output_len
    )

    train_history, validation_history, test_history = split_chronologically(
        histories, args.split_ratio
    )
    train_target, validation_target, test_target = split_chronologically(
        targets, args.split_ratio
    )
    train_timestamp, validation_timestamp, test_timestamp = split_chronologically(
        timestamp_history, args.split_ratio
    )

    scaler = StandardScaler(
        mean=train_history.mean().item(),
        standard_deviation=train_history.std().item(),
    )
    train_history = scaler.transform(train_history)
    validation_history = scaler.transform(validation_history)
    test_history = scaler.transform(test_history)

    print(
        f"Training mean: {scaler.mean:.2f}, "
        f"training standard deviation: {scaler.standard_deviation:.2f}"
    )
    print(
        f"Train: {tuple(train_history.shape)}, "
        f"validation: {tuple(validation_history.shape)}, "
        f"test: {tuple(test_history.shape)}"
    )

    train_dataset = TrafficDataset(
        train_history, train_target, train_timestamp
    )
    validation_dataset = TrafficDataset(
        validation_history, validation_target, validation_timestamp
    )
    test_dataset = TrafficDataset(test_history, test_target, test_timestamp)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=args.batch_size, shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )
    return train_loader, validation_loader, test_loader, scaler

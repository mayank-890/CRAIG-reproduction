#!/usr/bin/env python3
"""Dataset download script for MNIST and CIFAR-10."""

import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from torchvision import datasets, transforms


def download_mnist(data_dir: str = "./data"):
    print(f"Downloading MNIST to {data_dir}...")
    datasets.MNIST(root=data_dir, train=True, download=True)
    datasets.MNIST(root=data_dir, train=False, download=True)
    print("MNIST download complete.")


def download_cifar10(data_dir: str = "./data"):
    print(f"Downloading CIFAR-10 to {data_dir}...")
    datasets.CIFAR10(root=data_dir, train=True, download=True)
    datasets.CIFAR10(root=data_dir, train=False, download=True)
    print("CIFAR-10 download complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download benchmark datasets.")
    parser.add_argument("--dataset", choices=["mnist", "cifar10", "all"], default="mnist")
    parser.add_argument("--data_dir", default="./data")
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    if args.dataset in ("mnist", "all"):
        download_mnist(args.data_dir)
    if args.dataset in ("cifar10", "all"):
        download_cifar10(args.data_dir)

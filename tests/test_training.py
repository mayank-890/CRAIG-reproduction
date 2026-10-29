"""Unit tests for models, baseline selectors, and training loop."""

import unittest
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from training.reproducibility import set_seed, get_device
from models.logistic_regression import LogisticRegression
from datasets.mnist import create_synthetic_mnist, TensorSubsetDataset
from baselines.random_selector import RandomSubsetSelector
from baselines.full_dataset import FullDatasetSelector
from training.trainer import train_model
from training.evaluator import evaluate_model


class TestModelAndBaselines(unittest.TestCase):
    def setUp(self):
        set_seed(42)
        self.device = torch.device("cpu")  # use CPU for deterministic unit tests

    def test_logistic_regression_shape_and_params(self):
        model = LogisticRegression(input_dim=784, num_classes=10)
        self.assertEqual(model.count_parameters(), 7850)
        
        # Test 2D input [B, 784]
        x2d = torch.randn(8, 784)
        out2d = model(x2d)
        self.assertEqual(out2d.shape, (8, 10))
        
        # Test 4D input [B, 1, 28, 28] auto-flatten
        x4d = torch.randn(8, 1, 28, 28)
        out4d = model(x4d)
        self.assertEqual(out4d.shape, (8, 10))

    def test_random_selector_counts_and_weights(self):
        selector = RandomSubsetSelector(seed=42)
        N = 200
        
        # 10% unweighted
        idx10, w10, _ = selector.select(num_samples=N, subset_ratio=0.10, weighted=False)
        self.assertEqual(len(idx10), 20)
        self.assertEqual(len(np.unique(idx10)), 20)  # without replacement
        np.testing.assert_allclose(w10, np.ones(20))
        
        # 25% weighted
        idx25, w25, _ = selector.select(num_samples=N, subset_ratio=0.25, weighted=True)
        self.assertEqual(len(idx25), 50)
        self.assertTrue(np.isclose(w25.sum(), float(N)))
        np.testing.assert_allclose(w25, np.full(50, 4.0))

    def test_full_dataset_selector(self):
        selector = FullDatasetSelector()
        N = 150
        idx, w, _ = selector.select(num_samples=N)
        self.assertEqual(len(idx), 150)
        np.testing.assert_array_equal(idx, np.arange(150))
        np.testing.assert_allclose(w, np.ones(150))

    def test_reproducibility(self):
        # Two runs with same seed should produce identical weights
        def run_sim(seed: int):
            set_seed(seed)
            train_ds, _ = create_synthetic_mnist(num_train=64, num_test=16)
            loader = DataLoader(train_ds, batch_size=16, shuffle=False)
            model = LogisticRegression(input_dim=784, num_classes=10)
            opt = torch.optim.SGD(model.parameters(), lr=0.01)
            train_model(
                model=model,
                train_loader=loader,
                val_loader=None,
                criterion=nn.CrossEntropyLoss(),
                optimizer=opt,
                epochs=2,
                device=self.device,
            )
            return model.linear.weight.clone().detach()

        w1 = run_sim(1234)
        w2 = run_sim(1234)
        w3 = run_sim(9999)
        
        torch.testing.assert_close(w1, w2)
        self.assertFalse(torch.allclose(w1, w3))

    def test_weighted_training_execution(self):
        # Verify weighted batches run without error and converge
        train_ds, test_ds = create_synthetic_mnist(num_train=80, num_test=20)
        weights = torch.full((80,), 2.5)
        weighted_ds = TensorSubsetDataset(train_ds.inputs, train_ds.targets, weights)
        train_loader = DataLoader(weighted_ds, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=20)
        
        model = LogisticRegression(input_dim=784, num_classes=10)
        opt = torch.optim.SGD(model.parameters(), lr=0.05)
        
        res = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=test_loader,
            criterion=nn.CrossEntropyLoss(),
            optimizer=opt,
            epochs=3,
            device=self.device,
            use_weights=True,
        )
        self.assertTrue(len(res["history"]["train_loss"]) == 3)
        self.assertLess(res["final_train_loss"], 10.0)


if __name__ == "__main__":
    unittest.main()

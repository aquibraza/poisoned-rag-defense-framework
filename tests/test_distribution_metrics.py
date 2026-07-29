"""Tests for defense/distribution_metrics.py (CORAL distance, RBF-kernel MMD
computed from cosine/Gram matrices only -- no raw embeddings needed).

Run with: python -m unittest tests.test_distribution_metrics -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from defense.distribution_metrics import (
    coral_distance_from_gram,
    mmd_rbf_distance_from_gram,
    slice_gram_blocks,
    squared_euclidean_from_cosine,
)


def _l2_normalize(z: np.ndarray) -> np.ndarray:
    return z / np.linalg.norm(z, axis=1, keepdims=True)


def _random_unit_vectors(seed: int, n: int, dim: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return _l2_normalize(rng.normal(size=(n, dim)))


def _raw_coral_from_embeddings(z_p: np.ndarray, z_c: np.ndarray) -> float:
    """Direct, textbook CORAL distance computed from the actual
    d-dimensional embeddings (mean-centered covariance Frobenius norm).
    Used only as an independent reference to validate
    `coral_distance_from_gram`'s Gram-matrix-only derivation."""
    n_p, n_c = z_p.shape[0], z_c.shape[0]
    zp_c = z_p - z_p.mean(axis=0, keepdims=True)
    zc_c = z_c - z_c.mean(axis=0, keepdims=True)
    cov_p = (zp_c.T @ zp_c) / (n_p - 1)
    cov_c = (zc_c.T @ zc_c) / (n_c - 1)
    diff = cov_p - cov_c
    return float(np.sum(diff * diff))


class TestSquaredEuclideanFromCosine(unittest.TestCase):
    def test_matches_direct_computation_for_unit_vectors(self):
        z = _random_unit_vectors(seed=1, n=6, dim=5)
        cos = z @ z.T
        d2 = squared_euclidean_from_cosine(cos)
        direct = np.zeros((6, 6))
        for i in range(6):
            for j in range(6):
                direct[i, j] = np.sum((z[i] - z[j]) ** 2)
        np.testing.assert_allclose(d2, direct, atol=1e-10)

    def test_clips_tiny_negative_floating_point_noise(self):
        cos = np.array([[1.0000000001]])
        d2 = squared_euclidean_from_cosine(cos)
        self.assertGreaterEqual(d2[0, 0], 0.0)


class TestSliceGramBlocks(unittest.TestCase):
    def test_blocks_match_manual_indexing(self):
        m = np.arange(100, dtype=np.float64).reshape(10, 10)
        poison_idx = [0, 2, 4, 6, 8]
        clean_idx = [1, 3, 5, 7, 9]
        g_pp, g_pc, g_cc = slice_gram_blocks(m, poison_idx, clean_idx)
        self.assertEqual(g_pp.shape, (5, 5))
        self.assertEqual(g_pc.shape, (5, 5))
        self.assertEqual(g_cc.shape, (5, 5))
        np.testing.assert_array_equal(g_pp, m[np.ix_(poison_idx, poison_idx)])
        np.testing.assert_array_equal(g_pc, m[np.ix_(poison_idx, clean_idx)])
        np.testing.assert_array_equal(g_cc, m[np.ix_(clean_idx, clean_idx)])


class TestCoralDistanceFromGram(unittest.TestCase):
    def test_zero_for_identical_distributions(self):
        z = _random_unit_vectors(seed=2, n=5, dim=8)
        g = z @ z.T
        distance = coral_distance_from_gram(g, g, g)
        self.assertAlmostEqual(distance, 0.0, places=10)

    def test_matches_raw_embedding_covariance_distance(self):
        """Regression guard for the Gram-matrix-only derivation itself:
        must match a direct from-embeddings CORAL computation exactly (up
        to floating-point precision), for groups of unequal size too."""
        z_p = _random_unit_vectors(seed=3, n=5, dim=7)
        z_c = _random_unit_vectors(seed=4, n=6, dim=7)
        g_pp, g_pc, g_cc = z_p @ z_p.T, z_p @ z_c.T, z_c @ z_c.T
        gram_trick = coral_distance_from_gram(g_pp, g_pc, g_cc)
        raw = _raw_coral_from_embeddings(z_p, z_c)
        self.assertAlmostEqual(gram_trick, raw, places=8)

    def test_finite_and_non_negative_for_random_groups(self):
        for seed in range(10):
            z_p = _random_unit_vectors(seed=seed, n=5, dim=6)
            z_c = _random_unit_vectors(seed=seed + 100, n=5, dim=6)
            g_pp, g_pc, g_cc = z_p @ z_p.T, z_p @ z_c.T, z_c @ z_c.T
            distance = coral_distance_from_gram(g_pp, g_pc, g_cc)
            self.assertTrue(np.isfinite(distance))
            self.assertGreaterEqual(distance, 0.0)

    def test_raises_for_single_point_group(self):
        g_pp = np.array([[1.0]])
        g_pc = np.array([[0.5, 0.5]])
        g_cc = np.array([[1.0, 0.5], [0.5, 1.0]])
        with self.assertRaises(ValueError):
            coral_distance_from_gram(g_pp, g_pc, g_cc)


class TestMmdRbfDistanceFromGram(unittest.TestCase):
    def test_zero_for_identical_distributions(self):
        z = _random_unit_vectors(seed=5, n=5, dim=8)
        g = z @ z.T
        distance = mmd_rbf_distance_from_gram(g, g, g)
        self.assertAlmostEqual(distance, 0.0, places=10)

    def test_finite_and_non_negative_for_random_groups(self):
        for seed in range(10):
            z_p = _random_unit_vectors(seed=seed, n=5, dim=6)
            z_c = _random_unit_vectors(seed=seed + 100, n=5, dim=6)
            g_pp, g_pc, g_cc = z_p @ z_p.T, z_p @ z_c.T, z_c @ z_c.T
            distance = mmd_rbf_distance_from_gram(g_pp, g_pc, g_cc)
            self.assertTrue(np.isfinite(distance))
            self.assertGreaterEqual(distance, 0.0)

    def test_increases_as_groups_separate(self):
        """Sanity check on the metric's direction: two well-separated unit
        vector clusters should have strictly larger MMD than two
        overlapping (identical) clusters."""
        z_p = _random_unit_vectors(seed=6, n=5, dim=8)
        z_c_close = z_p.copy()
        # A clean group deliberately anti-correlated with the poison group.
        z_c_far = -z_p.copy()
        g_pp = z_p @ z_p.T
        mmd_close = mmd_rbf_distance_from_gram(g_pp, z_p @ z_c_close.T, z_c_close @ z_c_close.T)
        mmd_far = mmd_rbf_distance_from_gram(g_pp, z_p @ z_c_far.T, z_c_far @ z_c_far.T)
        self.assertAlmostEqual(mmd_close, 0.0, places=10)
        self.assertGreater(mmd_far, mmd_close)

    def test_different_gamma_changes_result_but_stays_non_negative(self):
        z_p = _random_unit_vectors(seed=7, n=5, dim=6)
        z_c = _random_unit_vectors(seed=8, n=5, dim=6)
        g_pp, g_pc, g_cc = z_p @ z_p.T, z_p @ z_c.T, z_c @ z_c.T
        d1 = mmd_rbf_distance_from_gram(g_pp, g_pc, g_cc, gamma=0.5)
        d2 = mmd_rbf_distance_from_gram(g_pp, g_pc, g_cc, gamma=2.0)
        self.assertGreaterEqual(d1, 0.0)
        self.assertGreaterEqual(d2, 0.0)
        self.assertNotAlmostEqual(d1, d2, places=6)


if __name__ == "__main__":
    unittest.main()

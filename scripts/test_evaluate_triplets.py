"""Small protocol regression tests; no GPU or dataset required."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import h5py
import numpy as np
import pandas as pd
from evaluate_triplets import (
    HORIZONS,
    binary_targets,
    choose_threshold,
    complete_target_mask,
    decode,
    graph_metrics,
    load_data,
    paired_f1_delta,
    take_splits,
    topk,
    training_vocabulary,
    valid_clip_ends,
)
from extract_multiview_features import clip_starts, nonblack_counts, read_features


class TripletProtocolTests(unittest.TestCase):
    def test_future_targets_are_after_last_input(self):
        starts, ends = valid_clip_ends(181)
        np.testing.assert_array_equal(ends, starts + 60)
        self.assertTrue(np.all(ends + 60 < 181))
        for h in HORIZONS[1:]:
            self.assertTrue(np.all(ends + h * 15 > ends))
        self.assertEqual(int(ends[0] + 15), 75)

    def test_too_short_take_fails(self):
        with self.assertRaises(ValueError):
            valid_clip_ends(120)

    def test_missing_annotation_is_not_an_empty_graph(self):
        ends = np.array([60, 75])
        available = {60, 75, 120, 135}
        np.testing.assert_array_equal(complete_target_mask(ends, available), [True, False])

    def test_vocabulary_excludes_test_only_combinations(self):
        labels = {h: [frozenset({"a|holding|b"}), frozenset({"c|touching|d"})] for h in HORIZONS}
        self.assertEqual(training_vocabulary(labels, np.array([0])), ["a|holding|b"])

    def test_splits_keep_all_views_of_a_take_together(self):
        groups = np.repeat([f"take_{i}" for i in range(10)], 3)
        tested = []
        for _, train, val, test in take_splits(groups):
            self.assertFalse(set(groups[train]) & set(groups[val]))
            self.assertFalse(set(groups[train]) & set(groups[test]))
            self.assertFalse(set(groups[val]) & set(groups[test]))
            tested.extend(test)
        self.assertEqual(sorted(tested), list(range(len(groups))))

    def test_unknown_ground_truth_is_a_false_negative(self):
        truth = [frozenset({"a|r|b", "c|r|d"})]
        prediction = [frozenset({"a|r|b"})]
        metrics, _ = graph_metrics(truth, prediction)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertAlmostEqual(metrics["micro_f1"], 2 / 3)
        self.assertEqual(metrics["macro_f1"], 0.5)

    def test_predicted_only_triplet_is_a_false_positive(self):
        metrics, _ = graph_metrics([frozenset({"a|r|b"})], [frozenset({"x|r|y"})])
        self.assertEqual((metrics["tp"], metrics["fp"], metrics["fn"]), (0, 1, 1))

    def test_threshold_uses_validation_and_returns_sets(self):
        vocabulary = ["a|r|b", "c|r|d"]
        truth = [frozenset({"a|r|b"}), frozenset({"c|r|d"})]
        scores = np.array([[0.9, 0.3], [0.2, 0.8]])
        threshold = choose_threshold(scores, truth, vocabulary)
        self.assertEqual(decode(scores, vocabulary, threshold), truth)
        np.testing.assert_array_equal(binary_targets(truth, vocabulary), np.eye(2))

    def test_topk_small_vocabulary_has_no_duplicates(self):
        self.assertEqual(topk(np.array([[0.2, 0.7]]), ["a|r|b", "c|r|d"], 10), [frozenset({"a|r|b", "c|r|d"})])

    def test_empty_graphs_and_invalid_inputs(self):
        metrics, _ = graph_metrics([frozenset()], [frozenset()])
        self.assertEqual(metrics["micro_f1"], 1.0)
        self.assertEqual(metrics["exact_match"], 1.0)
        with self.assertRaises(ValueError):
            graph_metrics([], [])

    def test_paired_bootstrap_identical_predictions(self):
        rows = pd.DataFrame(dict(take_id=["1_1", "1_2"], n_samples=[10, 20], tp=[5, 10], fp=[1, 2], fn=[3, 5]))
        self.assertEqual(paired_f1_delta(rows, rows), dict(delta_micro_f1=0.0, ci_low=0.0, ci_high=0.0))

    def test_paired_bootstrap_rejects_mismatched_takes(self):
        a = pd.DataFrame(dict(take_id=["1_1"], n_samples=[10], tp=[5], fp=[1], fn=[3]))
        b = a.assign(take_id="2_1")
        with self.assertRaises(ValueError):
            paired_f1_delta(a, b)

    def test_multiview_fusion_and_label_timestamps_end_to_end(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "synthetic.h5"
            n_frames = 181
            starts = clip_starts(n_frames)
            cameras = ["external_1", "external_2", "head_surgeon"]
            take = dict(
                file=path,
                proc="MISS",
                phase=1,
                take=1,
                path="/data/MISS/1/take/1",
                n_frames=n_frames,
                cams=cameras,
            )
            with h5py.File(path, "w") as file:
                for frame in range(60, n_frames, 15):
                    file.create_dataset(
                        f"{take['path']}/annotations/frame_{frame}/rel_annotations",
                        data=np.array([[b"subject", b"looking", str(frame).encode("ascii")]]),
                    )
            for name in ("vjepa2", "dinov2"):
                for cam, value in zip(cameras, (1.0, 3.0, 100.0)):
                    np.savez(
                        root / f"{name}_{cam}.npz",
                        feat=np.full((len(starts), 1024), value, dtype=np.float32),
                        seq=np.full((len(starts), 8, 1024), value, dtype=np.float16),
                        n_frames=n_frames,
                        t_start=starts,
                        t_center=starts + 30,
                    )

            def local_path(name, cam, _take):
                return root / f"{name}_{cam}.npz"

            counts = np.full((len(starts), len(cameras)), 16, dtype=np.uint8)
            counts[0, 1] = 0
            counts[1, 2] = 0
            with (
                patch("evaluate_triplets.inventory", return_value=[take]),
                patch("evaluate_triplets.feature_path", side_effect=local_path),
                patch("evaluate_triplets.read_validity", return_value=counts),
            ):
                meta, features, labels, coverage = load_data("MISS", "exo")
                expected = np.full((4, 1024), 2.0)
                expected[0] = 1.0
                np.testing.assert_array_equal(features["vjepa2"], expected)
                self.assertEqual(meta.iloc[0].end_frame, 60)
                self.assertEqual(meta.iloc[0].cameras, "external_1")
                self.assertEqual(labels[0][0], frozenset({"subject|looking|60"}))
                self.assertEqual(labels[1][0], frozenset({"subject|looking|75"}))
                self.assertEqual(labels[4][0], frozenset({"subject|looking|120"}))
                self.assertEqual(coverage[0]["kept"], 4)
                self.assertEqual(coverage[0]["excluded_no_usable_view"], 1)
                _, ego_features, _, _ = load_data("MISS", "ego")
                np.testing.assert_array_equal(ego_features["vjepa2"], np.full((4, 1024), 100.0))
            with self.assertRaises(ValueError):
                read_features(root / "vjepa2_external_1.npz", n_frames + 15)

    def test_black_padding_is_detected_before_colour_conversion(self):
        raw = np.zeros((5, 2, 4, 4, 3), dtype=np.uint8)
        raw[:, 0] = 1
        raw[2:, 1] = 255
        idx = np.array([[0, 1, 2], [2, 3, 4]])
        np.testing.assert_array_equal(nonblack_counts(raw, idx, [0, 1]), [[3, 1], [3, 3]])


if __name__ == "__main__":
    unittest.main()

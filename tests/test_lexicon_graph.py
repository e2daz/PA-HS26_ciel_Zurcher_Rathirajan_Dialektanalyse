"""Tests for Levenshtein ring graph data preparation."""

import sys
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "domain"))

from lexicon_graph import build_levenshtein_ring_graph


class TestLevenshteinRingGraph(unittest.TestCase):
    def test_builds_reference_rings_and_pairwise_edges(self):
        rows = pd.DataFrame({
            "hypothesis_word": ["hat", "mit", "hat", "nichts", None],
            "similarity": [0.5, 0.4, 0.6, 0.7, 0.1],
            "dialect_region": ["Zürich", "Bern", "Zürich", "Bern", "Bern"],
        })

        graph = build_levenshtein_ring_graph(
            rows,
            "nicht",
            tfidf_by_hypothesis={"mit": 0.3, "hat": 0.2, "nichts": 0.1},
        )

        self.assertEqual(graph["candidate_count"], 3)
        self.assertEqual(graph["shown_count"], 3)
        self.assertEqual(graph["nodes"][0]["label"], "nicht")
        self.assertEqual(graph["nodes"][1]["label"], "mit")
        self.assertEqual(graph["nodes"][1]["distance"], 3)
        self.assertEqual(graph["nodes"][2]["count"], 2)
        self.assertEqual(graph["nodes"][2]["region_counts"], {"Zürich": 2})
        self.assertEqual(len(graph["edges"]), 3)
        self.assertEqual(graph["max_edge_distance"], 5)

    def test_uses_count_as_tie_breaker_and_limits_candidates(self):
        rows = pd.DataFrame({
            "hypothesis_word": ["aaa", "bbb", "bbb", "ccc"],
            "similarity": [0.2, 0.3, 0.4, 0.5],
        })

        graph = build_levenshtein_ring_graph(rows, "ref", top_n=2)

        self.assertEqual([node["label"] for node in graph["nodes"]], ["ref", "bbb", "aaa"])
        self.assertEqual(graph["candidate_count"], 3)
        self.assertEqual(graph["shown_count"], 2)

    def test_max_edge_distance_drops_distant_pairs(self):
        rows = pd.DataFrame({
            "hypothesis_word": ["nid", "nit", "nüt", "gar nüt"],
            "similarity": [0.5, 0.5, 0.5, 0.5],
        })

        graph = build_levenshtein_ring_graph(rows, "nicht", max_edge_distance=1)

        self.assertEqual(len(graph["nodes"]), 5)
        self.assertTrue(all(edge["distance"] <= 1 for edge in graph["edges"]))
        self.assertEqual(graph["max_edge_distance"], 1)

    def test_empty_candidates_returns_reference_only(self):
        rows = pd.DataFrame({
            "hypothesis_word": [None, "ref"],
            "similarity": [0.2, 1.0],
        })

        graph = build_levenshtein_ring_graph(rows, "ref")

        self.assertEqual(graph["candidate_count"], 0)
        self.assertEqual(len(graph["nodes"]), 1)
        self.assertEqual(graph["edges"], [])


if __name__ == "__main__":
    unittest.main()

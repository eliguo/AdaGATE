import json
import os
import shutil
import tempfile
import unittest

import stress_test_variants as stv


SAMPLE_RECORDS = [
    {
        "_id": "hotpot-1",
        "question": "Which city is the home of the university attended by Ada Lovelace's fictional student?",
        "answer": "London",
        "supporting_facts": [["Ada Lovelace College", 0], ["London", 0]],
        "context": [
            ["Ada Lovelace College", ["Ada Lovelace College is a fictional university in London."]],
            ["London", ["London is the capital city of England."]],
            ["Charles Babbage", ["Charles Babbage was an English polymath."]],
        ],
    },
    {
        "_id": "hotpot-2",
        "question": "What river runs through the city where Example Author was born?",
        "answer": "Seine",
        "supporting_facts": [["Example Author", 0], ["Paris", 0]],
        "context": [
            ["Example Author", ["Example Author was born in Paris."]],
            ["Paris", ["Paris is traversed by the Seine river."]],
            ["France", ["France is a country in Europe."]],
        ],
    },
]

RETRIEVAL_STYLE_RECORDS = [
    {
        "id": "retrieval-1",
        "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
        "answers": ["yes"],
        "ctxs": [
            {
                "title": "Ed Wood (film)",
                "text": "Ed Wood is a 1994 American biographical period comedy-drama film directed by Tim Burton.",
            },
            {
                "title": "Scott Derrickson",
                "text": "Scott Derrickson is an American director, screenwriter and producer.",
            },
        ],
    }
]


class StressTestVariantTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="stress-tests-")
        self.input_path = os.path.join(self.temp_dir, "toy_hotpot.json")
        with open(self.input_path, "w", encoding="utf-8") as handle:
            json.dump(SAMPLE_RECORDS, handle)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_compute_injected_count(self):
        self.assertEqual(stv.compute_injected_count(10, 0.1), 2)
        self.assertEqual(stv.compute_injected_count(4, 0.5), 4)

    def test_noise_injection_adds_metadata(self):
        pool = stv.build_global_passage_pool(SAMPLE_RECORDS)
        mutated = stv.inject_noise(
            example=SAMPLE_RECORDS[0],
            pool=pool,
            ratio=0.3,
            rng=stv.random.Random(7),
            overlap_threshold=0.2,
            example_index=0,
        )
        self.assertEqual(mutated["stress_test_type"], "noise_injection")
        self.assertTrue(len(mutated["stress_test_passage_metadata"]) >= len(SAMPLE_RECORDS[0]["context"]))

    def test_load_records_accepts_leading_whitespace_json(self):
        whitespace_path = os.path.join(self.temp_dir, "whitespace.json")
        with open(whitespace_path, "w", encoding="utf-8") as handle:
            handle.write("\n")
            json.dump(SAMPLE_RECORDS, handle)

        records, fmt = stv.load_records(whitespace_path)
        self.assertEqual(fmt, "json")
        self.assertEqual(len(records), len(SAMPLE_RECORDS))

    def test_noise_injection_preserves_ctxs_schema(self):
        pool = stv.build_global_passage_pool(RETRIEVAL_STYLE_RECORDS)
        mutated = stv.inject_noise(
            example=RETRIEVAL_STYLE_RECORDS[0],
            pool=pool,
            ratio=0.3,
            rng=stv.random.Random(7),
            overlap_threshold=0.2,
            example_index=0,
        )
        self.assertIn("ctxs", mutated)
        self.assertNotIn("context", mutated)
        self.assertIsInstance(mutated["ctxs"][0], dict)
        self.assertIn("title", mutated["ctxs"][0])
        self.assertIn("text", mutated["ctxs"][0])

    def test_redundancy_injection_adds_passages(self):
        mutated = stv.inject_redundancy(
            example=SAMPLE_RECORDS[0],
            ratio=0.3,
            rng=stv.random.Random(11),
            source_mode="support_only",
        )
        self.assertGreater(len(mutated["context"]), len(SAMPLE_RECORDS[0]["context"]))
        self.assertEqual(mutated["stress_test_type"], "redundancy_injection")

    def test_position_perturbation_preserves_size(self):
        mutated = stv.perturb_positions(
            example=SAMPLE_RECORDS[0],
            mode="support_back",
            rng=stv.random.Random(3),
        )
        self.assertEqual(len(mutated["context"]), len(SAMPLE_RECORDS[0]["context"]))
        self.assertEqual(mutated["stress_test_type"], "position_perturbation")

    def test_full_cli_generation(self):
        output_dir = os.path.join(self.temp_dir, "outputs")
        namespace = type("Args", (), {})()
        namespace.input = self.input_path
        namespace.dataset = "hotpotqa"
        namespace.output_dir = output_dir
        namespace.noise_ratios = [0.1]
        namespace.redundancy_ratios = [0.1]
        namespace.position_modes = ["shuffle_all"]
        namespace.noise_overlap_threshold = 0.2
        namespace.redundancy_source = "support_only"
        namespace.seed = 5
        namespace.output_format = "json"
        namespace.input_format = "json"

        manifest = stv.generate_variants(SAMPLE_RECORDS, "hotpotqa", namespace)
        self.assertEqual(len(manifest["noise_variants"]), 1)
        self.assertEqual(len(manifest["redundancy_variants"]), 1)
        self.assertEqual(len(manifest["position_variants"]), 1)

        noise_path = manifest["noise_variants"][0]["path"]
        self.assertTrue(os.path.exists(noise_path))


if __name__ == "__main__":
    unittest.main()

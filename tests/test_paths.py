"""Data locations must work independently of the author's cluster mounts."""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
VARIABLES = ("NEURAL_VLM_DATA_ROOT", "NSD_ROOT", "THINGS_EEG2_ROOT", "COCO_VAL2017",
             "COCO_TRAIN2017", "THINGS_DB_ROOT", "ARO_ROOT")


def locations(overrides):
    env = {k: v for k, v in os.environ.items() if k not in VARIABLES}
    env.update(overrides)
    code = "from analysis import paths; import json; print(json.dumps({k: str(getattr(paths, k)) for k in " + repr(VARIABLES[1:]) + "}))"
    return json.loads(subprocess.check_output([sys.executable, "-c", code], cwd=ROOT, env=env, text=True))


class DatasetPathsTest(unittest.TestCase):
    def test_defaults_are_portable(self):
        paths = locations({})
        self.assertEqual(paths["NSD_ROOT"], "datasets/natural-scenes-dataset")
        self.assertTrue(all(not Path(p).is_absolute() for p in paths.values()))

    def test_root_and_individual_overrides(self):
        paths = locations({"NEURAL_VLM_DATA_ROOT": "/tmp/research data",
                           "NSD_ROOT": "/tmp/separate nsd", "COCO_VAL2017": "/tmp/coco"})
        self.assertEqual(paths["NSD_ROOT"], "/tmp/separate nsd")
        self.assertEqual(paths["COCO_VAL2017"], "/tmp/coco")
        self.assertEqual(paths["THINGS_EEG2_ROOT"], "/tmp/research data/things_eeg_2")


if __name__ == "__main__":
    unittest.main()

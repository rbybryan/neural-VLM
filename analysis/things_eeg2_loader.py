"""THINGS-EEG2 loader for H3 temporal generalization analysis.

Dataset available at /scratch/ccn_datasets/things_eeg_2.

Layout:
  preprocessed_data/sub-XX/
      preprocessed_eeg_test.npy     — dict with keys:
          'preprocessed_eeg_data': (200, 80, 17, 100)  [images, repeats, ch, times]
          'ch_names': list of 17 channel names
          'times': (100,) in seconds, range -0.2 to 0.79 s
      preprocessed_eeg_training.npy — dict:
          'preprocessed_eeg_data': (16540, 4, 17, 100)
  image_set/
      image_metadata.npy  — structured array mapping image index to THINGS concept
      test_images/00001_aircraft_carrier/ ... (200 concept dirs, 1 image each)
      training_images/               (1654 concept dirs, 10 images each)

H3 uses the test set (200 concepts, 80 repeats, high SNR) for temporal generalization.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from analysis.paths import (
    THINGS_EEG2_PREPROCESSED,
    THINGS_EEG2_SUBJECTS,
    assert_things_eeg2_available,
)


def load_subject_test(
    subject: str,
    average_repeats: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Load test-set EEG for one subject.

    Returns:
        eeg:   (n_images, n_channels, n_times) averaged, or
               (n_images, n_repeats, n_channels, n_times) if average_repeats=False
        times: (n_times,) in milliseconds
    """
    path = THINGS_EEG2_PREPROCESSED / subject / "preprocessed_eeg_test.npy"
    data = np.load(path, allow_pickle=True).item()
    eeg = data["preprocessed_eeg_data"].astype(np.float32)
    # eeg: (n_images, n_repeats, n_channels, n_times)
    times_ms = (data["times"] * 1000).astype(np.float32)  # convert s -> ms

    if average_repeats:
        eeg = eeg.mean(axis=1)  # (n_images, n_channels, n_times)

    return eeg, times_ms


def load_group_test(
    subjects: list[str] | None = None,
    average_repeats: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Load and average test-set EEG across subjects.

    Returns:
        eeg:   (n_images, n_channels, n_times) group average
        times: (n_times,) in milliseconds
    """
    if subjects is None:
        subjects = THINGS_EEG2_SUBJECTS
    eegs = []
    times_ms = None
    for sub in subjects:
        eeg, t = load_subject_test(sub, average_repeats=average_repeats)
        eegs.append(eeg if average_repeats else eeg.mean(axis=1))
        if times_ms is None:
            times_ms = t
    return np.stack(eegs, axis=0).mean(axis=0), times_ms


def eeg_for_temporal_generalization(
    subjects: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return EEG in the shape expected by temporal_generalization().

    temporal_generalization() expects meg: (n_stim, n_timebins, n_sensors).
    This returns (200, 100, 17) averaged across subjects.
    """
    eeg, times_ms = load_group_test(subjects=subjects, average_repeats=True)
    # eeg: (n_images, n_channels, n_times) -> (n_images, n_times, n_channels)
    return eeg.transpose(0, 2, 1), times_ms


def subject_list() -> list[str]:
    assert_things_eeg2_available()
    return sorted(
        p.name for p in THINGS_EEG2_PREPROCESSED.iterdir()
        if p.is_dir() and p.name.startswith("sub-")
    )

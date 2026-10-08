"""Tests for model setup that run before any weights are downloaded."""

from __future__ import annotations

import inspect

import pytest

from PytorchWildlife.models import detection as pw_detection


def test_megadetector_v6_rejects_an_unknown_version() -> None:
    """An unknown `version` must raise a `ValueError` that lists the valid versions.

    The check happens before the weights are downloaded, so this test needs no network.
    """
    with pytest.raises(ValueError, match="Select a valid model version"):
        pw_detection.MegaDetectorV6(version="not-a-version")


def test_megadetector_v6_default_version_is_valid() -> None:
    """The default `version` must be one of the valid versions.

    `MegaDetectorV6()` with no arguments is what the README's Quick Start uses. The
    valid versions are taken from the error message for an unknown version, so the
    test needs no network.
    """
    default = (
        inspect.signature(pw_detection.MegaDetectorV6.__init__)
        .parameters["version"]
        .default
    )
    with pytest.raises(ValueError) as error:
        pw_detection.MegaDetectorV6(version="not-a-version")
    assert default in str(error.value)


def test_megadetector_v6_class_names() -> None:
    """MegaDetector V6 detects animals, people and vehicles, in that order."""
    assert pw_detection.MegaDetectorV6.CLASS_NAMES == {
        0: "animal",
        1: "person",
        2: "vehicle",
    }

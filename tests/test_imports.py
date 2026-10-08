"""Tests that the package and its main modules can be imported."""

from __future__ import annotations

import importlib

import pytest


@pytest.mark.parametrize(
    "module",
    [
        "PytorchWildlife",
        "PytorchWildlife.models.detection",
        "PytorchWildlife.models.classification",
        "PytorchWildlife.data",
        "PytorchWildlife.utils",
    ],
)
def test_module_can_be_imported(module: str) -> None:
    """Importing the module must not raise, for example because of a missing dependency."""
    importlib.import_module(module)


def test_quick_start_classes_exist() -> None:
    """The classes used in the README's Quick Start must exist."""
    from PytorchWildlife.models import classification as pw_classification
    from PytorchWildlife.models import detection as pw_detection

    assert hasattr(pw_detection, "MegaDetectorV6")
    assert hasattr(pw_classification, "AI4GAmazonRainforest")

"""Tests for package-level startup configuration."""

import os

import pytest

import localtalk

pytestmark = pytest.mark.unit


def test_deprecated_hf_transfer_flag_migrates_to_xet(monkeypatch):
    monkeypatch.setenv("HF_HUB_ENABLE_HF_TRANSFER", "1")
    monkeypatch.delenv("HF_XET_HIGH_PERFORMANCE", raising=False)

    localtalk._migrate_deprecated_hf_transfer_env()

    assert "HF_HUB_ENABLE_HF_TRANSFER" not in os.environ
    assert os.environ["HF_XET_HIGH_PERFORMANCE"] == "1"


def test_explicit_xet_flag_is_preserved(monkeypatch):
    monkeypatch.setenv("HF_HUB_ENABLE_HF_TRANSFER", "true")
    monkeypatch.setenv("HF_XET_HIGH_PERFORMANCE", "0")

    localtalk._migrate_deprecated_hf_transfer_env()

    assert "HF_HUB_ENABLE_HF_TRANSFER" not in os.environ
    assert os.environ["HF_XET_HIGH_PERFORMANCE"] == "0"

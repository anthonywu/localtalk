"""Unit tests for MLX compatibility patches."""

from __future__ import annotations

import importlib
import sys
import types

import pytest

pytestmark = pytest.mark.unit


def _make_fake_mlx_lm_utils(*, has_save_model: bool, has_save_weights: bool) -> types.ModuleType:
    """Build a fake ``mlx_lm.utils`` module with the specified attributes."""
    fake_utils = types.ModuleType("mlx_lm.utils")
    if has_save_model:
        fake_utils.save_model = lambda *a, **kw: "save_model"
    if has_save_weights:
        fake_utils.save_weights = lambda *a, **kw: "save_weights"
    return fake_utils


def _make_fake_mlx_lm(fake_utils: types.ModuleType) -> types.ModuleType:
    """Build a fake ``mlx_lm`` package containing the given fake ``utils`` submodule."""
    fake = types.ModuleType("mlx_lm")
    fake.utils = fake_utils
    return fake


def _inject_mlx_lm(monkeypatch, fake_utils):
    """Inject fake mlx_lm + mlx_lm.utils into sys.modules and return the utils module."""
    fake = _make_fake_mlx_lm(fake_utils)
    monkeypatch.setitem(sys.modules, "mlx_lm", fake)
    monkeypatch.setitem(sys.modules, "mlx_lm.utils", fake_utils)
    return fake_utils


class TestPatchMlxLmUtils:
    def test_patch_when_save_model_exists_no_save_weights(self, monkeypatch):
        """When save_model exists but save_weights does not, alias is created."""
        fake_utils = _make_fake_mlx_lm_utils(has_save_model=True, has_save_weights=False)
        _inject_mlx_lm(monkeypatch, fake_utils)

        # Re-import the module so it runs patch_mlx_lm_utils with our fake
        import localtalk.utils.mlx_compat as compat

        importlib.reload(compat)

        assert hasattr(fake_utils, "save_weights")
        assert fake_utils.save_weights is fake_utils.save_model

    def test_patch_when_save_weights_already_exists(self, monkeypatch):
        """When save_weights already exists, it should not be overwritten."""
        fake_utils = _make_fake_mlx_lm_utils(has_save_model=True, has_save_weights=True)
        original = fake_utils.save_weights
        _inject_mlx_lm(monkeypatch, fake_utils)

        import localtalk.utils.mlx_compat as compat

        importlib.reload(compat)

        assert fake_utils.save_weights is original

    def test_patch_when_neither_exists(self, monkeypatch):
        """When neither save_model nor save_weights exists, no alias is created and no error raised."""
        fake_utils = _make_fake_mlx_lm_utils(has_save_model=False, has_save_weights=False)
        _inject_mlx_lm(monkeypatch, fake_utils)

        import localtalk.utils.mlx_compat as compat

        importlib.reload(compat)

        assert not hasattr(fake_utils, "save_weights")

    def test_patch_when_mlx_lm_absent(self, monkeypatch):
        """When mlx_lm cannot be imported, ImportError is swallowed silently."""
        # Remove any cached mlx_lm modules
        monkeypatch.delitem(sys.modules, "mlx_lm", raising=False)
        monkeypatch.delitem(sys.modules, "mlx_lm.utils", raising=False)

        # Make import of mlx_lm fail by setting it to None in sys.modules
        # (Python treats None in sys.modules as "module not found")
        monkeypatch.setitem(sys.modules, "mlx_lm", None)

        import localtalk.utils.mlx_compat as compat

        # Should not raise
        importlib.reload(compat)
        # If we got here without exception, the test passes

    def test_patch_creates_callable_alias(self, monkeypatch):
        """The created save_weights alias should be callable and behave like save_model."""
        fake_utils = _make_fake_mlx_lm_utils(has_save_model=True, has_save_weights=False)
        _inject_mlx_lm(monkeypatch, fake_utils)

        import localtalk.utils.mlx_compat as compat

        importlib.reload(compat)

        assert callable(fake_utils.save_weights)
        assert fake_utils.save_weights() == "save_model"


class TestPatchMlxMetalDeviceInfo:
    """Tests for the mx.metal.device_info → mx.device_info(mx.gpu) redirect."""

    def test_patch_redirects_to_device_info_with_gpu(self, monkeypatch):
        """The patched function should call mx.device_info with mx.gpu."""
        # Build a fake mlx.core module with metal submodule
        fake_mx = types.ModuleType("mlx.core")

        # Track what device_info is called with
        calls: list = []
        fake_gpu = "fake_gpu"

        def fake_device_info(device=None):
            calls.append(device)
            return {"device_name": "Test GPU", "max_recommended_working_set_size": 1000}

        fake_mx.device_info = fake_device_info
        fake_mx.gpu = fake_gpu

        fake_metal = types.ModuleType("mlx.metal")
        fake_metal.device_info = lambda: None  # Original deprecated function
        fake_mx.metal = fake_metal
        fake_mx.core = fake_mx  # Satisfy `import mlx.core as mx`

        monkeypatch.setitem(sys.modules, "mlx", fake_mx)
        monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)

        import localtalk.utils.mlx_compat as compat

        importlib.reload(compat)

        # The patched function should use mx.gpu
        result = fake_metal.device_info()
        assert result == {"device_name": "Test GPU", "max_recommended_working_set_size": 1000}
        assert calls == [fake_gpu]

    def test_patch_preserves_gpu_specific_behavior(self, monkeypatch):
        """The shim must always query GPU, not the current default device."""
        fake_mx = types.ModuleType("mlx.core")

        fake_gpu = "gpu_device"
        fake_cpu = "cpu_device"

        def fake_device_info(device=None):
            if device is fake_gpu:
                return {"device_name": "GPU", "max_recommended_working_set_size": 1000}
            return {"device_name": "CPU"}  # Missing the key mlx-lm needs

        fake_mx.device_info = fake_device_info
        fake_mx.gpu = fake_gpu
        fake_mx.cpu = fake_cpu

        fake_metal = types.ModuleType("mlx.metal")
        fake_metal.device_info = lambda: None
        fake_mx.metal = fake_metal
        fake_mx.core = fake_mx  # Satisfy `import mlx.core as mx`

        monkeypatch.setitem(sys.modules, "mlx", fake_mx)
        monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)

        import localtalk.utils.mlx_compat as compat

        importlib.reload(compat)

        # Even if someone changed the default device, the shim should return GPU info
        result = fake_metal.device_info()
        assert "max_recommended_working_set_size" in result
        assert result["device_name"] == "GPU"

    def test_patch_swallows_import_error(self, monkeypatch):
        """When mlx cannot be imported, the patch should not raise."""
        monkeypatch.setitem(sys.modules, "mlx", None)
        monkeypatch.setitem(sys.modules, "mlx.core", None)

        import localtalk.utils.mlx_compat as compat

        # Should not raise
        importlib.reload(compat)

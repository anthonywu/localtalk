"""Compatibility patches for MLX library version mismatches."""


def patch_mlx_metal_device_info():
    """Redirect mx.metal.device_info to the non-deprecated mx.device_info.

    MLX prints a deprecation warning to stderr (at the C++ level, bypassing
    Python's warnings module) every time mx.metal.device_info() is called.
    mlx-lm calls it during generation to set the wired memory limit.

    mx.metal.device_info() always queries Metal GPU 0, while bare
    mx.device_info() queries the current default device (which may be CPU
    if another library changed it). We wrap it in a function that
    explicitly passes mx.gpu to preserve the original GPU-specific behavior.
    """
    try:
        import mlx.core as mx

        metal = getattr(mx, "metal", None)
        if metal is not None and hasattr(mx, "device_info") and hasattr(metal, "device_info"):
            gpu = mx.gpu

            def _device_info(*args, **kwargs):  # noqa: ANN001, ANN002, ANN003
                return mx.device_info(gpu)

            metal.device_info = _device_info
    except (ImportError, AttributeError):
        pass


def patch_mlx_lm_utils():
    """Patch mlx_lm.utils to provide save_weights as an alias for save_model.

    This is needed because mlx_audio expects save_weights but newer mlx_lm
    only has save_model.
    """
    try:
        import mlx_lm.utils

        if not hasattr(mlx_lm.utils, "save_weights") and hasattr(mlx_lm.utils, "save_model"):
            # Create an alias
            mlx_lm.utils.save_weights = mlx_lm.utils.save_model
    except ImportError:
        pass


# Apply patches on import
patch_mlx_metal_device_info()
patch_mlx_lm_utils()

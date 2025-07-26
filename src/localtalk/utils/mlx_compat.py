"""Compatibility patches for MLX library version mismatches."""


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
patch_mlx_lm_utils()

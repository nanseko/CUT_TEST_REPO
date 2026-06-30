"""SAR preprocessing pipeline for the CUT training workflow.

See docs/README_pipeline.md for the design.
"""

from preprocessing.pipeline import (
    run_pipeline, preprocess_single, scan_images, export_cut_layout,
    default_config, build_steps, build_optical_reference_cdf,
    save_reference_cdf, load_reference_cdf,
)
from preprocessing.steps import (
    STEP_REGISTRY, DEFAULT_STEP_ORDER, SPECKLE_METHODS,
    HISTOGRAM_MODES, INTENSITY_MODES,
)
from preprocessing.metrics import (
    compute_dataset_metrics, image_metrics, format_metrics, save_metrics_log,
    METRIC_INFO, METRIC_KEYS,
)

__all__ = [
    'run_pipeline', 'preprocess_single', 'scan_images', 'export_cut_layout',
    'default_config', 'build_steps', 'build_optical_reference_cdf',
    'save_reference_cdf', 'load_reference_cdf',
    'STEP_REGISTRY', 'DEFAULT_STEP_ORDER', 'SPECKLE_METHODS',
    'HISTOGRAM_MODES', 'INTENSITY_MODES',
    'compute_dataset_metrics', 'image_metrics', 'format_metrics', 'save_metrics_log',
    'METRIC_INFO', 'METRIC_KEYS',
]

"""SAR preprocessing pipeline for the CUT training workflow.

See docs/README_pipeline.md for the design.
"""

# Core (must always be present). If this fails the package is fundamentally
# broken / mismatched and we let the error surface.
from preprocessing.pipeline import (
    run_pipeline, preprocess_single, scan_images, export_cut_layout,
    default_config, build_steps, build_optical_reference_cdf,
)
from preprocessing.steps import (
    STEP_REGISTRY, DEFAULT_STEP_ORDER, SPECKLE_METHODS,
    HISTOGRAM_MODES, INTENSITY_MODES,
)

__all__ = [
    'run_pipeline', 'preprocess_single', 'scan_images', 'export_cut_layout',
    'default_config', 'build_steps', 'build_optical_reference_cdf',
    'STEP_REGISTRY', 'DEFAULT_STEP_ORDER', 'SPECKLE_METHODS',
    'HISTOGRAM_MODES', 'INTENSITY_MODES',
]

# Newer optional features. These live in files added later; if the user updated
# only SOME files (e.g. an old preprocessing/pipeline.py without
# save_reference_cdf), importing them raises ImportError. Tolerate that so the
# core package + GUI still load, and print a clear, actionable message telling
# the user exactly which file is stale — instead of a cryptic hard ImportError.
try:
    from preprocessing.pipeline import save_reference_cdf, load_reference_cdf
    from preprocessing.metrics import (
        compute_dataset_metrics, image_metrics, format_metrics, save_metrics_log,
        METRIC_INFO, METRIC_KEYS,
    )
    from preprocessing.optimize import (
        optimize_orders, enumerate_candidates, build_pipeline_steps,
        evaluate_pipeline, PERMUTE_STEPS,
        resolve_inception_weights, INCEPTION_FILENAME,
        optimize_params, tunable_steps_in_order, load_best_pipeline,
    )
    __all__ += [
        'save_reference_cdf', 'load_reference_cdf',
        'compute_dataset_metrics', 'image_metrics', 'format_metrics', 'save_metrics_log',
        'METRIC_INFO', 'METRIC_KEYS',
        'optimize_orders', 'enumerate_candidates', 'build_pipeline_steps',
        'evaluate_pipeline', 'PERMUTE_STEPS',
        'resolve_inception_weights', 'INCEPTION_FILENAME',
        'optimize_params', 'tunable_steps_in_order', 'load_best_pipeline',
    ]
    FEATURES_OK = True
except Exception as _exc:  # stale / partially-updated files
    import warnings as _warnings
    FEATURES_OK = False
    _warnings.warn(
        '\n[preprocessing] 최신 기능(사전 히스토그램·성능지표·순서 최적화) 로드 실패: '
        f'{_exc}\n'
        '  -> preprocessing/ 폴더의 파일들이 서로 다른 버전(부분 업데이트)입니다.\n'
        '  -> preprocessing/pipeline.py, metrics.py, optimize.py, __init__.py 를 모두 '
        '최신본으로 교체하고 preprocessing/__pycache__/ 를 삭제하세요.',
        stacklevel=2)

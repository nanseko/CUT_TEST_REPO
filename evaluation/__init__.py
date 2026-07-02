"""CUT model output evaluation.

Given images produced by a trained CUT checkpoint (fake_B, real_A, real_B —
as saved by test.py / the Web-UI inference tab), compute:
  - FID / KID of fake_B against a real EO (optical) reference set
    (domain-gap / generative-quality proxy).
  - EPI / CC / PSNR of real_A vs fake_B (structure/edge preservation, guards
    against hallucination).
  - PSNR / SSIM of real_B vs idt_B = G(real_B) (identity-path fidelity; a
    genuinely paired sanity check for comparing backbones/hyperparameters).
  - No-reference quality of fake_B (sharpness/contrast/entropy).

Results are logged (append-only CSV + JSON) under
<results_dir>/<name>/eval_logs/ so backbone/attention/lambda sweeps can be
compared over time. See docs/EVALUATION.md.
"""

from evaluation.generate import (
    build_generator_from_cfg, load_generator_checkpoint, generate_from_folder,
)
from evaluation.evaluate import (
    compute_domain_metrics, compute_structure_metrics, compute_identity_metrics,
    compute_quality_metrics, run_evaluation, load_eval_log, EVAL_CSV_COLUMNS,
)

__all__ = [
    'build_generator_from_cfg', 'load_generator_checkpoint', 'generate_from_folder',
    'compute_domain_metrics', 'compute_structure_metrics', 'compute_identity_metrics',
    'compute_quality_metrics', 'run_evaluation', 'load_eval_log', 'EVAL_CSV_COLUMNS',
]

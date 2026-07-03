# 하이퍼파라미터 자동 탐색 (Successive Halving) — GUI 탭 10

**질문**: "학습을 하면서 스스로 CUT/Attention 파라미터를 최적화할 수 없나?"

**답**: 한 번의 학습 *도중* 온라인으로 파라미터를 바꾸는 것은 권장하지 않습니다 — **GAN 학습 손실은 품질 지표가 아니기 때문**입니다. `G_GAN`/`NCE`가 낮아진다고 이미지가 좋아지는 게 아니라(적대적 균형값일 뿐), 온라인 적응에 쓸 신뢰할 신호 자체가 없습니다. 신뢰할 수 있는 방법은 **여러 번의 짧은 학습을 자동으로 돌려, 별도 품질 지표(FID/EPI)로 랭킹**하는 것입니다. 이 모듈이 그걸 버튼 하나로 자동화합니다.

## 방식: Successive Halving (2단계 예산 배분)

```
후보 N개 샘플링 (attention × 손실가중치 조합)
  │
  ├─ Stage 1: 각 후보를 짧게 학습 (기본 300장 × 15 epoch)
  │           → test.py 추론 → FID(EO 대비)·EPI 평가 → 랭킹
  │
  ├─ Stage 2: 상위 K개만 이어학습(continue_train)으로 +45 epoch (전체 데이터)
  │           → 재평가 → 최종 1위
  │
  └─ best_hparams.json 저장 → 🧬 버튼으로 탭 4/5에 적용 → 본 학습(전체 epoch)
```

- **목적함수**: `fid`(↓, fake_B ↔ EO 세트 — SAR→EO 품질과 가장 직결) 또는 `epi`(↑, real_A ↔ fake_B 구조 보존). EO 폴더가 없거나 torch/가중치가 없으면 자동으로 epi로 전환됩니다.
- **재개 가능**: 완료된 (trial, stage)는 `hparam_results.csv`에 즉시 기록되고, 재실행 시 건너뜁니다. 중단해도 손해가 없습니다.
- **중복 제거(canonicalisation)**: 의미상 동일한 설정(예: `attention_type=none`일 때의 위치/reduction 조합, `lambda_grad=lambda_lap=0`일 때의 `reflector_weighted`)은 하나로 접혀 같은 trial을 두 번 돌리지 않습니다.
- InceptionV3와 EO 기준 특징은 **한 번만** 로드/추출해 모든 trial에서 재사용합니다. 오프라인 가중치는 `docs/OFFLINE_FID.md` 참고.

## 탐색 공간 (기본값)

| 파라미터 | 후보 |
|---|---|
| `attention_type` | none / coord / cbam |
| attention 위치 | enc / enc+res / enc+res+dec |
| `attention_reduction` | 8 / 16 |
| `lambda_grad` | 0 / 1.0 |
| `lambda_lap` | 0 / 0.5 |
| `lambda_coherence` | 0 / 0.5 |
| `lambda_color` | 0 / 1.0 |
| `reflector_boost` | 3 / 5 |
| `reflector_weighted` | off / on |
| `saliency_patch_sampling` | off / on |

전체 그리드는 수백 가지이므로 **랜덤 샘플링으로 N개**(기본 12)를 뽑습니다. `netG`(resnet/hrnet), crop_size 등 **탭 1~5의 나머지 설정은 그대로 기본값**으로 모든 trial에 적용됩니다 — 백본을 비교하고 싶으면 netG를 바꿔 탐색을 두 번 돌리고 CSV를 비교하세요.

코드에서 공간을 바꾸려면:
```python
from evaluation.hparam_search import hparam_search, DEFAULT_SPACE
space = dict(DEFAULT_SPACE, lambda_grad=[0.0, 0.5, 1.0, 2.0])
```

## 사용법 (GUI 탭 10)

1. 탭 1~5에서 기본 설정(dataroot, netG, crop 등)을 맞춘다.
2. 탭 10에서 EO 폴더·trial 수·stage 예산을 정하고 **🚀 자동 탐색 실행**.
3. 끝나면(또는 중단 후 재개) **🧬 최적 설정을 탭 4/5에 적용** → 💾 저장 → 본 학습.
   - 최적 trial의 체크포인트(`checkpoints_dir/hps_<sig>`)에서 **이어학습**으로 계속할 수도 있습니다.

## 예상 시간 (RTX 5080, 256px 기준 대략)

- trial당: 300장×15epoch ≈ 4,500 iter 학습 + 100장 추론 + 평가 → **수 분~10분대**
- 기본 설정(12 trial + 상위 3개 이어학습) → **한나절~하룻밤**
- 더 빠르게: stage1 장수/epoch을 줄이거나 trial 수를 줄이세요. 단 stage1이 너무 짧으면(< ~10 epoch) 랭킹 신뢰도가 떨어집니다.

## 주의

- **GPU 직렬 실행**입니다(한 번에 한 trial). trial 체크포인트가 `checkpoints_dir/hps_*` 로 여러 개 쌓이므로, 탐색용 `checkpoints_dir`를 따로 두는 걸 권장합니다.
- stage1 랭킹은 "짧은 학습" 기준의 근사입니다 — 초반 FID 순위와 최종 순위가 항상 일치하진 않습니다. 그래서 상위 K개를 stage2로 더 길게 확인하는 것이고, K를 늘리면 안전해지는 대신 느려집니다.
- Optuna 같은 베이지안 라이브러리도 대안이지만, 오프라인 사내망 + 수십 개 수준의 후보에서는 이 의존성 없는 랜덤+halving 구현이 실용적으로 충분합니다.
- 검증: `python tests/test_hparam_search.py` — 동등 설정 dedup, 결정적 샘플링, 실제 train/test를 통한 소형 엔드투엔드, 재개 시 재학습 0회를 확인합니다.

# HANDOFF — SAR→Optical CUT 프로젝트 인수인계 하네스

> **목적**: 이 문서 하나로 다른 Claude(또는 개발자)가 이 프로젝트의 현재 상태·설계·미완 작업을
> 즉시 파악하고 바로 작업을 이어받을 수 있도록 한다. **작업을 시작하기 전에 이 문서를 처음부터 끝까지 읽을 것.**

- **작성일**: 2026-07-08
- **대상 저장소**: `nanseko/cut_test_repo`
- **작업 브랜치**: `claude/nice-ptolemy-5UQLR` (⚠️ **이 브랜치에서만 개발·푸시**. 다른 브랜치 푸시 금지)
- **현재 BUILD 마커**: `gui.py` 안 `BUILD = '2026-07-08.5 (preprocessing-param-optimizer)'`
- **가장 최근 커밋 계열**: 전처리 **파라미터** 자동 최적화(⑨, 좌표하강) — 순서 탐색(⑧) 다음 단계로
  각 스텝의 세부 파라미터(클리핑 세기/speckle 윈도우/intensity mode/clahe)를 튜닝, best_pipeline.json
  자동 연결, 재개 가능(`docs/PARAM_OPTIMIZE.md`); 그 이전: stall watchdog을
  `util/subprocess_watchdog.py` 공용 모듈로 추출해
  **학습(탭 6) + 추론(탭 7) + 하이퍼파라미터 탐색(탭 10)의 모든 train.py/test.py 서브프로세스**에
  동일 적용(`docs/RESILIENT_TRAINING.md`) — 전에는 탭 6에만 있어서 탭 10에서 trial 하나가 멈추면
  탐색 전체가 영원히 멈추는 버그가 있었음; 그 이전: rectify 무음 예외 버그 수정 + 탭별 데이터 폴더
  지정(전처리/평가/후처리) + 하이퍼파라미터 탐색 그리드 확장(lambda 3단계 + grad_no_blur) + stage1
  top_k 기본값 5; 그 이전: 설정 자동저장 + 절대경로 수정 + 체크포인트별 하이퍼파라미터 로그/복원
  (`docs/CONFIG_PERSISTENCE.md`); 그 이전: 손실 그래프(`util/loss_plot.py`) + 추가 attention(self/eca/cbam_coord)

> ⚠️ **환경 주의**: 세션 컨테이너는 ephemeral이라 재개 시 torch/torchvision/matplotlib/dominate/gradio/
> opencv 등이 사라질 수 있다. 검증 전 `pip install torch torchvision matplotlib dominate gradio
> opencv-python-headless` 필요할 수 있음 (기본 pip 인덱스는 프록시 통과, `download.pytorch.org`는 403).
> visdom/GPUtil은 빌드 실패해도 `display_id=0` CPU 학습에는 불필요.
> **⚠️ opencv가 없으면 rectify(탭 9)가 "성공, 0개 검출"처럼 보이는 게 아니라 이제 명확한 오류를
> 낸다 — "동작 안 함"을 보면 먼저 `python -c "import cv2"` 로 opencv 설치를 확인할 것.**

---

## 0. 30초 요약 (TL;DR)

이 프로젝트는 **공식 CUT(Contrastive Unpaired Translation, PyTorch)** 를 포크해서
**SAR(위성 레이더) → Optical(광학) 영상 변환**에 특화시킨 것이다. 핵심 사용자는
**비개발자**이며, 거의 모든 조작을 **Gradio Web-UI(`gui.py`, 11개 탭)** 로 수행한다.

이번(그리고 이전) 세션에서 추가한 것은 크게 7덩어리:
1. **Attention**(CBAM/Coordinate) + **HRNet** 생성기 (블러/작은물체 대응)
2. **SAR 전처리 파이프라인** + **전처리 순서/파라미터 자동 탐색**(좌표하강)
3. **모델 출력 평가 모듈**(`evaluation/`) — FID/KID/EPI/SSIM
4. **작은 물체(요트·탱크·건물) 형상 보존** — reflector 가중 손실 + saliency 샘플링 + coherence 손실 + 직사각 후처리
5. **하이퍼파라미터 자동 탐색**(Successive Halving)
6. **시스템 전체 hang(정지) 자동 복구/감지 watchdog**(학습/추론/하이퍼파라미터 탐색 공통)
7. **표적(건물·차량·비행체) 시각적 강조**(CFAR+saliency 검출 → saliency 가중 국소 강조)

모든 기능은 **기본값에서 원본 CUT과 100% 동일하게 동작**하도록 설계됨(옵트인). 검증은 `tests/`.

---

## 1. 사용자 컨텍스트 (매우 중요)

- 사용자는 **한국어**로 소통하며, **비개발자에 가깝다**. 답변·UI·문서는 **한국어**로, 개념 설명을 곁들여서 한다.
- 실제 사용 환경: **로컬 Windows PC + RTX 5080(16GB)**, 데이터 **약 1000장** 규모, 학습에 **1~2일** 소요.
- 사용자가 겪은/관심 가진 실제 문제들 (지금까지 대화 맥락):
  1. 전처리 단계 **순서**가 결과에 영향 → 자동 탐색 요구
  2. 강반사체 주변 **블러**, 건물 뭉개짐 → HRNet + 구조 손실
  3. 요트/탱크가 **구름 모양 블롭**으로 변함 → reflector 가중 + coherence
  4. 각진 모서리가 **직각으로 안 나옴** → coherence 손실 + 직사각 후처리
  5. 학습 하이퍼파라미터를 **스스로 최적화**하고 싶음 → 자동 탐색
  6. 장시간 학습이 **도중에 멈춤** → watchdog + OS 설정 가이드
- **GAN 손실은 품질 지표가 아니다**라는 점을 사용자에게 반복적으로 안내해 왔다. 품질 판단은
  항상 **탭 8의 FID/EPI 수치**로 하도록 유도할 것.

---

## 2. 저장소 지도 (핵심 파일만)

```
gui.py                      ← ★ 11개 탭 Gradio Web-UI. 모든 기능의 진입점. (~2080줄)
train.py / test.py          ← 공식 CUT 학습/추론 스크립트 (거의 원본)
models/
  cut_model.py              ← ★ CUTModel. compute_G_loss/compute_D_loss에 부가손실 배선
  losses_extra.py           ← ★ 우리가 추가한 모든 손실(grad/lap/color/coherence/reflector)
  networks.py               ← 생성기/판별기. HRNetGenerator, PatchSampleF(weights 인자), GANLoss
  attention.py              ← CBAM / CoordinateAttention 모듈
options/
  base_options.py           ← attention_*, hrnet_*, netG choices 등
  train_options.py          ← lr/epochs/gan_mode 등
preprocessing/              ← SAR 전처리(순수 NumPy/Pillow, 선택적 cv2/torch)
  pipeline.py, steps.py     ← 파이프라인 엔진 + 개별 스텝
  optimize.py               ← 전처리 "순서" 자동 탐색(2단계, 재개형)
  img_metrics.py            ← psnr/cc/epi/ssim (공용, numpy)
  fid_utils.py              ← FID/KID + InceptionV3(오프라인 가중치 지원)
  metrics.py                ← ENL/speckle_index 등 무참조 품질
evaluation/                 ← ★ CUT "출력" 평가 + 후처리 + HP탐색
  evaluate.py               ← FID/KID/EPI/SSIM 종합 평가 + CSV 로깅
  generate.py               ← 체크포인트로 idt_B=G(real_B) 생성
  rectify.py                ← 직사각 스냅 후처리(cv2, 선택적)
  hparam_search.py          ← Successive Halving 자동 탐색
tests/                      ← ★ 회귀 테스트(작업 후 반드시 실행)
docs/                       ← 기능별 상세 설계 문서(아래 8절 참조)
```

⚠️ **`preprocessing/`와 `evaluation/`의 `__init__.py`는 방어적 import**(부분 업데이트/의존성
누락에도 나머지가 동작)로 되어 있다. 새 모듈 추가 시 이 패턴을 깨지 말 것.

---

## 3. 지금까지 완료한 작업 (커밋 단위)

이번 프로젝트 라인의 커밋 이력(최신 → 과거). 각 항목은 **완료·검증됨**.

| 커밋 | 내용 | 관련 문서/테스트 |
|---|---|---|
| (최신) | **표적(건물·차량·비행체) 시각적 강조**(탭 9 하단) — CFAR(sigma/ca)+saliency 검출(stage A) + saliency 가중 국소 강조(unsharp/guided/CLAHE, stage B). 순수 NumPy(clahe만 opencv) | `docs/TARGET_ENHANCEMENT_SPEC.md`, `docs/TARGET_ENHANCE.md`, `tests/test_target_enhance.py` |
| | **전처리 파라미터 자동 최적화**(⑨, 좌표하강) — 순서 탐색(⑧) 다음 단계, 스텝별 세부 파라미터 튜닝, best_pipeline.json 자동 연결, 재개 가능 | `docs/PARAM_OPTIMIZE.md`, `tests/test_param_optimize.py` |
| (최신) | **시스템 전체 stall watchdog** — `util/subprocess_watchdog.py` 신설, 학습/추론/하이퍼파라미터 탐색 3곳 모두에 동일 적용 | `docs/RESILIENT_TRAINING.md`, `tests/test_subprocess_watchdog.py` |
| (최신) | **rectify 무음 예외 버그 수정**(opencv 부재 시 "성공,0개"로 오인되던 문제) + **탭 3/8/9 데이터 폴더 직접 지정**(자동저장) + **하이퍼파라미터 그리드 확장**(lambda 3단계, grad_no_blur) + stage1 top_k=5 | `docs/SMALL_OBJECT_PRESERVATION.md`, `docs/CONFIG_PERSISTENCE.md`, `docs/HPARAM_SEARCH.md`, `tests/test_rectify.py` |
| (최신) | **설정 자동저장**(절대경로 수정) + **체크포인트별 하이퍼파라미터 로그/복원** + 아키텍처 불일치 경고 | `docs/CONFIG_PERSISTENCE.md`, `tests/test_config_persistence.py` |
| (최신) | **epoch별 D/G/NCE 손실 그래프**(`util/loss_plot.py`) + **하이브리드/Self/ECA attention** | `docs/LOSS_CURVES_AND_ATTENTION.md`, `tests/test_loss_plot.py` |
| `3bc69b2` | **학습 hang watchdog** + 자동 재시작 | `docs/RESILIENT_TRAINING.md`, `tests/test_training_watchdog.py` |
| `7a9f426` | **하이퍼파라미터 자동 탐색**(Successive Halving) + 탭 10 | `docs/HPARAM_SEARCH.md`, `tests/test_hparam_search.py` |
| `7fec2a8` | **coherence 손실** + **직사각 후처리**(탭 9) | `docs/SMALL_OBJECT_PRESERVATION.md` |
| `21943c2` | **reflector 가중 손실** + **saliency 패치 샘플링** | `docs/SMALL_OBJECT_PRESERVATION.md`, `tests/test_attention_port.py` |
| `8f700aa` | **CUT 출력 평가 모듈**(`evaluation/`) + 탭 8 | `docs/EVALUATION.md`, `tests/test_evaluation.py` |
| `8fe8efb`,`0b1236c`,`e3c1811`,`fe27c0d`,`5b08a75` | **전처리 순서 자동 탐색** + FID/오프라인 가중치 | `docs/OFFLINE_FID.md`, `preprocessing/optimize.py` |
| `b2c9e72` | **HRNet** 옵션 + 사용자 지정 PatchNCE 레이어 + 지표 로깅 | `docs/ATTENTION_PORT.md` |
| (이전) | **Attention 포팅** + 구조/색 손실 + Web-UI 기반 | `docs/ATTENTION_PORT.md`, `docs/GUI.md` |

**작업 파일 상태**: 워킹트리 clean(모든 작업 커밋·푸시 완료). 진행 중이던 미완 코드 없음.

---

## 4. 도구별 상세 설계 (인수자가 반드시 이해할 핵심)

### 4.1 손실 함수 (`models/losses_extra.py`) — 모든 것이 여기 모여 있음

모든 손실은 **`[-1,1]` 범위 NCHW 텐서**를 입력받고, `compute_G_loss()`에서
`self.loss_G`에 `lambda_* ×`로 더해진다. `lambda=0`이면 원본과 동일.

| 함수 | 시그니처 | 비교 대상 | 하는 일 |
|---|---|---|---|
| `gradient_loss` | `(source, generated, blur, weighted, boost)` | real_A ↔ fake_B | 1차 미분(에지 위치) L1 |
| `laplacian_loss` | `(source, generated, blur, weighted, boost)` | real_A ↔ fake_B | 2차 미분(뾰족함/고주파) L1 |
| `color_moment_loss` | `(generated, reference)` | idt_B ↔ real_B | 채널별 평균/표준편차 매칭 |
| `reflector_saliency_map` | `(source, window=5, boost=3)` | (맵 생성) | SAR 국소 밝기 피크 → `[1,1+boost]` 가중치 맵 |
| `reflector_saliency_weights_for_shapes` | `(source, shapes, boost)` | (맵 생성) | 위 맵을 PatchNCE 레이어 해상도로 리샘플 |
| `coherence_loss` | `(source, generated, boost, window, energy_scale)` | real_A(가중) ↔ fake_B(sharpness) | 강반사체 위치의 "블롭화" 억제 |
| `edge_sharpness_map` | `(x, window, energy_scale)` | (맵 생성) | **energy × coherence** (구조텐서) |

**⚠️ coherence_loss의 핵심 설계 함정 (반드시 기억)**:
- 방향 일관성(coherence)만 쓰면 **부드럽게 blur된 원("구름")도 방향은 일관**되어 높은 점수를 받는다.
  → 반드시 **energy(에지 강도)를 곱해서** 걸러야 한다 (`edge_sharpness_map`).
- energy 정규화는 **고정 절대 스케일(`energy_scale`)** 사용. 이미지 자기 자신의 percentile로
  정규화하면 흐린 이미지의 가장 흐린 에지도 "최대 선명"으로 오인됨(초기 구현에서 실제로 겪은 버그).
- **coherence_loss는 90도 직각을 보장하지 않는다.** 직각 모서리는 두 방향이 만나 오히려
  coherence가 낮아지는 구조적 한계. "뭉개짐 억제"용이지 "정확한 형상"용이 아님.
  정확한 직각이 필요하면 **탭 9 직각 후처리(`evaluation/rectify.py`)** 를 쓴다.

**reflector 3종 세트** (모두 `reflector_saliency_map` 공유, `reflector_boost`로 강도 조절):
- `--reflector_weighted`: grad/lap 손실을 물체 위치에서 가중
- `--saliency_patch_sampling`: PatchNCE 패치 샘플링을 물체 쪽으로 편향
  (`PatchSampleF.forward`의 `weights` 인자, `patch_ids=None`일 때만 적용, 하위호환 유지)
- `--reflector_boost`: 위 둘 + coherence 공용 강도(기본 3.0)

**권장 시작값**(사용자에게 이미 안내한 값): `lambda_grad=1.0, lambda_lap=0.5,
lambda_coherence=0.5, lambda_color=1.0, reflector_weighted=on,
saliency_patch_sampling=on, reflector_boost=3.0, grad_no_blur=off`.

### 4.2 CUT 출력 평가 (`evaluation/`)

- `evaluate.py::run_evaluation()` — 제너레이터(로그 스트리밍). 4축 평가:
  - **FID/KID** (fake_B ↔ EO 세트) — 주 품질 지표, 낮을수록 좋음
  - **EPI/CC/PSNR** (real_A ↔ fake_B) — 구조 보존/허상 가드레일, 파일명 stem 매칭
  - **PSNR/SSIM** (real_B ↔ idt_B=G(real_B)) — identity 충실도, **유일한 진짜 짝 비교**
  - 무참조 품질(sharpness/entropy) — fake_B 단독
  - 결과는 `<results_dir>/<name>/eval_logs/eval_results.csv`에 **실험별 누적**(재개형).
- `generate.py` — `test.py`가 idt_B를 저장 안 하므로(학습 때만 visual), 체크포인트로 직접 `G(real_B)` 재계산.
  `build_generator_from_cfg(cfg)`는 gui.py의 CONFIG_KEYS와 동일한 아키텍처 옵션으로 생성기 복원.
- `fid_utils.py` — FID + **KID(소표본 안정)**. InceptionV3 가중치는 **오프라인 로컬 우선 탐색**(사내망 대응).

### 4.3 직사각 후처리 (`evaluation/rectify.py`, 탭 9)

- cv2 기반(선택적 의존성). Otsu 임계 → 컨투어 → `minAreaRect`/`approxPolyDP` 스냅.
- **원형도(rectangularity) 필터** 기본 0.85: 원은 이론상 π/4≈0.785이므로 이보다 높게 잡아야
  원/블롭이 걸러진다(0.5로 두면 원이 통과됨 — 실측으로 확인한 값).
- 학습이 아니라 **후처리**라 90도 기하학적 보장. 단 "벡터 도형처럼" 보임 → 형상 추출/판독 용도.
- **⚠️ 과거 버그(수정됨)**: `rectify_folder`의 파일별 처리가 `except Exception: continue`로 **완전히
  무음**이었다. opencv 미설치 시 모든 파일에서 `ImportError`가 나는데 그게 조용히 삼켜져서 "폴더 전체를
  분석 안 하고 아무 결과도 안 나오는데 오류도 없음"으로 보였다(GUI는 "✅ 완료: 0개 검출"이라는
  **오해의 소지가 있는 성공 메시지**까지 보여줬음). 수정: `_require_cv2()`를 루프 진입 전 **한 번만**
  호출해 즉시 명확한 오류로 fail-fast, 파일별 실패는 개수+사유를 모아 반환(`(csv_path, n_regions,
  n_processed, n_failed, failures)` — **반환 튜플이 2개→5개로 바뀌었으니 새 호출부 추가 시 주의**).
  `gui.py::cut_rectify`도 "N/M 실패"를 명시적으로 보여주도록 갱신. 검증: `tests/test_rectify.py`.
- 입력 폴더는 이제 `comp['rectify_input_dir']`(탭 9, 비우면 자동 유도)로 **직접 지정 가능**.

### 4.4 하이퍼파라미터 자동 탐색 (`evaluation/hparam_search.py`, 탭 10)

- **온라인 자가 최적화는 의도적으로 안 만듦**(GAN 손실 = 품질 아님). 대신 **짧은 학습 N개 →
  FID/EPI 랭킹 → 상위 K개 이어학습**(Successive Halving). **stage1 top_k 기본값 = 5**(과거 3).
- `canonicalize()`가 동등 설정을 중복 제거(attention=none일 때 위치/reduction 무시,
  lambda_grad=lambda_lap=0일 때 reflector_weighted/grad_no_blur 무시 등).
- 재개형: `hparam_results.csv`에 (trial, stage)별 기록, 재실행 시 건너뜀.
- 학습/추론 명령은 **주입된 `build_train_cmd`/`build_test_cmd`**(gui.py 것)를 그대로 사용 →
  gui.py와 항상 동일한 명령 보장, hparam_search.py는 gradio 의존성 없음.
- 탐색 공간: attention(type/위치/reduction) + 손실 가중치. **모든 `lambda_*`는 동일하게 [0, 0.5, 1.0]
  3단계**(과거엔 항목마다 그리드가 달라 비교가 왜곡됐음), on/off 토글(`reflector_weighted`,
  `saliency_patch_sampling`, `grad_no_blur` — 마지막은 과거에 탐색 공간에서 누락돼 있었음)도 전부
  독립적으로 켜짐/꺼짐 탐색.

### 4.9 탭별 데이터 폴더 지정 (탭 3/8/9)

- 탭 8(평가)/탭 9(후처리)는 원래 `results_dir/name/test_<epoch>/images/...` 로 **강제 유도**된 경로만
  볼 수 있었음(사용자가 임의 폴더를 분석할 방법이 없었음). `eval_fake_dir`/`eval_real_a_dir`/
  `rectify_input_dir`(신규) + `eval_eo_dir`/`eval_real_b_dir`(기존 로컬 위젯 → `CONFIG_KEYS` 승격)로
  **비우면 기존처럼 자동, 채우면 그 폴더 사용**. `evaluation/evaluate.py::run_evaluation()`에
  `fake_dir`/`real_a_dir` 오버라이드 파라미터 추가(하위호환: 기본 None → 기존 자동유도 동작).
- 탭 3에는 `dataroot`의 **단방향 미러**(탭3→탭1로만 반영, `.change()` 1개)를 추가. 양방향 동기화는
  Gradio 이벤트 순환 위험을 피하려고 의도적으로 안 함(설계 노트: 이 문서 앞부분 참고).
- 전처리(탭 2)의 `pp_in`/`pp_out`은 이미 있었지만 **같은 상대경로 버그**(`PP_CONFIG_PATH`)와
  **자동저장 누락**이 있었음 — `gui_config.json`과 동일한 방식으로 수정.

### 4.10 전처리 파라미터 최적화 (`preprocessing/optimize.py::optimize_params`, 탭 2 아코디언 ⑨)

- **순서 탐색(⑧ `optimize_orders`)의 다음 단계**: 순서를 고정하고 각 스텝의 수치 파라미터를 **좌표하강**
  으로 튜닝(한 번에 한 파라미터 그리드 스윕 → 최적값 확정 → 다음). 후보 수가 곱이 아니라 합.
- **각 스텝이 자기 `PARAM_SPACE`를 선언**(`preprocessing/steps.py`) — 비어 있으면 자동 제외이므로
  구조 스텝(resize/channel/normalize/validate)은 특별 처리 없이 빠진다. 새 파라미터 추가는 여기에
  후보 리스트만 넣으면 되고, 중첩키는 dotted(`'clahe.clip_limit'`). 조건부 유효성(frost 전용 damping,
  cv2 필요 clahe)은 `_relevant_param_space`에서 pruning.
- **stage-1과 대부분 재사용**: `build_steps`/평가 루프/PSNR·CC·EPI 등 지표/재개형 append-only CSV를
  그대로 씀. `evaluate_param_pipeline`은 `evaluate_pipeline`에 param_overrides만 얹은 것.
- **자동 연결**: order/speckle 미지정 시 out_dir의 `best_pipeline.json`(⑧ 결과)에서 로드. 결과는
  `best_params_pipeline.json`(order+param_overrides+full_steps).
- **⚠️ 재개 결정론성 함정(수정됨)**: 좌표하강 경로는 metric의 strict 비교로 갈리는데, fresh run은
  full-precision metric으로, resume은 CSV에서 읽은 4자리 반올림값으로 경로를 정하면 경로가 갈라져
  재개가 no-op이 아니게 된다. → `evaluate()`가 **fresh/resume 모두 4자리 반올림값을 쓰도록** 통일해
  경로를 결정론적으로 만듦(테스트로 3회 재실행 시 행 증가 0 확인).

### 4.11 표적 시각적 강조 (`evaluation/target_enhance.py`, 탭 9 하단)

- `docs/TARGET_ENHANCEMENT_SPEC.md`(연구 명세, 코드 없음)의 stage A+B 구현. C(클래스별 형상/신뢰도
  주석)·D(표적 초해상)는 **미구현**(요청 시 별도 진행).
- **Stage A(검출)**: `cfar_mask`가 픽셀 주변 **훈련 링(가드 링 제외)** 통계로 적응 임계값 판정(SAR
  표준 CFAR). `sigma`(국소 평균+k·표준편차, 분포무관, 기본) / `ca`(고전 Cell-Averaging, 지수분포
  클러터 가정 — 렌더링된 8비트 이미지에는 `pfa`를 느슨하게 잡아야 함, 실측: pfa=1e-3이면 target/mean
  대비가 3배 정도로는 전혀 안 걸림). `brightness_saliency01`(국소 대비, 최댓값 근처로 정규화)로
  교차검증해 고립 픽셀 오검출 억제 → 형태학 열림/닫힘 → 연결요소(cv2 가속, 순수 파이썬 BFS
  폴백)+면적 필터.
- **Stage B(강조)**: `unsharp_mask`/`guided_detail_boost`/`masked_clahe` 모두 **saliency 맵을 픽셀별
  블렌딩 가중치**로 사용(가중치=0인 배경은 원본과 동일하게 유지) — 전역 적용 시 SAR 스페클이 배경까지
  증폭되는 문제(명세서 §1.2/§7)를 원천 차단. `guided_detail_boost`는 Guided Filter(He, Sun & Tang
  2010, `evaluation/rectify.py`와 무관한 독립 구현)로 base/detail 분해 후 디테일만 증폭 — 일반
  박스블러 대비 엣지 근처 halo가 적다(테스트에서 정량 확인: 엣지 근처 평균오차 0.0155 vs 0.2727).
  `masked_clahe`만 Lab L채널 CLAHE + opencv 필요.
- `evaluation/rectify.py::rectify_folder`와 **동일한 실패 처리 관례**를 그대로 따름: `enhance_folder`가
  `'clahe' in methods`이면 루프 진입 전 opencv 존재를 **한 번만** 확인해 fail-fast, 파일별 실패는
  개수+사유를 모아 `(csv_path, n_regions, n_processed, n_failed, failures)` 5-튜플로 반환.
- GUI(`gui.cut_enhance_targets`)는 `cut_rectify`와 같은 UX 패턴: `comp['enhance_input_dir']`(비우면
  `results_dir/name/test_<epoch>/images/fake_B` 자동 유도) + epoch 텍스트박스, 결과는 `rectified/`와
  별도인 `enhanced/` 폴더에 저장(사각 스냅과 독립 실행 가능, 같은 `fake_B`를 입력으로 공유).
- 테스트 함정(재사용 시 주의): CFAR의 **가드 링은 반드시 대상 물체 전체를 덮을 만큼 커야** 한다 —
  합성 테스트에서 가드 링보다 큰 블롭을 쓰면 블롭 자체가 자신의 훈련 링에 새어 들어가 임계값이
  터무니없이 커지는(threshold>1) 실패를 실제로 겪었다(`tests/test_target_enhance.py` 주석 참고).

### 4.5 시스템 전체 stall watchdog (`util/subprocess_watchdog.py`)

- **처음엔 탭 6(학습)에만** hang 감지가 있었다. 이후 "하이퍼파라미터 탐색/추론에서도 hang이 생기는 것
  같다"는 리포트로 확인해보니 **탭 7(추론)·탭 10(탐색)의 서브프로세스 실행부는 워치독이 전혀 없는
  순수 블로킹 `readline` 루프**였다(특히 탭 10의 `_stream_subprocess`는 한 trial이 멈추면 **전체
  다중 시간 탐색이 영원히 멈추는** 심각한 버그였음). 그래서 로직을 `util/subprocess_watchdog.py`
  하나로 추출해 세 곳 모두 같은 구현을 쓰도록 통일했다.
- **두 인터페이스, 하나의 구현**:
  - `run_watched(cmd, cwd, on_line, ...)` — 블로킹 콜백형. `gui.py::training_worker`가 사용
    (백그라운드 스레드에서 실행되므로 자체적으로 스트리밍할 필요 없음, STATE에 side-effect만 남기면 됨).
  - `run_watched_stream(cmd, cwd, holder, ...)` — 제너레이터형(라인을 즉시 yield, 최종 결과는
    `holder` dict에 기록). `run_inference`(Gradio 제너레이터라 진행상황을 직접 스트리밍해야 함)와
    `evaluation/hparam_search.py::_stream_subprocess`(기존 `_stream_subprocess(cmd,cwd,log,tag,holder)`
    관례와 그대로 호환)가 사용. `run_watched`는 내부적으로 `run_watched_stream`을 소비하는 얇은 래퍼.
- 배경 스레드+`queue.Queue`로 stdout을 read-with-timeout(순수 블로킹 `readline()`엔 타임아웃이 없어서
  이 패턴이 필요) → **N분(기본 20) 진행 로그 없으면** 프로세스 kill.
- `MIN_STALL_SECONDS=60` 하한(오탐 방지, 테스트에서 `모듈.MIN_STALL_SECONDS` monkeypatch로 낮춤).
  **주의**: `evaluation/hparam_search.py`가 `from util.subprocess_watchdog import ... MIN_STALL_SECONDS`로
  값을 복사해오므로, 몽키패치는 `sys.modules['evaluation.hparam_search'].MIN_STALL_SECONDS`에 해야
  한다(`util.subprocess_watchdog.MIN_STALL_SECONDS`를 바꿔도 이미 복사된 값엔 영향 없음). 게다가
  `evaluation/__init__.py`가 `from evaluation.hparam_search import hparam_search`로 **같은 이름의
  함수를 재노출**하기 때문에 `import evaluation.hparam_search as hs`는 서브모듈이 아니라 그 **함수**를
  가리키게 된다 — 실제로 테스트 작성 중 이 함정에 걸렸다(`tests/test_hparam_search.py`의 주석 참고).
- 대상별로 "멈추면 어떻게 되는지"가 다르다(4.4/4.6 참고): 학습(탭 6)=`--continue_train` 자동 재시작,
  추론(탭 7)=오류 표시 후 수동 재시도, 탐색(탭 10)=그 trial만 실패 처리하고 다음 trial로 진행(기존
  `run_stage()`의 `holder.get('rc',1)!=0` 실패 판정 로직이 `rc=None`(hang으로 kill됨)도 이미
  올바르게 처리하고 있어서 **`run_stage()` 자체는 수정할 필요가 없었음** — `_stream_subprocess`만
  내부적으로 교체).
- **코드로 못 고치는 OS 원인**은 `docs/RESILIENT_TRAINING.md`에 정리(절전 모드, Windows Update
  재부팅, 네트워크 드라이브 I/O, num_threads, 백신 예외).

### 4.6 손실 그래프 (`util/loss_plot.py`, 탭 6)

- train.py/visualizer가 이미 남기는 `checkpoints_dir/<name>/loss_log.txt`를 **단일 진실 소스**로
  파싱(학습 스텝에 상태를 끼워넣지 않음) → epoch별 평균 → `loss_history.csv` + `loss_curve.png`.
  train.py epoch 끝에 **한 줄 훅**(`update_loss_plot`, best-effort)으로 매 epoch 갱신.
- `D = mean(D_real, D_fake)`로 **합성**(원본 로그는 D_real/D_fake만 찍음). matplotlib 선택적(없으면
  CSV만). **그래프 텍스트는 반드시 ASCII** — matplotlib 기본 폰트에 한글 glyph가 없어 깨짐(실측 확인).
- GUI 표시: `STATE.loss_png` → `_format_status`가 7-튜플 반환 → `monitor_outputs`의 `gr.Image`.

### 4.7 attention 확장 (`models/attention.py`)

- 6종: `none/cbam/coord/eca/self/cbam_coord`. 화이트리스트가 **4곳**에 흩어져 있어 추가 시 전부
  갱신 필요: `attention.py`(factory+`ATTENTION_TYPES`), `networks.py`(assert), `base_options.py`
  (choices), `gui.py`(Radio) — 그리고 `evaluation/hparam_search.py`의 탐색 공간.
- `self`(non-local self-attention)는 `gamma=0` 초기화로 **학습 시작 시점엔 항등 함수**. 메모리
  O((HW)²)라 **저해상도(resblocks)에만** 삽입 권장 — encoder/decoder(고해상도)에 켜면 매우 무거움.
- `cbam_coord`는 `SequentialAttention(CBAM, CoordinateAttention)` — 직렬 결합, 순서 고정(CBAM 먼저).

### 4.8 설정 영속성 (`gui.py`, 전역 + 탭 6)

- **`DEFAULT_CONFIG_PATH`는 반드시 절대경로**(`REPO_ROOT` 기준)여야 함 — 상대경로였을 때 "서버
  재시작마다 경로가 초기화됨" 버그의 근본 원인이었음(CWD에 따라 다른 파일을 가리킴).
- **자동 저장**: `ordered_inputs`의 모든 위젯에 `.change(do_save, ...)` 배선 → 어떤 필드를 바꿔도
  전체 cfg가 즉시 저장됨. 새 CONFIG_KEYS 위젯 추가 시 이 반복문이 `ordered_inputs`에서 자동으로
  주워가므로 별도 배선 불필요(단, 5절의 5단계 배선 규칙은 그대로 지킬 것).
- **체크포인트별 스냅샷**: `start_training`이 매 실행 시 `checkpoints_dir/<name>/gui_train_config.json`
  에 전체 cfg를 저장(신규/이어서 학습 무관하게 항상 최신 시도를 기록). `cfg_apply_checkpoint`가
  `hps_apply_best`(4.4)와 동일 패턴으로 전체 `CONFIG_KEYS` 위젯에 `gr.update()` 반환 — **단
  `continue_train`은 액션 플래그라 복원 대상에서 제외**(그대로 복원하면 처음 스냅샷의 False로
  덮어써져 혼란을 줌).
- **아키텍처 불일치 가드**: `continue_train=True`로 시작할 때마다 `ARCH_CRITICAL_KEYS`(netG/normG/
  attention_*/no_antialias*/hrnet_*, 가중치 shape을 결정하는 키들)를 저장된 스냅샷과 비교해
  다르면 ⚠️⚠️ 로그(막지는 않음). 새 아키텍처 관련 CONFIG_KEYS를 추가하면 **이 리스트에도 추가할 것**.

---

## 5. GUI 탭 구조 (`gui.py`, 진입점 `build_ui()`)

| 탭 | 이름 | 주요 콜백 |
|---|---|---|
| 0 | 데이터셋 다운로드/정리 (M4-SAR) | (Colab 전용 다운로드) |
| 1 | 데이터셋 (dataroot) | — |
| 2 | SAR 전처리 (학습 전) | `optimize_orders` (순서 탐색) |
| 3 | 기본 학습 파라미터 | — |
| 4 | CUT 파라미터 | (손실 가중치·reflector·coherence 위젯) |
| 5 | Attention 설정 | `gui_recommend_nce` |
| 6 | 학습 실행/모니터링 | `start_training`/`stop_training`/`training_worker`/`refresh_checkpoint_dropdown`/`cfg_apply_checkpoint` |
| 7 | 추론/테스트 | `run_inference` |
| 8 | 모델 평가 (CUT 출력) | `cut_evaluate`/`eval_table_rows` |
| 9 | 형상 후처리 (직사각 스냅) + 표적 시각적 강조 | `cut_rectify` / `cut_enhance_targets` |
| 10 | 하이퍼파라미터 자동 탐색 | `cut_hparam_search`/`hps_apply_best` |

**핵심 배선 규칙**:
- `CONFIG_KEYS`(gui.py 상단, **54개**) = 모든 탭 위젯의 순서. `ordered_inputs`로 콜백에 전달.
- 새 파라미터 추가 시 반드시: ① `CONFIG_KEYS`에 추가 ② `DEFAULTS`에 기본값
  ③ (학습 파라미터라면) `build_train_cmd`에 CLI 인자 ④ 탭 UI에 `comp['key']=...` 위젯 ⑤ (해당되면) options 파일에 argparse.
  검증: `set(comp.keys()) == set(CONFIG_KEYS)` (테스트에서 자동 확인).
- `HPS_APPLY_KEYS` = 탭 10이 "최적 설정 적용"으로 덮어쓰는 12개 위젯.
- **⚠️ `ordered_inputs = [comp[k] for k in CONFIG_KEYS]` 는 반드시 모든 탭(0~10)의 `comp[...]` 위젯이
  전부 만들어진 뒤, 즉 `build_ui()` 맨 끝(탭 10 블록이 끝난 직후, `return demo` 전)에서만 조립된다.**
  `ordered_inputs`(또는 `HPS_APPLY_KEYS` 같은 하위집합이 아닌 전체)를 쓰는 `.click()`/`.change()` 배선도
  전부 그 지점에 모아서 한다 — **위젯 "정의"는 각자의 `with gr.Tab(...)` 블록 안에 그대로 두고,
  이벤트 "배선"만 파일 끝으로 옮긴다** (Gradio는 위젯 인스턴스만 있으면 되고, `.click()` 호출이 그
  위젯의 레이아웃 블록 안에 있을 필요는 없음). 탭 8/9에 새 `comp[...]` 필드를 추가했다가 `ordered_inputs`
  조립 지점(예전엔 탭 5 직후)보다 뒤에 있어서 `KeyError`가 난 적이 있다 — 이후 이 지점을 파일 끝으로
  옮겨 해결했다. **새 CONFIG_KEYS 위젯을 탭 6 이후에 추가할 때 이 구조를 유지할 것.**

---

## 6. 작업 방식 규칙 (이 프로젝트에서 반드시 지킬 것)

1. **브랜치**: `claude/nice-ptolemy-5UQLR`에서만. 푸시는 `git push -u origin <branch>`, 네트워크 실패 시
   지수 백오프 재시도(2/4/8/16초). **PR은 명시적 요청 없이 만들지 않음.**
2. **옵트인 원칙**: 새 기능은 기본값(0/off)에서 **원본 CUT과 동작이 완전히 동일**해야 함.
3. **검증 필수**: 커밋 전 반드시 회귀 테스트 실행(7절). 수치 손실은 **합성 데이터로 물리적 타당성까지**
   검증(예: 선명한 사각형 < 블러 < 노이즈 순으로 손실 증가). 감으로 넘어가지 말 것.
4. **BUILD 마커**: 의미 있는 변경 시 `gui.py`의 `BUILD` 갱신(`날짜.순번 (요약)`).
5. **문서화**: 새 기능은 `docs/`에 한국어 설계 문서. 기존 문서 톤(개념 설명 + 코드 근거 + 주의사항) 유지.
6. **방어적 import**: `evaluation/`, `preprocessing/`의 `__init__.py` 패턴 유지(선택적 의존성 cv2/torch).
7. **한국어 소통 + 비개발자 배려**. 손실값이 아니라 FID/EPI로 품질 판단하도록 안내.
8. **모델 정체성**: 커밋 메시지/PR/코드에 모델 ID를 넣지 않음(채팅 답변에만).

---

## 7. 검증 방법 (커밋 전 실행)

```bash
cd /home/user/CUT_TEST_REPO
python -m py_compile preprocessing/*.py evaluation/*.py gui.py models/*.py options/*.py tests/*.py util/*.py
python tests/test_attention_port.py       # attention/hrnet/reflector/coherence + PatchSampleF 가중샘플링
python tests/test_preprocessing.py        # 전처리 파이프라인
python tests/test_param_optimize.py       # 전처리 파라미터 좌표하강(PARAM_SPACE/자동연결/단조개선/재개) (numpy만)
python tests/test_evaluation.py           # FID/KID/EPI/SSIM + 체크포인트 round-trip + CSV
python tests/test_subprocess_watchdog.py  # 공용 stall watchdog primitive + run_inference 통합 (~수초)
python tests/test_hparam_search.py        # canonicalize/재개/그리드/top_k/hang-중에도-탐색계속/소형 end-to-end (torch 필요, ~1분)
python tests/test_training_watchdog.py    # hang감지/재시작/최대횟수/Stop (가짜 trainer, ~수초)
python tests/test_loss_plot.py            # 손실 로그 파싱/D 합성/CSV·PNG
python tests/test_config_persistence.py   # 경로 영속성/체크포인트 스냅샷/탭별 폴더 필드 (torch+gradio 필요)
python tests/test_rectify.py              # 폴더 전체 처리/opencv 부재 fail-fast/부분실패 보고 (cv2 필요)
python tests/test_target_enhance.py       # CFAR(sigma/ca)+saliency 검출, 강조 3종 weight_map 게이팅, guided filter 엣지보존, 폴더 배치 (numpy만, clahe 방법만 cv2)
python -c "import gui; gui.build_ui()"    # GUI 빌드(위젯/CONFIG_KEYS 정합성)
```

⚠️ `test_hparam_search.py`는 실제 tiny 학습을 돌려 시간이 걸린다(hang 테스트 포함 시 ~1분). 전체를 한
명령으로 묶으면 2분 타임아웃에 걸릴 수 있으니 개별 실행 권장.

**테스트 환경 참고**: cv2/torch/scipy가 없으면 `pip install opencv-python-headless scipy` 필요.
InceptionV3 가중치는 이 샌드박스에서 다운로드가 프록시에 막히므로 FID는 로컬 가중치로만 검증됨
(구조/identity/품질 지표는 무관하게 동작).

---

## 8. 참고 문서 (docs/)

| 문서 | 내용 |
|---|---|
| `TARGET_ENHANCEMENT_SPEC.md` | 표적 시각적 강조 연구 명세(A/B 구현완료, C/D 미구현) — 문헌조사·설계 근거 |
| `TARGET_ENHANCE.md` | (구현) CFAR+saliency 검출 + saliency 가중 국소 강조 — 사용법/API |
| `CONFIG_PERSISTENCE.md` | 설정 자동저장(절대경로) + 체크포인트별 하이퍼파라미터 로그/복원 + 아키텍처 불일치 경고 |
| `LOSS_CURVES_AND_ATTENTION.md` | epoch별 D/G/NCE 손실 그래프 + 하이브리드/Self/ECA attention |
| `SMALL_OBJECT_PRESERVATION.md` | reflector 가중/saliency/coherence + 직사각 후처리, 물리적 근거·한계 |
| `EVALUATION.md` | CUT 출력 평가 4축 지표, 사용법 |
| `HPARAM_SEARCH.md` | (모델) Successive Halving 원리·공간·예산·주의 |
| `PARAM_OPTIMIZE.md` | (전처리) 파라미터 좌표하강 — 순서 확정 후 스텝별 파라미터 튜닝 |
| `RESILIENT_TRAINING.md` | hang 원인 진단 + watchdog + OS 설정 체크리스트 |
| `ATTENTION_PORT.md` | Attention/HRNet 통합 설계 |
| `OFFLINE_FID.md` | 오프라인 InceptionV3 가중치 배치 |
| `README_pipeline.md` | SAR 전처리 파이프라인 상세 설계 |
| `GUI.md` | Web-UI 개요 |
| `*_explained.html` | attention/patchnce/edge_color 손실 시각 설명(비개발자용) |

---

## 9. 다음에 할 수 있는 작업 (미완/후보 — 우선순위 순)

현재 **긴급한 미완 작업은 없음**(모든 커밋 완료·검증됨). 아래는 사용자와 논의된 향후 후보:

1. **좌표 매칭 짝 손실(`lambda_pixel`) — 논의 중, 미구현.**
   사용자가 **SAR↔Optical 좌표 매칭 데이터를 보유**. 현재 손실 설계상 `serial_batches`만으로는
   짝 정보가 **어떤 손실에도 안 쓰인다**(real_A↔real_B를 직접 비교하는 항이 없음 — 4.1 참조).
   → **fake_B ↔ 같은 좌표 real_B를 직접 비교하는 보조 손실**(L1/perceptual)을 신설하면 실제
   형상 정확도 향상 가능. **단 전제**: 데이터 정합 정밀도 확인 필요(픽셀 단위 정합이면 L1,
   정합 오차가 있으면 흐림 유발 → perceptual/구조 기반 + 작은 가중치 권장).
   **인수자 액션**: 사용자에게 정합 정밀도(픽셀/타일 단위)를 물어보고 그에 맞는 손실 형태로 설계.

2. **512 해상도 학습 가이드/프리셋** — 작은 물체 보존에 근본적 도움. RTX 5080 16GB에서 batch=1 가능.

3. **LoRA/사전학습 대형 모델 방향** — 사용자가 관심 표명. 단 현재 CUT(scratch 학습)엔 부적합하며,
   **CUT가 아니라 사전학습 EO 모델 + ControlNet류**로 아키텍처를 바꾸는 큰 결정. 별도 심층 논의 필요.

4. **전처리 강반사 보호 프리셋** — outlier_clipping 완화 + speckle 약하게, 순서탐색에서 EPI 높은 후보 선택.

---

## 10. 자주 하는 실수 / 함정 (인수자 주의)

- **`nce_idt`가 off면 `lambda_color`·`NCE_Y`가 무력화**된다. FastCUT 모드는 nce_idt 기본 off이므로 주의.
- **coherence_loss는 energy×coherence 둘 다 필요**(4.1). 한쪽만 쓰면 구름을 못 잡는다.
- **rectangularity 임계값은 0.785(π/4)보다 높아야** 원이 걸러진다.
- **evaluation/generate.py의 아키텍처 cfg는 체크포인트 학습 때와 정확히 일치**해야 로드됨
  (attention/hrnet 옵션 불일치 시 state_dict 로드 실패 → 명확한 에러 메시지 있음).
- **GUI 파라미터 추가 시 5단계 배선**(5절)을 하나라도 빠뜨리면 `build_ui()` 정합성 체크 실패.
- **watchdog의 stall 시간은 epoch 소요 시간보다 넉넉히** 줘야 오탐 없음(큰 데이터셋은 epoch이 20분 초과 가능).
- **경로 관련 기본값은 항상 절대경로**로(`os.path.join(REPO_ROOT, ...)`) — 상대경로는 실행 위치(CWD)에
  따라 조용히 다른 파일을 가리켜서 "저장한 게 사라졌다"는 증상으로 나타난다(4.8 참조, 실제 겪은 버그).
- **새 아키텍처 관련 CONFIG_KEYS를 추가하면 `ARCH_CRITICAL_KEYS`(gui.py)에도 추가**할 것 — 안 하면
  이어서 학습 시 그 옵션의 불일치가 감지되지 않는다.
- **matplotlib으로 그래프를 그릴 때 한글 텍스트를 쓰지 말 것** — 기본 폰트에 글리프가 없어 깨진다
  (4.6에서 실측으로 발견). 축/제목/범례는 영문으로.
- **세션 컨테이너는 ephemeral**: torch/torchvision/gradio/matplotlib/dominate/opencv가 재개 때마다
  없어질 수 있다. "설치 안 되어 있음"을 곧바로 "이 기능이 안 됨"으로 오판하지 말고 먼저 설치를 시도할 것.
- **여러 파일을 순회하는 루프에서 `except Exception: continue`를 절대 무음으로 쓰지 말 것.**
  opencv 미설치 같은 흔한 원인이 "폴더 전체가 조용히 실패 → 결과 0개인데 성공 메시지"로 둔갑한 실제
  버그가 있었다(4.3). 최소한 실패 개수+사유를 모아서 반환/로그하고, 가능하면 루프 진입 전에 공통
  실패 원인(의존성 부재 등)을 한 번만 검사해 fail-fast 할 것.
- **`ordered_inputs`(또는 그에 준하는 "전체 위젯 리스트")를 쓰는 이벤트 배선은 반드시 그 리스트가
  조립되는 지점(현재 `build_ui()` 맨 끝) 이후에 있어야 한다.** 탭 8/9에 새 CONFIG_KEYS 위젯을 추가했을
  때 `ordered_inputs`가 탭 5 직후에 조립되던 옛 구조와 충돌해 `KeyError`가 났었다(5절 참고) — 이제
  위젯 "정의"는 각 탭 블록에, 이벤트 "배선"은 파일 끝에 모아둔다.
- **행(hang) 보호가 필요한 새 서브프로세스 호출은 반드시 `util/subprocess_watchdog.py`의
  `run_watched`/`run_watched_stream`을 통해서만 만들 것.** 처음엔 학습(탭 6)에만 워치독을 만들었다가
  나중에 "탐색/추론에서도 hang이 생긴다"는 리포트를 받고서야 그 두 곳엔 아예 보호가 없었다는 걸
  발견했다(4.5) — `subprocess.Popen` + 순수 블로킹 `readline()` 패턴을 새로 작성하지 말 것.
- **stall 테스트에서 `stall_minutes`를 너무 타이트하게 잡지 말 것.** 진짜 `train.py`는 torch import 등으로
  **첫 진행 로그까지 실측 2초 이상** 걸린다 — 가짜 스크립트로만 테스트하면 이 사실을 놓치기 쉽고,
  실제 train.py를 끼워 쓰는 테스트(`test_hparam_search.py`의 hang 테스트처럼)에서 `stall_minutes`가
  이 실측값보다 작으면 **멀쩡한 trial까지 오탐으로 죽는다**. 최소 0.15~0.2분(9~12초) 이상 확보할 것.
- **`evaluation/__init__.py`가 서브모듈과 같은 이름의 함수를 재노출하면 `import evaluation.<모듈명> as x`가
  모듈이 아니라 그 함수를 가리킬 수 있다**(예: `hparam_search`). 모듈 자체를 확실히 잡으려면
  `sys.modules['evaluation.hparam_search']`처럼 접근할 것(테스트에서만 발생하는 문제지만 디버깅에 시간을
  뺏기기 쉽다).

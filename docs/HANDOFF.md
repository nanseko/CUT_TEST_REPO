# HANDOFF — SAR→Optical CUT 프로젝트 인수인계 하네스

> **목적**: 이 문서 하나로 다른 Claude(또는 개발자)가 이 프로젝트의 현재 상태·설계·미완 작업을
> 즉시 파악하고 바로 작업을 이어받을 수 있도록 한다. **작업을 시작하기 전에 이 문서를 처음부터 끝까지 읽을 것.**

- **작성일**: 2026-07-08
- **대상 저장소**: `nanseko/cut_test_repo`
- **작업 브랜치**: `claude/nice-ptolemy-5UQLR` (⚠️ **이 브랜치에서만 개발·푸시**. 다른 브랜치 푸시 금지)
- **현재 BUILD 마커**: `gui.py` 안 `BUILD = '2026-07-08.1 (loss-curves+hybrid-self-eca-attention)'`
- **가장 최근 커밋 계열**: 손실 그래프(`util/loss_plot.py`) + 추가 attention(self/eca/cbam_coord)

> ⚠️ **환경 주의**: 세션 컨테이너는 ephemeral이라 재개 시 torch/torchvision/matplotlib/dominate 등이
> 사라질 수 있다. 검증 전 `pip install torch torchvision matplotlib dominate` 필요할 수 있음
> (기본 pip 인덱스는 프록시 통과, `download.pytorch.org`는 403). visdom/GPUtil은 빌드 실패해도
> `display_id=0` CPU 학습에는 불필요.

---

## 0. 30초 요약 (TL;DR)

이 프로젝트는 **공식 CUT(Contrastive Unpaired Translation, PyTorch)** 를 포크해서
**SAR(위성 레이더) → Optical(광학) 영상 변환**에 특화시킨 것이다. 핵심 사용자는
**비개발자**이며, 거의 모든 조작을 **Gradio Web-UI(`gui.py`, 11개 탭)** 로 수행한다.

이번(그리고 이전) 세션에서 추가한 것은 크게 6덩어리:
1. **Attention**(CBAM/Coordinate) + **HRNet** 생성기 (블러/작은물체 대응)
2. **SAR 전처리 파이프라인** + **전처리 순서 자동 탐색**
3. **모델 출력 평가 모듈**(`evaluation/`) — FID/KID/EPI/SSIM
4. **작은 물체(요트·탱크·건물) 형상 보존** — reflector 가중 손실 + saliency 샘플링 + coherence 손실 + 직사각 후처리
5. **하이퍼파라미터 자동 탐색**(Successive Halving)
6. **학습 중단(hang) 자동 복구 watchdog**

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

### 4.4 하이퍼파라미터 자동 탐색 (`evaluation/hparam_search.py`, 탭 10)

- **온라인 자가 최적화는 의도적으로 안 만듦**(GAN 손실 = 품질 아님). 대신 **짧은 학습 N개 →
  FID/EPI 랭킹 → 상위 K개 이어학습**(Successive Halving).
- `canonicalize()`가 동등 설정을 중복 제거(attention=none일 때 위치/reduction 무시 등).
- 재개형: `hparam_results.csv`에 (trial, stage)별 기록, 재실행 시 건너뜀.
- 학습/추론 명령은 **주입된 `build_train_cmd`/`build_test_cmd`**(gui.py 것)를 그대로 사용 →
  gui.py와 항상 동일한 명령 보장, hparam_search.py는 gradio 의존성 없음.
- 탐색 공간: attention(type/위치/reduction) + 손실 가중치(grad/lap/coherence/color, boost, 옵션들).

### 4.5 학습 hang watchdog (`gui.py::training_worker`, 탭 6)

- **아키텍처 핵심**: 학습은 이미 **데몬 스레드 + 서브프로세스**로 돌아 브라우저 연결과 무관.
  gui.py 프로세스 자체만 살아있으면 학습 지속. (사용자에게 이 사실을 명확히 전달했음)
- watchdog: 배경 스레드+`queue.Queue`로 stdout을 read-with-timeout →
  **N분(기본 20) 진행 로그 없으면** 프로세스 kill 후 **`--continue_train`으로 자동 재시작**.
  최대 재시작 횟수(기본 20)까지. 비정상 종료(exit≠0)도 동일 처리. Stop은 backoff 중에도 즉시.
- `MIN_STALL_SECONDS=60` 하한(오탐 방지, 테스트에서 monkeypatch로 낮춤).
- **코드로 못 고치는 OS 원인**은 `docs/RESILIENT_TRAINING.md`에 정리(절전 모드, Windows Update
  재부팅, 네트워크 드라이브 I/O, num_threads, 백신 예외).

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
| 6 | 학습 실행/모니터링 | `start_training`/`stop_training`/`training_worker` |
| 7 | 추론/테스트 | `run_inference` |
| 8 | 모델 평가 (CUT 출력) | `cut_evaluate`/`eval_table_rows` |
| 9 | 형상 후처리 (직사각 스냅) | `cut_rectify` |
| 10 | 하이퍼파라미터 자동 탐색 | `cut_hparam_search`/`hps_apply_best` |

**핵심 배선 규칙**:
- `CONFIG_KEYS`(gui.py 상단, 48개) = 모든 탭 위젯의 순서. `ordered_inputs`로 콜백에 전달.
- 새 파라미터 추가 시 반드시: ① `CONFIG_KEYS`에 추가 ② `DEFAULTS`에 기본값
  ③ `build_train_cmd`에 CLI 인자 ④ 탭 UI에 `comp['key']=...` 위젯 ⑤ (해당되면) options 파일에 argparse.
  검증: `set(comp.keys()) == set(CONFIG_KEYS)` (테스트에서 자동 확인).
- `HPS_APPLY_KEYS` = 탭 10이 "최적 설정 적용"으로 덮어쓰는 12개 위젯.

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
python -m py_compile preprocessing/*.py evaluation/*.py gui.py models/*.py options/*.py tests/*.py
python tests/test_attention_port.py       # attention/hrnet/reflector/coherence + PatchSampleF 가중샘플링
python tests/test_preprocessing.py        # 전처리 파이프라인
python tests/test_evaluation.py           # FID/KID/EPI/SSIM + 체크포인트 round-trip + CSV
python tests/test_hparam_search.py        # canonicalize/재개/소형 end-to-end (torch 필요, ~수십초)
python tests/test_training_watchdog.py    # hang감지/재시작/최대횟수/Stop (가짜 trainer, ~수초)
python -c "import gui; gui.build_ui()"    # GUI 빌드(위젯/CONFIG_KEYS 정합성)
```

⚠️ `test_hparam_search.py`는 실제 tiny 학습을 돌려 시간이 걸린다. 전체를 한 명령으로 묶으면
2분 타임아웃에 걸릴 수 있으니 개별 실행 권장.

**테스트 환경 참고**: cv2/torch/scipy가 없으면 `pip install opencv-python-headless scipy` 필요.
InceptionV3 가중치는 이 샌드박스에서 다운로드가 프록시에 막히므로 FID는 로컬 가중치로만 검증됨
(구조/identity/품질 지표는 무관하게 동작).

---

## 8. 참고 문서 (docs/)

| 문서 | 내용 |
|---|---|
| `SMALL_OBJECT_PRESERVATION.md` | reflector 가중/saliency/coherence + 직사각 후처리, 물리적 근거·한계 |
| `EVALUATION.md` | CUT 출력 평가 4축 지표, 사용법 |
| `HPARAM_SEARCH.md` | Successive Halving 원리·공간·예산·주의 |
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

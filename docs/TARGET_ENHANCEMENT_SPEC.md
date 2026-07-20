# 표적(건물·차량·비행체) 시각 식별성 강화 후처리 — 연구/설계 명세서

> **목적**: SAR→Optical 변환 결과(`fake_B`)에서 **건물·차량·비행체 같은 표적이 사람 눈에 더 뚜렷하게
> 식별되도록** 하는 후처리 방법을 최근 SAR·객체탐지 논문 기반으로 조사하고, 현재 코드 설계에 어떻게
> 얹을지 명세한다.
> **범위**: 이 문서는 **설계 명세서**이며 코드 변경을 포함하지 않는다. 구현은 승인 후 별도 진행한다.
> **작성일**: 2026-07-08
>
> **구현 현황 (2026-07-20 갱신)**: **A(CFAR+saliency 검출)+B(saliency 가중 국소 강조) 구현 완료**
> (`evaluation/target_enhance.py`, GUI 탭 9 하단, `docs/TARGET_ENHANCE.md` 참고). **C(클래스별 형상/주석)
> 와 D(표적 초해상)는 미구현** — 요청 시 별도 진행.

---

## 0. 요약 (TL;DR)

현재 후처리(`evaluation/rectify.py`)는 **직사각형 스냅**(Otsu 임계 → 윤곽 → `minAreaRect`) **한 가지**뿐이다.
"표적을 눈으로 더 잘 식별"하려는 목적에는 다음 **3개 축**을 조합하는 것이 최근 문헌의 주류다:

1. **표적 검출 강화** — 순진한 Otsu 대신 **CFAR**(SAR 표준 적응 임계) 로 표적 후보를 더 견고하게 찾는다.
2. **표적 영역 국소 강조** — **saliency-guided 국소 대비/선명도 향상**으로 배경 스페클은 안 키우고
   표적만 또렷하게 한다(unsharp/guided filter/CLAHE를 표적 마스크 안에서만).
3. **주석·형상 오버레이** — 회전 바운딩박스 + **클래스별 형상 템플릿**(비행체=십자, 차량=소형 직사각,
   건물=대형 직사각) + 신뢰도 표시로 "무엇인지"를 명시.

**권장 우선순위**: ①국소 강조(가장 범용적·저위험) → ②CFAR 검출 → ③클래스별 형상/주석 → (선택)④표적
초해상. 모두 **학습 재실행 없이** `fake_B`에 후처리로 적용 가능하다.

---

## 1. 현재 상태 진단

### 1.1 지금 있는 것
- `evaluation/rectify.py`: 윤곽 검출 → 최소외접회전사각형/다각형 스냅. **기하 정확성**(직각 보장)이 목적이라
  "벡터 도형처럼" 보이며, **표적을 강조**하기보다 **형상을 요약**한다.
- `models/losses_extra.py`: `reflector_saliency_map`(SAR 국소 밝기 피크 = 강반사체 후보), `edge_sharpness_map`
  (구조텐서 energy×coherence), `coherence_loss` — **학습 시점**의 표적 보존 도구.

### 1.2 공백 (이 명세가 채우는 것)
- **표적 검출이 Otsu 단일 임계** → 복잡한 배경(도심·해안)에서 오검출/미검출.
- **강조(enhancement) 후처리 없음** → 표적이 흐릿해도 그대로 둔다.
- **클래스 구분 없음** → 비행체/차량/건물이 전부 "직사각형"으로만 처리(비행체는 직사각형이 아님).
- **신뢰도 개념 없음** → 흐릿한 허상 블롭도 실제 표적과 똑같이 표시(오판 위험).

---

## 2. 문헌 조사 요약 (2024–2025 중심)

### 2.1 SAR 표적을 "돋보이게" 하는 접근의 큰 갈래

| 갈래 | 핵심 아이디어 | 후처리 적용성 | 대표 근거 |
|---|---|---|---|
| **Saliency 기반 검출/강조** | 표적은 배경 대비 국소적으로 튀는 영역 → saliency map으로 검출·강조 | ★★★ (마스크만 있으면 순수 이미지 후처리) | Bayesian saliency SAR detection, 사르 saliency change detection |
| **CFAR 적응 검출** | 국소 클러터 통계로 적응 임계 → 오경보율 일정 유지 | ★★★ (임계 계산만, 학습 불필요) | RmSAT-CFAR, joint CFAR, "가장 널리 쓰이는 SAR 검출법" |
| **Confidence-guided 생성 신뢰도** | 표적 영역의 생성 신뢰도를 추정해 신뢰 낮은 곳을 구분/보정 | ★★ (신뢰도 맵을 후처리 플래그로) | C-DiffSET (confidence-guided reliable object generation) |
| **Downstream-task 유도** | 탐지/분할 같은 하류 과제로 변환을 유도해 표적을 또렷하게 | ★ (학습 재실행 필요, 후처리 아님) | Seg-CycleGAN |
| **표적 영역 초해상(SR)** | 표적 크롭을 업스케일해 디테일 복원 | ★★ (검출된 크롭에만 SR) | SAR SR 계열, CMAR-Net(3D 재구성) |
| **회전 바운딩박스·클래스 형상** | 차량 등은 방향성 있는 회전 박스로 표기(축정렬보다 정확) | ★★★ (검출 후 표기) | SIVED(rotatable bbox vehicle), Strip R-CNN |
| **국소 대비 향상(고전 영상처리)** | unsharp/guided filter/CLAHE를 표적에만 적용 | ★★★ (순수 후처리) | adaptive unsharp masking, saliency-guided detail enhancement, guided filter |

### 2.2 핵심 통찰 (이 프로젝트에 직접 시사하는 것)
- **"전역 향상"은 스페클까지 키운다.** 모든 논문이 공통적으로 **표적 마스크 안에서만** 향상을 적용한다
  (saliency-guided). 우리 프로젝트에 이미 있는 `reflector_saliency_map` 개념이 그 마스크 역할을 할 수 있다.
- **CFAR가 SAR 검출의 사실상 표준.** 단순 임계(Otsu)보다 배경 통계에 적응해 도심/해안 클러터에서 훨씬
  견고하다. 우리의 `detect_candidate_regions`를 대체/보강할 1순위.
- **클래스마다 형상이 다르다.** 비행체는 직사각형이 아니라 십자/T. 현재 rectify의 "직사각형도(rectangularity)
  필터"는 비행체를 오히려 **탈락**시킨다 → 클래스별 형상 모델 필요.
- **신뢰도 표시가 오판을 막는다.** SAR→EO는 없는 것을 지어낼 수 있으므로(hallucination), 흐릿한 저신뢰
  영역을 **다른 색/점선**으로 표기해 분석자가 신뢰 수준을 알게 하는 것이 실무적으로 중요.

---

## 3. 제안 방법 (후처리 파이프라인)

전체를 **검출 → 강조 → 주석**의 3단 파이프라인으로 설계한다. 각 단계는 독립 토글이 가능해야 한다.

```
fake_B ─▶ [A. 표적 검출: CFAR + saliency] ─▶ 표적 마스크/후보영역
                                              │
        ┌─────────────────────────────────────┤
        ▼                                     ▼
[B. 표적 영역 국소 강조]               [C. 클래스별 형상·주석 오버레이]
 unsharp/guided/CLAHE (마스크 내부만)    회전 bbox + 형상 템플릿 + 신뢰도
        │                                     │
        └──────────────▶ 강조된 fake_B ◀──────┘  (+ 오버레이 버전 별도 저장)
```

### A. 표적 검출 강화 (Otsu → CFAR + saliency)

**목적**: 표적 후보 영역/마스크를 견고하게 얻는다(이후 B·C의 입력).

- **A1. CFAR 검출기** (`cell-averaging CFAR`, 순수 NumPy 구현 가능):
  각 픽셀에 대해 주변 링(guard+training) 통계로 국소 임계를 계산 → 임계 초과 픽셀을 표적 후보로.
  - 배경이 균질하지 않은 도심/해안에서 Otsu보다 오검출↓. `false_alarm_rate` 파라미터로 민감도 제어.
  - opencv 불필요(연산은 박스필터/적분영상으로 NumPy 구현; 이미 `preprocessing`에 `_box_filter` 있음).
- **A2. Saliency 결합**: `reflector_saliency_map`(밝기 피크)과 `edge_sharpness_map`(구조텐서)을 결합해
  "표적스러움" 점수를 만들고, CFAR 마스크와 AND/가중 → 노이즈성 단발 픽셀 억제.
- **A3. 형태학적 정리**: open/close로 잡음 제거, 연결요소 라벨링으로 후보 영역 분리(면적 필터는 기존 유지).

**산출**: 표적별 (마스크, 회전 bbox, saliency 점수, 면적).

### B. 표적 영역 국소 강조 (핵심 — 시각 식별성 직접 향상)

**목적**: **표적만** 또렷하게, 배경 스페클은 안 키운다. 문헌의 saliency-guided enhancement를 그대로 적용.

- **B1. 국소 대비 향상 — 마스크 내부 CLAHE**: 표적 영역에만 CLAHE(적응 히스토그램 평활)로 대비를 올린다.
  전역 CLAHE는 배경 노이즈를 키우므로 반드시 **마스크 가중**.
- **B2. 적응 unsharp masking**: `enhanced = img + amount * (img - blur(img))` 를 **표적 마스크로 가중**해
  적용(표적=강하게, 배경=약하게/무). adaptive unsharp masking 논문의 "디테일 영역만 선명화" 원리.
- **B3. Guided filter 디테일 부스트**: 이미지를 base+detail로 분해 → detail을 표적 마스크로 가중 증폭 →
  재합성. 에지 보존하며 표적 디테일만 강조(saliency-guided detail enhancement).
- **혼합 규칙**: 최종 픽셀 = `saliency_weight * enhanced + (1-saliency_weight) * original`.
  saliency_weight는 A의 표적 점수 → **부드러운 경계**로 인공 이음매 방지.

**설계 주의**: 강조 강도(`amount`, `clip_limit`)를 너무 키우면 **없던 텍스처를 만들어** 오히려 허상처럼
보인다. 기본값은 보수적으로(예: unsharp amount 0.3~0.8), 파라미터 노출.

### C. 클래스별 형상·주석 오버레이 (무엇인지 명시)

**목적**: 검출된 표적을 **클래스에 맞는 형상**과 **주석**으로 표기해 "이것은 차량/건물/비행체"임을 전달.
(원본 강조 이미지와 **별도 오버레이 레이어**로 저장 — 분석자가 켜고 끌 수 있게.)

- **C1. 회전 바운딩박스**: 현재 `minAreaRect`(이미 있음) 활용, 차량처럼 방향성 있는 표적에 축정렬보다 적합
  (SIVED rotatable bbox 근거).
- **C2. 클래스별 형상 판별(규칙 기반, 학습 불필요)**: 면적·종횡비·직사각형도·대칭성으로 대략 분류:
  - **건물**: 큰 면적 + 높은 직사각형도 + 낮은 종횡비 극단 → 대형 직사각.
  - **차량**: 작은 면적 + 중간 직사각형도 + 방향성 → 소형 회전 직사각.
  - **비행체**: **십자/T 형 대칭**(주축+직교 날개) → 직사각형도는 낮지만 대칭축이 2개. 현재의 단순
    "직사각형도 필터"로는 못 잡으므로 **대칭성/골격(skeleton) 기반 판별**을 추가.
- **C3. 신뢰도 표시(C-DiffSET 착안)**: 표적별 신뢰도 = saliency 점수 × edge_sharpness(또렷함).
  - 높음 → 실선 + 클래스 라벨. 낮음(흐릿한 블롭) → **점선/경고색** = "저신뢰(추정)".
  - 흐릿한 허상을 실제 표적처럼 오판하는 것을 방지(SAR→EO의 hallucination 실무 리스크 대응).
- **C4. 주석 정보**: 클래스, 크기(px, 가능하면 GSD 있으면 m), 방위각(회전각), 신뢰도를 박스 옆/CSV에 표기.

### D. (선택) 표적 영역 초해상

**목적**: 표적 크롭만 업스케일해 디테일 복원. 비용이 크고 별도 SR 모델이 필요하므로 **후순위**.
- 검출된 각 표적 크롭을 SR(예: 경량 ESRGAN/실시간 SR) → 원위치 합성. 배경은 건드리지 않음.
- **주의**: SR도 없는 디테일을 생성할 수 있어(초해상 hallucination), 신뢰도 표시(C3)와 함께 써야 함.

---

## 4. 현재 설계에서 무엇을 어떻게 바꾸나

### 4.1 코드 배치 (제안)
- **`evaluation/target_enhance.py` (신규)**: A(CFAR/saliency 검출) + B(국소 강조)의 순수 이미지 연산.
  numpy 우선, opencv는 선택적(현재 `rectify.py`의 `_require_cv2` 패턴 재사용). CLAHE만 cv2 필요.
- **`evaluation/rectify.py` (확장)**: C(형상·주석). 이미 `minAreaRect`/`rectify_regions`가 있으므로
  클래스 판별(C2)·신뢰도(C3)만 추가. 기존 함수 시그니처는 하위호환 유지.
- **재사용**: `models/losses_extra.py`의 `reflector_saliency_map`/`edge_sharpness_map`을 **학습이 아니라
  후처리에서도** 호출(현재 순수 torch지만, numpy 포팅 또는 torch 유무에 따라 분기). `preprocessing`의
  `_box_filter`(적분영상)로 CFAR/guided filter를 opencv 없이 구현.

### 4.2 GUI (탭 9 "형상 후처리" 확장)
- 현재 탭 9는 rectify(형상 스냅) 단일 기능. 여기에 **강조 모드 토글**(A/B/C 개별 on/off) + 파라미터
  (CFAR 민감도, unsharp amount, CLAHE clip_limit, 신뢰도 임계) 추가.
- 입력 폴더 지정·폴더 일괄 처리·결과 CSV·오버레이 갤러리 등 **기존 rectify_folder 인프라 재사용**
  (opencv 부재 시 fail-fast, 파일별 실패 집계 — 이미 구현된 패턴 그대로).
- **출력 3종**: (1) 강조된 이미지, (2) 주석 오버레이, (3) 표적 CSV(클래스·bbox·신뢰도).

### 4.3 하이퍼파라미터 자동 최적화와의 연결 (선택)
- 강조 파라미터(unsharp amount 등)도 **정량 지표로 튜닝** 가능. 단, "시각 식별성"은 무참조 지표가
  까다로움 → 표적 영역의 **국소 대비/선명도(edge_sharpness) 증가량**과 **배경 노이즈 증가 억제**를
  동시에 보는 복합 지표를 제안. (모델 하이퍼파라미터 탐색 인프라와는 별개, 저비용 그리드로 충분.)

### 4.4 하위호환·안전
- 모든 강조는 **옵트인**(기본 off면 기존 rectify 동작과 동일).
- **허상 경고 원칙**: 강조·SR은 "없는 디테일 생성" 위험이 있으므로, C3 신뢰도 표시를 **항상 함께** 노출.
  문서에 "강조는 판독 보조이지 새로운 사실이 아니다"를 명시.

---

## 5. 권장 실행 순서 (구현 우선순위)

| 순위 | 항목 | 이유 | 난이도 | 의존성 | 상태 |
|---|---|---|---|---|---|
| 1 | **B. 표적 영역 국소 강조**(unsharp+CLAHE, saliency 마스크) | 가장 범용적, 즉시 체감, 저위험 | 중 | 기존 saliency 재사용 | ✅ 구현 완료 |
| 2 | **A. CFAR 검출** | 표적 마스크 품질↑ → B/C 전부 개선 | 중 | numpy만 | ✅ 구현 완료 |
| 3 | **C. 클래스별 형상+신뢰도 주석** | "무엇인지" 명시, 비행체 대응 | 중상 | rectify 확장 | ⬜ 미구현 |
| 4 | (선택) **D. 표적 초해상** | 디테일 복원, 단 비용·허상 위험 | 상 | 별도 SR 모델 | ⬜ 미구현 |

**1차 목표(MVP)**: A(CFAR)+B(국소 강조) 만으로도 "표적이 눈에 더 잘 들어오는" 효과를 낼 수 있고,
학습 재실행이 전혀 없다. C는 판독 UI로서 그 다음.

**구현 완료** (`evaluation/target_enhance.py`, GUI 탭 9, `docs/TARGET_ENHANCE.md`): A는 sigma/CA 두 CFAR
변형 + brightness saliency 교차검증 + 형태학 정리 + 연결요소·면적 필터로 표적 후보를 검출하고, B는 그
saliency 맵을 픽셀별 블렌딩 가중치로 써서 unsharp masking / guided-filter 디테일 증폭 / masked CLAHE 중
선택한 방법(들)을 표적 영역에만 적용한다. C/D는 이 구현에 포함되지 않았다.

---

## 6. 평가 방법 (효과 검증)

"시각 식별성"은 주관적이므로 **정량 + 정성**을 병행:
- **정량(무참조)**: 표적 마스크 내부의 국소 대비(RMS contrast)·선명도(edge_sharpness/gradient) **증가량**,
  대비 배경 영역 노이즈 증가율(작아야 함). "표적↑ / 배경≈유지" 를 확인.
- **정성**: 강조 전/후 나란히 비교(현재 `SendUserFile`/갤러리로 제시), 저신뢰 표기가 흐릿한 블롭에 제대로
  붙는지 육안 확인.
- **하류 과제(있다면)**: 탐지/판독 정확도 변화 — 가장 확실하지만 라벨 필요.

---

## 7. 리스크와 한계

- **허상 증폭**: 강조·SR은 없는 디테일을 만들 수 있다 → 신뢰도 표시(C3) 필수, 강조 강도 보수적 기본값.
- **클래스 판별의 한계**: 규칙 기반(면적/종횡비/대칭)은 대략적. 정밀 분류는 별도 탐지 모델(YOLO/oriented
  R-CNN)이 필요하며, 이는 후처리 범위를 넘어 학습 파이프라인 추가에 해당(향후 과제).
- **GSD 미상**: 실제 크기(m) 표기는 지상표본거리(GSD)를 알아야 정확. 없으면 픽셀 단위로만.
- **cv2 의존**: CLAHE 등 일부는 opencv 필요 — 기존 fail-fast 패턴으로 명확히 안내.

---

## 8. 참고 문헌 (조사 출처)

- C-DiffSET: Latent Diffusion for SAR-to-EO Translation with **Confidence-Guided Reliable Object Generation** — https://arxiv.org/abs/2411.10788
- Seg-CycleGAN: **Downstream-task guided** SAR-to-optical translation — https://arxiv.org/html/2408.05777v1
- Object Detection in Single SAR Images via a **Saliency Framework (Bayesian Inference + Adaptive Iteration)** — https://www.mdpi.com/2072-4292/17/17/2939
- Saliency-Based SAR Target Detection via Convolutional Sparse Feature Enhancement and Bayesian Inference — https://www.researchgate.net/publication/367196152
- **Light-weight SAR saliency enhancement** (sea–land segmentation preference) — https://doi.org/10.3390/rs17050795
- RmSAT-CFAR: Fast and accurate target detection in radar images (**CFAR**) — https://www.sciencedirect.com/science/article/pii/S235271101730047X
- A Correlation-Based Joint **CFAR** Detector Using Adaptively-Truncated Statistics in SAR — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5419799/
- **SIVED**: SAR Image Dataset for Vehicle Detection Based on **Rotatable Bounding Box** — https://doi.org/10.3390/rs15112825
- Vehicle Localization in Complex SAR Images via **Feature Reconstruction and Aggregation** — https://pmc.ncbi.nlm.nih.gov/articles/PMC11510714/
- **Adaptive Unsharp Masking** for image enhancement — https://www.researchgate.net/publication/5597643
- **Saliency-Guided** Image Detail Enhancement — https://ieeexplore.ieee.org/document/8732250/
- Visual saliency detection based on region contrast and **guided filter** — https://ieeexplore.ieee.org/document/8167232
- Fifty Years of Object Detection from SAR Remote Sensing: The Road Forward (2025 survey) — https://arxiv.org/pdf/2509.22159
- M4-SAR: Multi-source Dataset/Benchmark for optical-SAR Object Detection — https://arxiv.org/pdf/2505.10931
- Physics-guided interpretable CNN for SAR target recognition — https://www.sciencedirect.com/science/article/pii/S1000936124003960
- CMAR-Net: Cross-modal 3D-SAR reconstruction of vehicle targets — https://arxiv.org/html/2406.04158v6

> **주의**: 위 논문 대다수는 **학습 기반 탐지/변환**이 주제다. 이 명세는 그 아이디어(saliency 마스크,
> CFAR, 신뢰도, 클래스 형상, 국소 강조) 중 **후처리로 이식 가능한 부분만** 추려 현재 코드 위에 얹는
> 방향으로 재구성한 것이다. 학습 기반 정밀 탐지는 별도 과제(§7)로 남긴다.

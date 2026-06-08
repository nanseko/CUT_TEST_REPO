# M4-SAR 기반 SAR 전처리 파이프라인 설계 문서

> 대상: `nanseko/nanse_test_repo`의 CUT / FastCUT 학습 환경에 연결할 SAR 전처리 파이프라인  
> 목적: M4-SAR 공개데이터 또는 사용자가 지정한 SAR 폴더를 Web UI에서 선택하고, 모듈형 전처리 순서를 자유롭게 구성한 뒤 CUT 학습 입력 폴더로 내보내기  
> 산출물: 전처리된 SAR 이미지, 설정 파일, 처리 manifest, 미리보기 이미지, 로그

---

## 1. 배경 및 목표

M4-SAR는 optical-SAR 융합 객체 탐지용으로 공개된 대규모 SAR/Optical 데이터셋이다. 본 프로젝트에서는 이 데이터를 그대로 객체 탐지에 쓰기보다는, SAR 이미지를 optical-like 도메인으로 변환하는 CUT 기반 학습의 입력으로 활용한다.

CUT는 paired image-to-image translation이 아니라 unpaired translation 구조이므로, SAR와 Optical 이미지가 반드시 1:1로 정렬되어 있을 필요는 없다. 다만 CUT 학습에는 일반적으로 source domain A와 target domain B가 필요하다. 따라서 본 파이프라인은 다음 두 가지 목적을 동시에 지원한다.

1. SAR 이미지를 CUT의 source domain `trainA/testA`에 넣기 좋은 형태로 정규화한다.
2. SAR-only 상황에서도 optical-like 톤을 갖는 히스토그램 매핑 결과를 만들 수 있게 한다.

주의할 점은, SAR-only 히스토그램 매핑은 실제 optical semantic 정보를 새로 생성하는 것이 아니라 SAR intensity 분포를 optical-like tone curve로 변환하는 전처리다. 진짜 optical translation 품질을 얻으려면 unpaired라도 target optical domain `trainB/testB`를 함께 사용하는 것이 권장된다.

---

## 2. 요구사항 요약

| 요구사항 | 설계 반영 |
|---|---|
| 입력 폴더 지정 | Web UI와 CLI 모두 `input_dir` 입력 |
| 출력 폴더 지정 | Web UI와 CLI 모두 `output_dir` 입력 |
| 처리 데이터 개수 지정 | `max_items`로 폴더 내 앞 N개 또는 샘플링 N개 처리 |
| 전처리 모듈화 | 모든 전처리를 `PreprocessStep` 인터페이스로 구현 |
| 전처리 순서 변경 | `pipeline.steps` 배열 순서를 바꾸면 실행 순서 변경 |
| 전처리 추가/삭제 | `enabled: true/false`, step registry로 추가/제거 |
| Speckle filter 선택 | Lee, Frost, BM3D, Refined Lee, Gamma-MAP 중 Web UI에서 1개 선택 |
| Outlier clipping | min/max percentile 기반 clipping |
| Histogram mapping | paired SAR-optical 없이도 SAR-only 모드 지원 |
| CUT repo 연결 | 결과를 `datasets/M4-SAR-cut/{trainA,trainB,testA,testB}` 형태로 export |

---

## 3. 권장 전체 구조

기존 CUT repo에 다음 디렉터리와 파일을 추가하는 것을 권장한다.

```text
nanse_test_repo/
├── gui.py                         # 기존 CUT 학습 Web UI
├── train.py
├── inference.py
├── modules/
│   └── ...
├── preprocessing/
│   ├── __init__.py
│   ├── pipeline.py                # Pipeline runner
│   ├── registry.py                # step registry
│   ├── io.py                      # image load/save, folder scan
│   ├── config.py                  # YAML/JSON config load/save
│   ├── manifest.py                # processed file manifest 기록
│   └── steps/
│       ├── __init__.py
│       ├── base.py                # PreprocessStep base class
│       ├── speckle.py             # Lee/Frost/BM3D/RefinedLee/GammaMAP
│       ├── clipping.py            # percentile clipping
│       ├── histogram.py           # SAR-only/unpaired/preset histogram mapping
│       ├── normalization.py       # dB/log/scale/[-1,1]/uint8
│       ├── resize_tile.py         # resize, crop, tile
│       └── qa.py                  # no-data, NaN, histogram report
├── configs/
│   ├── m4sar_pipeline.default.yaml
│   └── m4sar_pipeline.example.yaml
├── scripts/
│   ├── preprocess_pipeline.py     # CLI runner
│   └── export_cut_layout.py       # CUT 폴더 구조 변환
├── docs/
│   └── README_pipeline.md
└── tests/
    └── test_preprocessing_pipeline.py
```

---

## 4. 데이터 입출력 규칙

### 4.1 입력 폴더

`input_dir`는 SAR 이미지가 들어 있는 폴더를 의미한다.

지원 확장자 예시:

```text
.png, .jpg, .jpeg, .bmp, .tif, .tiff
```

권장 규칙:

- 하위 폴더 recursive scan 지원
- 숨김 파일 제외
- 이미지가 아닌 annotation 파일 제외
- 처리 순서는 기본적으로 파일명 정렬
- `shuffle: true`일 때는 seed 기반 deterministic shuffle
- `max_items: 0` 또는 `null`이면 전체 처리

### 4.2 출력 폴더

`output_dir`에는 다음 결과물을 저장한다.

```text
output_dir/
├── images/                        # 최종 전처리 이미지
├── preview/                       # Web UI 미리보기용 before/after
├── intermediate/                  # 옵션: step별 중간 결과
├── logs/
│   └── preprocess_YYYYMMDD_HHMMSS.log
├── manifest.csv                   # 입력/출력/적용 step/통계
├── pipeline_config.resolved.yaml  # 실행 당시 확정 설정
└── cut_layout/                    # 옵션: CUT용 trainA/testA export
    ├── trainA/
    └── testA/
```

### 4.3 CUT 학습용 출력

CUT repo에서 사용하는 일반적인 구조는 다음과 같다.

```text
datasets/
└── M4-SAR-cut/
    ├── trainA/    # SAR 또는 전처리된 SAR
    ├── trainB/    # Optical target domain, unpaired 가능
    ├── testA/
    └── testB/
```

전처리 파이프라인은 기본적으로 `trainA/testA`를 생성한다. Optical 데이터가 이미 있다면 `trainB/testB`는 원본 optical 폴더를 symlink 또는 copy 방식으로 연결한다.

---

## 5. Pipeline 실행 흐름

권장 기본 실행 순서는 다음과 같다.

```text
입력 이미지 로드
  ↓
No-data / NaN / Inf 처리
  ↓
SAR intensity 변환
  - linear amplitude/intensity 확인
  - 필요 시 log 또는 dB compression
  ↓
Speckle filtering
  - Lee / Frost / BM3D / Refined Lee / Gamma-MAP 중 1개
  ↓
Outlier clipping
  - 예: min 0.2%, max 99.8%
  ↓
Histogram mapping
  - SAR-only / unpaired optical reference / preset target histogram
  ↓
Resize / crop / tile
  - CUT image_size에 맞춤
  ↓
Channel adapter
  - grayscale → 3-channel
  - multi-pol → 선택 채널 또는 RGB-like stack
  ↓
저장 및 manifest 기록
```

전처리 순서는 config의 `steps` 배열 순서를 그대로 따른다. 예를 들어 clipping을 speckle filtering보다 먼저 적용하고 싶으면 배열 순서를 바꾸면 된다.

---

## 6. 설정 파일 예시

`configs/m4sar_pipeline.example.yaml`

```yaml
project:
  name: "m4sar_sar_preprocessing_for_cut"
  seed: 42

io:
  input_dir: "./datasets/M4-SAR/raw_sar"
  output_dir: "./datasets/M4-SAR-preprocessed"
  recursive: true
  extensions: [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
  max_items: 0                 # 0이면 전체 처리
  shuffle: false
  save_format: "png"           # png 또는 tif 권장
  overwrite: false
  save_intermediate: false

runtime:
  num_workers: 4
  device: "auto"               # auto, cpu, cuda
  log_level: "info"

pipeline:
  steps:
    - name: "validate_image"
      enabled: true
      params:
        drop_empty: true
        handle_nan: "zero"     # zero, median, skip
        handle_inf: "clip"

    - name: "sar_intensity_transform"
      enabled: true
      params:
        mode: "log1p"          # none, log1p, db
        eps: 1.0e-6

    - name: "speckle_filter"
      enabled: true
      params:
        method: "refined_lee"  # lee, frost, bm3d, refined_lee, gamma_map
        window_size: 7
        enl: "auto"
        damping_factor: 2.0
        bm3d_sigma: "auto"

    - name: "outlier_clipping"
      enabled: true
      params:
        mode: "percentile"
        min_percentile: 0.2
        max_percentile: 99.8
        percentile_scope: "per_image"  # per_image 또는 dataset
        ignore_zero: true

    - name: "histogram_mapping"
      enabled: true
      params:
        mode: "sar_only"       # sar_only, unpaired_optical_reference, preset
        target_profile: "optical_like_v1"
        optical_reference_dir: null
        bins: 1024
        preserve_rank: true
        clahe:
          enabled: false
          clip_limit: 2.0
          tile_grid_size: [8, 8]

    - name: "resize_or_tile"
      enabled: true
      params:
        mode: "resize"         # resize, center_crop, tile
        image_size: 256
        interpolation: "area"

    - name: "channel_adapter"
      enabled: true
      params:
        output_channels: 3
        strategy: "repeat_gray" # repeat_gray, vv_vh_ratio, selected_channels

    - name: "normalize_for_cut"
      enabled: true
      params:
        output_range: "uint8"   # uint8 저장 후 train.py에서 [-1,1] 로드
```

---

## 7. 모듈 인터페이스 설계

모든 전처리 기능은 동일한 인터페이스를 갖는다.

```python
class PreprocessStep:
    name: str

    def __init__(self, **params):
        self.params = params

    def apply(self, image, context):
        """
        Args:
            image: np.ndarray 또는 torch/tf tensor
            context: dict
                - input_path
                - output_path
                - metadata
                - image_stats
                - runtime options

        Returns:
            image: processed image
            context: updated context
        """
        raise NotImplementedError
```

Step registry 예시:

```python
STEP_REGISTRY = {
    "validate_image": ValidateImageStep,
    "sar_intensity_transform": SARIntensityTransformStep,
    "speckle_filter": SpeckleFilterStep,
    "outlier_clipping": OutlierClippingStep,
    "histogram_mapping": HistogramMappingStep,
    "resize_or_tile": ResizeOrTileStep,
    "channel_adapter": ChannelAdapterStep,
    "normalize_for_cut": NormalizeForCUTStep,
}
```

Pipeline runner 예시:

```python
class PreprocessPipeline:
    def __init__(self, steps):
        self.steps = steps

    def run_one(self, image, context):
        for step in self.steps:
            if not step.enabled:
                continue
            image, context = step.apply(image, context)
        return image, context
```

이 구조를 사용하면 다음이 가능하다.

- Web UI에서 step 순서 변경
- step enable/disable
- 새 step 추가
- 특정 step만 unit test
- step별 중간 결과 저장
- step별 파라미터만 UI에 노출

---

## 8. Web UI 설계

기존 Gradio 기반 UI에 `SAR Preprocessing` 탭을 추가하는 방식이 가장 단순하다.

### 8.1 UI 탭 구성

```text
[Tab 1] Dataset / Folder
  - SAR input folder
  - Optional optical target folder
  - Output folder
  - Recursive scan
  - Max items
  - Shuffle
  - Scan button
  - File count preview

[Tab 2] Pipeline Builder
  - Step list
  - Enable / disable checkbox
  - Move up / move down button
  - Reset to recommended order
  - Save config
  - Load config

[Tab 3] Speckle Filter
  - Method radio/dropdown
    * Lee
    * Frost
    * BM3D
    * Refined Lee
    * Gamma-MAP
  - Window size
  - ENL
  - Frost damping factor
  - BM3D sigma

[Tab 4] Clipping / Histogram
  - Min percentile
  - Max percentile
  - Percentile scope
  - Histogram mapping mode
  - Target profile
  - Optional optical reference dir
  - CLAHE option

[Tab 5] Preview
  - Select sample image
  - Before image
  - After image
  - Histogram before/after
  - Apply preview only

[Tab 6] Run / Export
  - Run preprocessing
  - Stop
  - Progress bar
  - Current file
  - Log window
  - Export to CUT layout
  - Open output folder path
```

### 8.2 Web UI 필수 파라미터

| UI 항목 | 타입 | 기본값 | 설명 |
|---|---:|---:|---|
| `input_dir` | text | `./datasets/M4-SAR/raw_sar` | 처리할 SAR 폴더 |
| `output_dir` | text | `./datasets/M4-SAR-preprocessed` | 결과 저장 폴더 |
| `max_items` | number | `0` | 0이면 전체, N이면 N개만 처리 |
| `recursive` | checkbox | `true` | 하위 폴더 포함 |
| `shuffle` | checkbox | `false` | 처리 대상 섞기 |
| `step_order` | sortable list | 권장 순서 | 전처리 실행 순서 |
| `speckle_method` | dropdown | `refined_lee` | speckle filter 1개 선택 |
| `clip_min_percentile` | number | `0.2` | 하위 clipping percentile |
| `clip_max_percentile` | number | `99.8` | 상위 clipping percentile |
| `histogram_mode` | dropdown | `sar_only` | histogram mapping 방식 |
| `image_size` | number | `256` | CUT 입력 크기 |
| `export_cut_layout` | checkbox | `true` | trainA/testA 구조로 export |

---

## 9. Speckle Filtering

SAR 영상은 coherent imaging 특성 때문에 speckle noise가 발생한다. Speckle filtering은 SAR의 edge와 구조를 가능한 보존하면서 multiplicative noise를 줄이는 단계다.

Web UI에서는 아래 필터 중 정확히 하나만 선택해 사용한다. 파이프라인 관점에서는 `speckle_filter` step 하나가 있고, 내부 파라미터 `method`가 바뀌는 구조다.

### 9.1 지원 필터 목록

| Method | 권장 상황 | 장점 | 주의점 | 주요 파라미터 |
|---|---|---|---|---|
| `lee` | 빠른 기본 denoise | 구현 쉬움, 속도 빠름 | edge가 약간 부드러워질 수 있음 | `window_size`, `enl` |
| `frost` | edge 보존이 필요한 경우 | 거리 기반 가중치로 edge 보존 | damping 설정에 민감 | `window_size`, `damping_factor`, `enl` |
| `bm3d` | 품질 우선, 소량/오프라인 처리 | texture 보존과 denoise 성능 우수 | 느림, 파라미터 추정 필요 | `bm3d_sigma` |
| `refined_lee` | 기본 추천 | 방향성 window로 edge 보존 우수 | Lee보다 구현 복잡 | `window_size`, `enl` |
| `gamma_map` | multi-look SAR, multiplicative noise 모델 | SAR 통계 모델과 잘 맞음 | ENL 추정 품질 영향 | `window_size`, `enl` |

### 9.2 기본 추천

권장 기본값:

```yaml
speckle_filter:
  method: "refined_lee"
  window_size: 7
  enl: "auto"
```

이유:

- Lee보다 edge 보존이 좋다.
- Frost보다 파라미터 민감도가 낮다.
- BM3D보다 속도가 빠르다.
- SAR 전처리 기본값으로 안전하다.

### 9.3 필터 선택 가이드

- 빠른 실험: `lee`
- edge/structure 보존 중요: `refined_lee`
- 강한 speckle 억제: `frost`
- 품질 우선, 처리량 작음: `bm3d`
- multi-look SAR 통계 기반 처리: `gamma_map`

---

## 10. Outlier Clipping

Outlier clipping은 SAR intensity의 극단적인 bright scatterer 때문에 전체 contrast가 눌리는 문제를 줄이기 위한 단계다.

### 10.1 Percentile 기반 clipping

예시:

```yaml
outlier_clipping:
  mode: "percentile"
  min_percentile: 0.2
  max_percentile: 99.8
  ignore_zero: true
```

의미:

- 하위 0.2%보다 작은 픽셀은 하위 임계값으로 clipping
- 상위 99.8%보다 큰 픽셀은 상위 임계값으로 clipping
- 0 값이 no-data 또는 background라면 percentile 계산에서 제외

### 10.2 추천값

| 목적 | min | max |
|---|---:|---:|
| 보수적 clipping | 0.1 | 99.9 |
| 기본 추천 | 0.2 | 99.8 |
| contrast 강화 | 0.5 | 99.5 |
| 강한 highlight 억제 | 1.0 | 99.0 |

### 10.3 Scope

`percentile_scope`는 두 가지를 지원한다.

```yaml
percentile_scope: "per_image"
```

- 이미지별로 percentile 계산
- 각 이미지 contrast가 균일해짐
- scene 간 절대 intensity 차이는 약해질 수 있음

```yaml
percentile_scope: "dataset"
```

- 전체 데이터셋에서 percentile 계산
- scene 간 상대 intensity 유지
- 사전 scan 필요
- 대규모 데이터에서는 sampling 기반 추정 권장

초기 구현은 `per_image`를 기본값으로 두고, 추후 dataset-wide 통계를 추가하는 것이 좋다.

---

## 11. Histogram Mapping

Histogram mapping은 SAR intensity 분포를 optical-like intensity 분포로 재배치하는 단계다.

### 11.1 지원 모드

```yaml
histogram_mapping:
  mode: "sar_only"
```

SAR 이미지 하나만으로 optical-like tone curve를 만든다. paired optical 데이터가 없어도 동작한다.

```yaml
histogram_mapping:
  mode: "unpaired_optical_reference"
  optical_reference_dir: "./datasets/M4-SAR/raw_optical"
```

1:1 pair가 아니어도 optical 폴더 전체의 target histogram을 추정하여 SAR 분포를 매핑한다. CUT 학습과 가장 잘 맞는 방식이다.

```yaml
histogram_mapping:
  mode: "preset"
  target_profile: "optical_like_v1"
```

미리 정의된 optical-like target CDF를 사용한다. optical 데이터가 전혀 없을 때 reproducible baseline으로 사용한다.

### 11.2 SAR-only histogram mapping 방식

SAR-only 모드는 다음 순서로 처리한다.

```text
1. SAR image를 float32로 변환
2. no-data / NaN / Inf 제거
3. log1p 또는 dB compression
4. percentile clipping
5. robust min-max normalization
6. optical_like_v1 target CDF 생성
7. source CDF → target CDF로 rank-preserving mapping
8. 필요 시 CLAHE 적용
9. 0~255 uint8 또는 0~1 float로 저장
```

핵심은 `rank-preserving`이다. 픽셀의 상대 밝기 순위는 유지하되, 전체 톤 분포만 optical-like로 바꾼다.

### 11.3 optical_like_v1 target profile 예시

`optical_like_v1`은 다음 특성을 갖는 target CDF로 정의한다.

- shadow 영역이 완전히 뭉개지지 않도록 하위 tone을 완만하게 확장
- 중간 tone contrast를 높임
- highlight 영역은 clipping 이후 부드럽게 roll-off
- 전체 출력은 0~255 범위에 안정적으로 분포

예시 구현 아이디어:

```python
def optical_like_v1_cdf(num_bins=1024):
    x = np.linspace(0, 1, num_bins)

    # gamma curve: SAR dark-heavy 분포를 optical-like mid-tone으로 이동
    y = np.power(x, 0.75)

    # highlight roll-off
    y = 1.0 - np.power(1.0 - y, 1.15)

    # monotonic CDF 보장
    y = np.maximum.accumulate(y)
    y = (y - y.min()) / (y.max() - y.min() + 1e-8)
    return x, y
```

### 11.4 Unpaired optical reference mapping

Optical reference 폴더가 있으면 1:1 pair가 아니어도 다음 방식으로 target CDF를 만든다.

```text
1. optical_reference_dir scan
2. max_reference_items 개 sampling
3. RGB optical 이미지를 luminance Y로 변환
4. optical histogram 누적
5. target CDF 생성
6. SAR source CDF를 optical target CDF에 매핑
```

이 방식은 SAR-only preset보다 target domain 분포가 실제 optical 데이터에 가까우므로, CUT 학습 전처리로 더 권장된다.

---

## 12. 추가 권장 전처리 기능

STN은 현재 우선순위가 낮아도 된다. CUT는 unpaired translation이 가능하므로 pixel-level 1:1 정렬을 강제하지 않는다. 대신 아래 전처리를 우선 구현하는 것이 실용적이다.

### 12.1 SAR intensity transform

SAR 원본이 linear scale인지 dB scale인지에 따라 contrast가 크게 달라진다. 입력을 통일하기 위해 다음 옵션을 둔다.

```yaml
sar_intensity_transform:
  mode: "log1p"  # none, log1p, db
  eps: 1.0e-6
```

권장:

- 일반 이미지 파일로 받은 SAR: `log1p`
- 이미 dB로 저장된 SAR: `none`
- linear power/intensity GeoTIFF: `db`

### 12.2 No-data / NaN / Inf handling

SAR 또는 GeoTIFF 계열 데이터에는 no-data, NaN, Inf, border artifact가 섞일 수 있다.

```yaml
validate_image:
  drop_empty: true
  handle_nan: "zero"
  handle_inf: "clip"
  nodata_value: 0
```

### 12.3 Resize / crop / tile

CUT 입력 크기와 GPU memory를 고려해 `image_size`를 통일한다.

```yaml
resize_or_tile:
  mode: "resize"
  image_size: 256
```

대형 SAR tile을 보존하고 싶으면 `tile` 모드를 쓴다.

```yaml
resize_or_tile:
  mode: "tile"
  tile_size: 256
  stride: 256
  drop_empty_tiles: true
```

### 12.4 Channel adapter

CUT repo가 RGB 3-channel 입력을 기대한다면 SAR grayscale을 3-channel로 바꿔야 한다.

```yaml
channel_adapter:
  output_channels: 3
  strategy: "repeat_gray"
```

multi-polarization 데이터가 있다면 다음 전략도 지원할 수 있다.

```yaml
strategy: "vv_vh_ratio"
```

예시:

- R = VV
- G = VH
- B = VV / (VH + eps)

### 12.5 QA report

전처리 결과가 너무 어둡거나 밝게 saturate되는 것을 자동 감지한다.

기록할 통계:

- min / max
- mean / std
- percentile 1, 50, 99, 99.8
- zero ratio
- saturated low/high ratio
- NaN/Inf count
- 적용 step 목록
- 처리 시간

---

## 13. CLI 사용 예시

### 13.1 기본 실행

```bash
python scripts/preprocess_pipeline.py \
  --config configs/m4sar_pipeline.example.yaml
```

### 13.2 입력/출력/개수만 override

```bash
python scripts/preprocess_pipeline.py \
  --input_dir ./datasets/M4-SAR/raw_sar \
  --output_dir ./datasets/M4-SAR-preprocessed \
  --max_items 100
```

### 13.3 Speckle filter 변경

```bash
python scripts/preprocess_pipeline.py \
  --config configs/m4sar_pipeline.example.yaml \
  --speckle_method bm3d \
  --max_items 20
```

### 13.4 CUT layout export

```bash
python scripts/export_cut_layout.py \
  --preprocessed_sar_dir ./datasets/M4-SAR-preprocessed/images \
  --optical_dir ./datasets/M4-SAR/raw_optical \
  --out_dir ./datasets/M4-SAR-cut \
  --test_ratio 0.1 \
  --link_mode symlink
```

---

## 14. CUT 학습 연결

전처리 후 CUT 학습은 다음과 같이 연결한다.

```bash
python train.py \
  --mode cut \
  --train_src_dir ./datasets/M4-SAR-cut/trainA \
  --train_tar_dir ./datasets/M4-SAR-cut/trainB \
  --test_src_dir ./datasets/M4-SAR-cut/testA \
  --test_tar_dir ./datasets/M4-SAR-cut/testB \
  --save_n_epoch 10
```

SAR-only baseline을 먼저 보고 싶다면 `trainB`를 실제 optical 대신 histogram-mapped pseudo-optical 결과로 둘 수 있다. 단, 이 경우 모델은 진짜 optical domain을 배우는 것이 아니라 전처리된 pseudo domain을 학습하게 된다.

권장 실험 순서:

1. SAR-only histogram mapping 결과를 시각적으로 확인
2. unpaired optical reference가 있을 경우 `trainB`로 optical 폴더 연결
3. `trainA`는 전처리된 SAR
4. `trainB`는 optical target domain
5. CUT 또는 FastCUT 학습
6. inference 결과와 histogram-mapped baseline 비교

---

## 15. Manifest 설계

`manifest.csv` 예시 컬럼:

```csv
input_path,output_path,status,error,original_height,original_width,output_height,output_width,steps,speckle_method,clip_min_p,clip_max_p,hist_mode,p01,p50,p99,p998,zero_ratio,sat_low_ratio,sat_high_ratio,elapsed_sec
```

예시 row:

```csv
./raw/000001.tif,./images/000001.png,success,,512,512,256,256,"validate|log1p|refined_lee|clip|hist|resize|channel",refined_lee,0.2,99.8,sar_only,3,91,221,248,0.02,0.001,0.002,0.143
```

Manifest는 재현성과 디버깅에 중요하다. Web UI에서도 manifest를 다운로드할 수 있게 한다.

---

## 16. 로그 정책

로그에는 다음 정보를 남긴다.

- 실행 config path
- resolved config
- 입력 폴더
- 출력 폴더
- 처리 대상 개수
- 현재 처리 파일
- step별 처리 시간
- skip된 파일과 이유
- exception traceback
- 최종 성공/실패 개수

예시:

```text
[12:03:11] scan input_dir=./datasets/M4-SAR/raw_sar recursive=True
[12:03:12] found 112184 image files
[12:03:12] max_items=100 -> selected 100 files
[12:03:13] start pipeline: validate -> log1p -> speckle_filter(refined_lee) -> clipping -> histogram_mapping -> resize -> channel_adapter
[12:03:13] processing 000001.tif
[12:03:13] done 000001.tif elapsed=0.142s
...
[12:04:22] completed success=100 failed=0
```

---

## 17. 테스트 계획

### 17.1 Unit test

각 step 단위로 테스트한다.

- 입력 shape 유지 여부
- dtype 변환 규칙
- NaN/Inf 제거 여부
- percentile clipping 범위
- histogram mapping monotonicity
- speckle filter 실행 가능 여부
- config serialization/deserialization

### 17.2 Smoke test

작은 폴더에서 5~10장만 처리한다.

```bash
python scripts/preprocess_pipeline.py \
  --input_dir ./tests/sample_sar \
  --output_dir ./tmp/preprocess_smoke \
  --max_items 10
```

성공 조건:

- output image 개수 = input selected 개수
- manifest 존재
- log 존재
- preview 생성
- Web UI preview 정상 표시

### 17.3 Visual QA

Web UI에서 다음을 비교한다.

- 원본 SAR
- speckle filtering 결과
- clipping 결과
- histogram mapping 결과
- 최종 CUT 입력 이미지
- histogram before/after

---

## 18. 권장 기본 preset

### 18.1 빠른 실험 preset

```yaml
pipeline:
  steps:
    - name: "validate_image"
      enabled: true
    - name: "sar_intensity_transform"
      enabled: true
      params:
        mode: "log1p"
    - name: "speckle_filter"
      enabled: true
      params:
        method: "lee"
        window_size: 5
    - name: "outlier_clipping"
      enabled: true
      params:
        min_percentile: 0.2
        max_percentile: 99.8
    - name: "histogram_mapping"
      enabled: true
      params:
        mode: "sar_only"
    - name: "resize_or_tile"
      enabled: true
      params:
        image_size: 256
    - name: "channel_adapter"
      enabled: true
      params:
        output_channels: 3
```

### 18.2 품질 우선 preset

```yaml
pipeline:
  steps:
    - name: "validate_image"
      enabled: true
    - name: "sar_intensity_transform"
      enabled: true
      params:
        mode: "log1p"
    - name: "speckle_filter"
      enabled: true
      params:
        method: "refined_lee"
        window_size: 7
        enl: "auto"
    - name: "outlier_clipping"
      enabled: true
      params:
        min_percentile: 0.1
        max_percentile: 99.9
    - name: "histogram_mapping"
      enabled: true
      params:
        mode: "unpaired_optical_reference"
        optical_reference_dir: "./datasets/M4-SAR/raw_optical"
        bins: 2048
    - name: "resize_or_tile"
      enabled: true
      params:
        image_size: 256
    - name: "channel_adapter"
      enabled: true
      params:
        output_channels: 3
```

### 18.3 SAR-only baseline preset

```yaml
pipeline:
  steps:
    - name: "validate_image"
      enabled: true
    - name: "sar_intensity_transform"
      enabled: true
      params:
        mode: "log1p"
    - name: "speckle_filter"
      enabled: true
      params:
        method: "gamma_map"
        window_size: 7
        enl: "auto"
    - name: "outlier_clipping"
      enabled: true
      params:
        min_percentile: 0.5
        max_percentile: 99.5
    - name: "histogram_mapping"
      enabled: true
      params:
        mode: "sar_only"
        target_profile: "optical_like_v1"
        clahe:
          enabled: true
          clip_limit: 2.0
          tile_grid_size: [8, 8]
    - name: "resize_or_tile"
      enabled: true
      params:
        image_size: 256
    - name: "channel_adapter"
      enabled: true
      params:
        output_channels: 3
```

---

## 19. 구현 우선순위

### Phase 1: 최소 동작 버전

- 폴더 scan
- `input_dir`, `output_dir`, `max_items`
- step enable/disable
- step 순서 config 기반 변경
- Lee / Frost / Refined Lee
- percentile clipping
- SAR-only histogram mapping
- resize
- 3-channel 저장
- manifest/log
- Web UI preview

### Phase 2: 품질 개선

- BM3D
- Gamma-MAP
- dataset-wide percentile
- unpaired optical reference histogram
- CLAHE option
- intermediate save
- histogram before/after plot
- multiprocessing

### Phase 3: CUT workflow 통합

- M4-SAR 폴더 자동 분류
- `trainA/trainB/testA/testB` export
- 기존 `gui.py` 학습 탭과 config 공유
- 전처리 완료 후 학습 config 자동 채우기
- 전처리 preset 저장/불러오기

---

## 20. 주의사항

1. Speckle filtering을 너무 강하게 적용하면 CUT가 학습해야 할 SAR 구조 정보가 사라질 수 있다.
2. Histogram mapping은 optical semantic을 생성하지 않는다. 색, 재질, 계절감 등은 target optical domain이 있어야 학습된다.
3. M4-SAR가 aligned pair를 제공하더라도 CUT 학습은 paired alignment를 필수로 요구하지 않는다.
4. STN은 초기 구현 우선순위에서 제외해도 된다. 대신 preprocessing consistency, channel normalization, histogram QA가 더 중요하다.
5. SAR 원본이 이미 8-bit PNG/JPG로 변환된 상태라면 dB 변환을 중복 적용하지 않도록 `sar_intensity_transform.mode`를 확인해야 한다.
6. Web UI에서 사용자가 step 순서를 바꿀 수 있게 하되, `Reset to recommended order` 버튼을 제공해야 한다.

---

## 21. 참고 자료

- CUT repo: https://github.com/nanseko/nanse_test_repo
- M4-SAR dataset: https://huggingface.co/datasets/wchao0601/m4-sar
- M4-SAR paper: https://arxiv.org/abs/2505.10931

# 전처리 파라미터 자동 최적화 (좌표하강)

전처리 **순서 자동 최적화**(⑧, `optimize_orders`)의 다음 단계입니다. 순서를 확정한 뒤, 각 전처리
스텝의 **세부 파라미터**(클리핑 세기, speckle 윈도우 크기, intensity 모드 등)를 자동으로 튜닝합니다.
GUI 탭 2의 아코디언 **⑨ 전처리 파라미터 자동 최적화**.

## 왜 별도 단계인가

순서 탐색은 4스텝의 순열(24) × speckle(5) = 120 후보를 다루지만, **각 스텝의 파라미터는 고정된
기본값**을 씁니다(예: `outlier_clipping`은 항상 0.2/99.8 퍼센타일). 파라미터까지 순서와 동시에
전수 탐색하면 조합이 수천~수만으로 폭발합니다. 그래서 **"순서 먼저, 파라미터 나중"** 2단계로 나눕니다.

## 탐색 방식: 좌표하강 (coordinate descent)

한 번에 **한 파라미터만** 그리드로 훑어 최적값을 확정하고, 다음 파라미터로 넘어갑니다. 나머지
파라미터는 그때까지의 최적값으로 고정합니다. 후보 수가 **곱이 아니라 합**으로 늘어 매우 저렴합니다
(예: 3+4+5+5+2+3 ≈ 20여 회 평가). 각 파라미터의 최적값이 로그에 명확히 남아 해석도 쉽습니다.

- `passes`(기본 1): 전체 스텝을 한 바퀴 도는 횟수. 2로 하면 스텝 간 상호작용을 한 번 더 반영하지만
  비용이 대략 2배가 됩니다. 개선이 없으면 자동 조기 종료합니다.

## 조절 대상 파라미터

각 스텝이 **자기 자신에** `PARAM_SPACE`를 선언합니다(`preprocessing/steps.py`). 비어 있으면 탐색에서
자동 제외되므로, 리사이즈 등 구조 스텝은 특별한 처리 없이 빠집니다.

| 스텝 | 파라미터 | 후보 |
|---|---|---|
| `sar_intensity_transform` | `mode` | none / log1p / db |
| `speckle_filter` | `window_size` | 5 / 7 / 9 / 11 |
| | `damping_factor` | 1.0 / 2.0 / 3.0 (**frost일 때만**) |
| `outlier_clipping` | `min_percentile` | 0.0 / 0.2 / 0.5 / 1.0 / 2.0 |
| | `max_percentile` | 98.0 / 99.0 / 99.5 / 99.8 / 99.95 |
| `histogram_mapping` | `clahe.enabled` | off / on (**cv2 설치 시만**) |
| | `clahe.clip_limit` | 1.0 / 2.0 / 4.0 |
| **제외(구조/포맷 고정)** | `resize_or_tile`, `channel_adapter`, `normalize_for_cut`, `validate_image` | — |

**적용 안 되는 파라미터는 자동으로 걸러냅니다**(relevance pruning): `damping_factor`는 speckle 방법이
frost가 아니면 무의미하므로 스윕하지 않고, `clahe.*`는 cv2가 없으면(설치 안 됐으면) 실질적으로 무효과라
스윕하지 않습니다. `histogram_mapping`의 `mode`는 "어떤 참조를 쓸지"라는 데이터 가용성 선택이지
수치 노브가 아니라서 스윕 대상에서 뺐습니다(순서 탐색과 동일하게 사용자가 지정).

## 순서 → 파라미터 자동 연결

⑨의 결과 폴더를 ⑧과 **같은 폴더**로 두면(기본값이 그렇습니다), ⑨는 그 폴더의
`best_pipeline.json`(⑧이 저장한 최적 순서)에서 **순서와 speckle을 자동으로 읽어옵니다.** 사용자는
버튼만 누르면 됩니다.

⑧을 건너뛰고 파라미터만 바로 튜닝하고 싶으면 UI의 **"순서 직접 지정"** 에 순서(쉼표/`>` 구분)와
speckle 방법을 입력하면 됩니다.

## 랭킹 지표

순서 탐색과 동일한 이미지 평가지표(원본 SAR 기준 PSNR·CC·EPI, 그리고 ENL·speckle_index·composite)를
씁니다. 좌표하강은 선택한 `primary` 지표를 기준으로 **개선이 있을 때만** 값을 갱신하므로, 결과 metric은
기본 파라미터 대비 **항상 같거나 더 좋습니다**(단조 개선). FID는 이 단계에선 쓰지 않습니다(무참조
이미지 지표만으로 충분히 빠르게 반복하기 위함 — FID 최종 확인은 순서 탐색이나 모델 평가에서).

## 결과물

```
<결과폴더>/
  param_search_results.csv        # 모든 평가 (signature, overrides, 지표) — 재개용
  best_params_pipeline.json       # 순서 + 최적 param_overrides + full_steps
```

`best_params_pipeline.json`의 `full_steps`는 최적 파라미터가 반영된 완전한 스텝 설정이라, 실제
전처리 실행 설정으로 바로 쓸 수 있습니다.

## 재개 (resumable)

`param_search_results.csv`에 모든 평가가 signature와 함께 기록되어, **중단 후 다시 실행하면 이미
평가한 조합은 건너뜁니다.** 좌표하강 경로가 실행 간 흔들리지 않도록, metric을 CSV 저장 정밀도(소수
4자리)로 **일관되게 반올림**해서 경로를 결정론적으로 만들었습니다 — 그래서 완료된 탐색을 다시 돌리면
새 평가가 0건인 완전한 no-op입니다(구현 중 이 부분에서 재개가 no-op이 아니던 버그를 발견해 수정).

## 사용법 (GUI)

1. 탭 2 아코디언 **⑧**에서 순서 자동 최적화 실행 → `best_pipeline.json` 생성.
2. 바로 아래 **⑨**에서 결과 폴더를 ⑧과 같게 두고 SAR 폴더·평가 장수만 확인 → **🚀 파라미터 자동
   최적화 실행**.
3. `best_params_pipeline.json`의 `full_steps`를 전처리 설정으로 사용.

## 코드로 사용

```python
import preprocessing as PP
for line in PP.optimize_params(
        sar_dir='./datasets/M4-SAR/raw_sar',
        out_dir='./datasets/_order_search',   # ⑧과 같은 폴더 -> best_pipeline.json 자동 연결
        n_images=300, primary='composite', passes=1):
    print(line)
# 또는 순서를 직접 지정(⑧ 없이):
#   PP.optimize_params(sar_dir, out_dir, order=[...], speckle_method='refined_lee', ...)
```

## 확장 방법 (다른 파라미터 추가)

스텝 클래스의 `PARAM_SPACE`에 후보 리스트를 추가하기만 하면 자동으로 탐색 대상이 됩니다. 중첩
파라미터는 dotted 키(`'clahe.clip_limit'`)를 씁니다. 새 파라미터가 특정 조건에서만 유효하면
`optimize.py::_relevant_param_space`에 pruning 규칙을 한 줄 추가하세요.

## 검증

`python tests/test_param_optimize.py` — PARAM_SPACE 선언(조절 대상 vs 구조 스텝), dotted 병합/오버라이드
적용, tunable 필터링 + frost 전용 damping pruning, best_pipeline.json 자동 연결, 좌표하강의 단조 개선
(기본값 대비 metric이 나빠지지 않음), 결정론적 재개(재실행 시 새 평가 0건)를 확인합니다.

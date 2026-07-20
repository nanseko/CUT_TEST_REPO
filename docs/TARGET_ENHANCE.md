# 표적(건물·차량·비행체) 시각적 강조 (CFAR + saliency)

`docs/TARGET_ENHANCEMENT_SPEC.md`의 stage A(CFAR+saliency 검출)+stage B(saliency 가중 국소 강조)
구현입니다. GUI 탭 9 **"9. 형상 후처리 (직사각 스냅)"** 하단, 기존 사각 스냅 섹션과 독립적으로 실행할
수 있습니다. C(클래스별 형상/신뢰도 주석)와 D(표적 초해상)는 이 구현에 포함되지 않았습니다.

## 왜 전역이 아니라 표적 영역에만 강조를 거나

`fake_B` 전체에 선명화/대비 강화를 걸면 SAR 유래 스페클(배경 잡음)까지 함께 증폭됩니다. 그래서 이
모듈은 **항상** 표적다움(saliency) 가중치로 블렌딩합니다: saliency가 높은(표적일 가능성이 큰) 픽셀만
강조되고, 나머지(배경)는 원본에 가깝게 유지됩니다. 근거 논문은 `TARGET_ENHANCEMENT_SPEC.md` §8 참고.

## Stage A: 검출 (CFAR + saliency)

**CFAR**(Constant False Alarm Rate)는 SAR 표준 적응형 국소 임계값 기법입니다. 각 픽셀 주변의
**훈련 링(training annulus)** — 가드 링을 제외한 바깥 사각 링 — 의 통계로 그 픽셀이 배경 대비 얼마나
튀는지 판단합니다. 가드 링을 두는 이유는 표적 자체가 몇 픽셀에 걸쳐 있을 수 있어, 표적 픽셀이 자신의
배경 추정에 섞여 들어가는 것(leak)을 막기 위함입니다.

두 방식 제공:
- **`sigma`**(기본, 권장): 국소 평균 + `k_sigma`×국소 표준편차를 임계값으로. 분포를 가정하지 않아
  렌더링된 8비트 `fake_B`(보정된 원시 SAR 강도가 아님)에 적합합니다.
- **`ca`**: 고전 Cell-Averaging CFAR(지수분포 클러터 가정, `T = N·(Pfa^(-1/N) − 1)`). 보정된 실제 SAR
  강도(`real_A`)에 적용할 때를 위해 제공. 8비트 렌더링 이미지에서는 `pfa`를 낮게(예: 1e-3) 두면 과도하게
  보수적일 수 있습니다.

CFAR 결과는 **brightness saliency**(국소 대비 기반, `saliency_floor` 미만이면 제외)로 교차검증해
고립된 잡음 픽셀의 오검출을 줄이고, 형태학적 열림/닫힘으로 정리한 뒤, 연결요소 + 면적 필터
(`min_area`~`max_area_frac`)로 최종 표적 후보 영역을 확정합니다.

## Stage B: 강조 (saliency 가중 국소 강조)

Stage A의 **연속값 saliency 맵**(이진 마스크가 아님 — 표적 가장자리에서 강조가 부드럽게 사라짐)을
블렌딩 가중치로 써서 세 가지 방법 중 선택 적용(순서대로 누적 가능):

- **`unsharp`**: 적응형 언샤프 마스킹. `out = in + amount·(in − blur(in))·weight`.
- **`guided`**: Guided filter(He, Sun & Tang 2010) 기반 base/detail 분해 후 디테일을 배율 증폭.
  일반 블러 대비 엣지 근처 halo가 적습니다(엣지 보존 스무딩).
- **`clahe`**: Lab 색공간 L채널에만 CLAHE 적용(RGB 채널별 CLAHE의 색조 왜곡 회피) 후 가중 블렌딩.
  **opencv 필요** — 미설치 시 명확한 오류로 실패합니다(다른 두 방법은 순수 NumPy).

## 사용법 (GUI)

1. 탭 9 하단 "표적(건물·차량·비행체) 시각적 강조" 섹션.
2. 입력 폴더 비우면 기본값은 `results_dir/name/test_<epoch>/images/fake_B` (탭 7 추론 결과).
3. 강조 방법 체크박스에서 선택(기본 unsharp+guided) → **▶ 표적 강조 실행**.
4. 결과는 `results_dir/name/test_<epoch>/enhanced/`에 저장되고(사각 스냅 결과 `rectified/`와 별도
   폴더), `target_regions.csv`에 검출된 모든 표적 영역(위치/크기/평균 saliency)이 기록됩니다.

## 코드로 사용

```python
import evaluation as EV

csv_path, n_regions, n_ok, n_fail, failures = EV.enhance_folder(
    'results/sar_cut/test_latest/images/fake_B',
    'results/sar_cut/test_latest/enhanced',
    methods=('unsharp', 'guided'))

# 단일 이미지:
import numpy as np
from PIL import Image
img = np.asarray(Image.open('fake_B/0001.png').convert('RGB'))
enhanced, info = EV.enhance_targets(img, methods=('unsharp', 'guided'), return_detection=True)
# info['mask'], info['saliency'], info['regions']
```

## 검증

`python tests/test_target_enhance.py` — CFAR(sigma/ca) 검출 정확도, saliency 국소대비 특성, 형태학/
연결요소/면적 필터, 각 강조 방법의 weight_map 게이팅(가중치 0=원본 유지, 1=변화), guided filter의
엣지 보존(평면 블러 대비 정량 비교), `enhance_folder`의 폴더 전체 처리·부분 실패 카운팅·opencv 미설치
시 즉시 실패(clahe 요청 시), GUI 래퍼(`gui.cut_enhance_targets`) 종단 동작을 확인합니다.

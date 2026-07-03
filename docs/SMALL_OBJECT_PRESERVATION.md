# 소형 물체(요트·탱크·건물) 형상 보존 — 강반사체 가중 손실 / 패치 샘플링

**증상**: SAR→EO 변환 시 원본 optical과 달리 건물이 뭉개지거나, 요트 같은 작은 물체가 구름/블롭 모양으로 바뀐다.

## 왜 이런 일이 생기는가

CUT는 비짝(unpaired) GAN이라 "EO처럼 보이면" 통과되고, **개별 물체의 정체성을 지키라는 압력이 없습니다.** 특히 요트·탱크처럼 작고 드문 강체 물체는:

1. **PatchNCE가 거의 못 봄** — 대조손실이 이미지 전체에서 `num_patches`개를 **균일 랜덤**으로 뽑기 때문에, 물체가 차지하는 픽셀이 적으면 그 위치가 표본에 뽑힐 확률도 작습니다.
2. **구조/에지 손실도 이미지 전체 평균** — `lambda_grad`/`lambda_lap`은 손실을 **전체 픽셀 평균**으로 계산하므로, 물체가 차지하는 비율이 작으면 손실에 대한 기여도도 작습니다.
3. 결과적으로 G는 손실을 더 줄이는 "안전한 평균"(뭉개짐/블롭)으로 수렴합니다.

요트·탱크의 금속 표면은 SAR에서 **강반사체(코너리플렉터)** 로 나타나 국소적으로 매우 밝은 픽셀 뭉치를 만듭니다. 이 특성을 이용해 "작은 물체가 있을 만한 위치"를 저렴하게 추정할 수 있습니다.

## 해결: 강반사체 saliency 가중치

`models/losses_extra.py`의 `reflector_saliency_map(source, boost)` 이 SAR 입력에서 **국소적으로 주변보다 밝은 픽셀**(강반사체 후보)에 `[1, 1+boost]` 범위의 가중치를 부여합니다. `boost=0`이면 균일(기존과 동일 동작).

이 saliency 맵을 두 곳에 적용합니다:

### ① `--reflector_weighted` — 구조 손실 가중
`lambda_grad`/`lambda_lap`이 **강반사체 위치에서 더 강하게** 벌점을 주도록 바꿉니다. 작은 물체 하나가 사라지면, 그 소수의 픽셀에 대한 손실이 `1+boost`배 커져서 평균에 묻히지 않습니다.

### ② `--saliency_patch_sampling` — PatchNCE 샘플링 편향
대조손실의 패치 샘플링을 균일 랜덤 대신 **saliency에 비례한 가중 샘플링**으로 바꿉니다. 강반사체 위치가 패치로 뽑힐 확률이 크게 올라가, PatchNCE가 그 물체의 내용을 실제로 감독하게 됩니다.

두 옵션 모두 `--reflector_boost`(기본 3.0) 로 강도를 조절합니다.

## 사용법

GUI 탭 4의 **"소형 물체 형상 보존"** 섹션, 또는 CLI:

```bash
python train.py --dataroot ./datasets/M4-SAR-cut --name sar_small_obj --CUT_mode CUT \
  --netG hrnet --lambda_grad 1.0 --lambda_lap 0.5 \
  --reflector_weighted --saliency_patch_sampling --reflector_boost 3.0
```

**권장 조합**: `netG=hrnet`(해상도 병목 자체를 줄임) + `lambda_grad`/`lambda_lap` + 위 두 옵션. HRNet만으로는 "물체가 다운샘플에서 사라지는 것"만 막고, 손실/샘플링의 "물체를 지켜야 할 이유가 없음" 문제는 이 두 옵션이 직접 해결합니다.

## 참고

- `boost`가 너무 크면 강반사체 위치에 과적합되어 배경 품질이 떨어질 수 있습니다. 3~5부터 시작해 조정하세요.
- `--saliency_patch_sampling`은 `real_A`(SAR→EO)와 `real_B`(identity, `nce_idt` 사용 시)에도 각각 적용됩니다. `real_B`(optical)에도 밝은 소형 물체가 있으면 동일한 방식으로 유리하게 작동합니다.
- **512 해상도 학습**, **전처리에서 강반사 신호 보호**(과도한 `outlier_clipping`/speckle 필터링 피하기, 순서 최적화 탭에서 EPI 높은 후보 선택)와 함께 쓰면 효과가 더 큽니다.
- 효과 검증은 `evaluation/` 모듈의 **EPI**(구조 보존)로 변경 전후를 비교하세요.
- 검증: `python tests/test_attention_port.py` — saliency 맵 값 범위, 가중 손실이 무가중보다 커짐, 가중 패치 샘플링이 고saliency 위치를 유의미하게 더 많이 뽑음(수십 배)을 확인합니다.

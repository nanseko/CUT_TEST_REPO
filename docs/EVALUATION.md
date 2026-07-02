# CUT 모델 출력 평가 (`evaluation/`)

백본(ResNet/HRNet)·attention·`lambda_*` 값을 바꿔가며 실험할 때, **각 실험의 SAR→EO 변환 품질을 객관적으로 비교**하기 위한 모듈입니다. GUI **탭 8. 모델 평가**에서 버튼 하나로 실행하고, 결과는 실험명별로 누적 비교됩니다.

## 왜 이 지표들인가

CUT는 **비짝(unpaired)** 이라 `fake_B`(SAR→EO 변환 결과)에 대응하는 "정답 EO"가 없습니다. 그래서 픽셀 단위 비교(PSNR/SSIM)를 fake_B에 직접 쓸 수 없고, 아래 4개 축을 조합합니다.

| 축 | 지표 | 비교 대상 | 방향 | 의미 |
|---|---|---|---|---|
| 도메인 근접도 | **FID** | fake_B ↔ EO 참조 세트 | ↓ | "EO 도메인에 얼마나 가까운가" — 핵심 지표 |
| | **KID** | fake_B ↔ EO 참조 세트 | ↓ | FID의 소표본 안정 버전 (unbiased MMD) |
| 구조 보존(허상 가드레일) | **EPI** | real_A ↔ fake_B | ↑ | 에지/윤곽 보존 — 낮으면 구조를 지어냈다는 신호 |
| | CC, PSNR | real_A ↔ fake_B | ↑ | 보조 구조 유사도 (참고용, 단독 판단 금지) |
| Identity 충실도 (짝 데이터!) | **PSNR/SSIM** | real_B ↔ idt_B=G(real_B) | ↑ | G가 EO 콘텐츠를 얼마나 안 망가뜨리는가. 유일하게 진짜 짝인 비교라 백본 비교에 저렴하고 정확 |
| 무참조 품질 | mean/std/sharpness/entropy | fake_B 단독 | — | 선명도·대비·정보량 (보조) |

**권장 판단 순서**: FID/KID로 "도메인에 가까운가" 먼저 보고, EPI로 "그런데 허상은 아닌가" 확인, idt PSNR/SSIM으로 "백본이 콘텐츠를 잘 보존하는가" 교차 확인.

## 사용법 (GUI)

1. **"7. 추론/테스트"** 로 `test.py` 를 실행해 `results_dir/name/test_<epoch>/images/{fake_B,real_A,real_B}` 를 생성합니다.
2. **"8. 모델 평가"** 탭에서:
   - `epoch` : 추론에 쓴 값과 동일하게
   - `실험명` : 이번 설정을 구분할 이름 (예: `hrnet_coord_lgrad1.0`)
   - `EO 참조 폴더` : 실제 광학 이미지 폴더 (FID/KID용)
   - `Identity 평가` 체크 시, **탭 4/5의 현재 설정**(netG/attention/hrnet 옵션)으로 생성기를 재구성해 체크포인트를 불러오고 `real_B → idt_B` 를 만들어 평가합니다. **⚠️ 탭 4/5 설정이 해당 체크포인트를 학습할 때와 동일해야** 합니다.
3. **▶ 평가 실행** → 로그 스트리밍 + 결과가 `eval_results.csv` 에 누적, 비교표에 표시됩니다.

오프라인(사내망)에서 FID/KID를 쓰려면 `docs/OFFLINE_FID.md` 대로 InceptionV3 가중치를 로컬에 두고 "InceptionV3 가중치 .pth 경로" 에 지정하세요.

## 결과 위치

```
<results_dir>/<name>/eval_logs/
  eval_results.csv     # 실험별 한 행 (누적, resumable)
  eval_<timestamp>.json
  idt_B_<epoch>/        # identity 평가 시 생성된 G(real_B) 이미지
```

## 코드로 사용

```python
import evaluation as EV

for line in EV.run_evaluation(
        results_dir='./results', name='sar_hr', epoch='latest',
        experiment='hrnet_coord_lgrad1.0',
        eo_dir='./datasets/Optical/trainB',
        checkpoints_dir='./checkpoints',
        cfg={'netG': 'hrnet', 'attention_type': 'coord', 'attention_resblocks': True,
             'hrnet_branches': 3, 'hrnet_modules': 3, 'hrnet_blocks': 2,
             'normG': 'instance', 'crop_size': 256},
        compute_identity=True):
    print(line)

rows = EV.load_eval_log('./results', 'sar_hr')   # 비교표용 누적 결과
```

`cfg` 는 체크포인트를 학습할 때 쓴 것과 **동일한 아키텍처 옵션**(netG/normG/attention_*/hrnet_*/no_antialias*)을 담은 dict입니다 — GUI에서는 탭 1~5의 현재 설정이 그대로 쓰입니다.

## 참고 / 주의

- **FID/KID는 표본이 어느 정도 있어야** 신뢰도가 있습니다(수백 장 권장). 너무 적으면 두 지표 모두 노이즈가 큽니다.
- **idt_B는 test.py가 저장하지 않습니다** (`nce_idt` 시각화는 학습 중에만 활성화). 그래서 평가 모듈이 체크포인트로 직접 `G(real_B)` 를 다시 계산합니다.
- 검증: `python tests/test_evaluation.py` — 지표 수식, ResNet/HRNet+attention 체크포인트 round-trip, CSV 로깅/누적(중복 없음)을 확인합니다.

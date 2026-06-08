# CUT + CBAM / Coordinate Attention (+ 구조·색 손실) — PyTorch 통합 가이드

이 문서는 TensorFlow CUT 포크(`nanseko/nanse_test_repo`)에 있던
**CBAM · Coordinate Attention** 과 **구조/색 손실**을, 공식 **PyTorch** CUT
(`taesungp/contrastive-unpaired-translation` 기반인 이 저장소)에 **직접 통합**한
내용을 설명합니다.

TF 포크가 PyTorch 포팅 파일(`pytorch/attention.py`, `pytorch/losses_extra.py`)을
이미 제공했기 때문에, 그 코드를 **그대로 재활용**하고 나머지는 공식 generator에
녹여 넣었습니다.

## 1. 추가/수정된 파일

| 파일 | 내용 |
|---|---|
| `models/attention.py` | `ChannelAttention`, `SpatialAttention`, `CBAM`, `CoordinateAttention`, `make_attention()` (NCHW). TF 포팅본을 **그대로 재사용**. |
| `models/losses_extra.py` | `gradient_loss`, `color_moment_loss` (선택적 구조/색 손실). TF 포팅본 **그대로 재사용**. |
| `models/networks.py` | 기존 `ResnetGenerator` / `ResnetBlock` 에 attention 삽입 로직을 통합하고, attention 인지 PatchNCE tap(`nce_default`)을 노출. `define_G` 가 opt에서 attention 옵션을 전달. |
| `models/cut_model.py` | attention 활성 시 `nce_layers` 자동 보정, `lambda_grad`/`lambda_color` 손실 연결. |
| `options/base_options.py` | `--attention_type/reduction/encoder/resblocks/decoder` 옵션 추가. |
| `tests/test_attention_port.py` | torch 스모크 테스트. |

> **핵심 설계 선택**: 별도의 `ResnetAttnGenerator` 를 두지 않고 **공식
> `ResnetGenerator` 에 통합**했습니다. 덕분에 공식 antialias `Downsample`/`Upsample`,
> `define_G`, `init_net`, 체크포인트 포맷을 그대로 재사용합니다.
> `attention_type == 'none'`(기본값)일 때 모듈 배치가 원본과 **바이트 단위로 동일**해
> 기존 체크포인트가 그대로 로드됩니다.

## 2. 옵션

```
--attention_type      none | cbam | coord     (기본 none = 완전 OFF)
--attention_reduction 16                       (채널 bottleneck 축소비)
--attention_encoder                            (stem / 각 downsampling 뒤에 삽입)
--attention_resblocks                          (각 ResnetBlock residual 정제)
--attention_decoder                            (각 upsampling 뒤에 삽입)
--lambda_grad         0.0                       (real_A↔fake_B 구조/그래디언트 보존)
--lambda_color        0.0                       (idt_B↔real_B 색 모멘트 일관성, nce_idt 필요)
```

## 3. ⚠️ nce_layers 자동 보정 (가장 중요)

공식 CUT 기본값은 `--nce_layers 0,4,8,12,16` 입니다. **attention 모듈이 끼면
`nn.Sequential` 인덱스가 밀립니다.** 이 저장소는 generator가 현재 구성에 맞는
올바른 tap 인덱스를 `netG.nce_default` 로 노출하고, `cut_model.py` 가 attention이
켜져 있으면 자동으로 `self.nce_layers` 를 그 값으로 교체합니다. (사용자가 직접
맞출 필요 없음.)

- attention OFF → `nce_default == [0, 4, 8, 12, 16]` (공식과 동일)
- `coord, encoder+resblocks, antialias` → `[0, 5, 10, 15, 19]`

## 4. 학습 명령 예시

```bash
# baseline (기존 CUT, 동작/결과 변화 없음)
python train.py --dataroot ./datasets/M4-SAR-cut --name sar_cut --CUT_mode CUT

# coordinate attention (encoder + resblocks) + 구조/색 손실
python train.py --dataroot ./datasets/M4-SAR-cut --name sar_cut_coord --CUT_mode CUT \
  --attention_type coord --attention_encoder --attention_resblocks \
  --attention_reduction 16 --lambda_grad 1.0 --lambda_color 1.0
# nce_layers 는 자동 보정됩니다.

# 추론 (학습 때와 같은 attention 옵션을 줘야 가중치가 맞습니다)
python test.py --dataroot ./datasets/M4-SAR-cut --name sar_cut_coord --CUT_mode CUT \
  --attention_type coord --attention_encoder --attention_resblocks
```

## 5. 검증

```bash
python tests/test_attention_port.py    # attention/generator/손실 스모크
python tests/test_preprocessing.py     # 전처리 파이프라인 스모크 (torch 불필요)
```

attention 종류/위치/리덕션의 의미는 TF 버전 문서
(`docs/attention_explained.html`, `docs/patchnce_explained.html`)와 동일합니다.

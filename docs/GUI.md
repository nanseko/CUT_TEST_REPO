# Web-UI (gui.py) — PyTorch CUT 전처리 · 학습 · 추론 GUI

`gui.py` 는 이 저장소의 PyTorch CUT 파이프라인(`train.py` / `test.py`)을 브라우저에서
구동하는 [Gradio](https://gradio.app) 기반 Web-UI 입니다. 일반 PC와 Google Colab
모두에서 동작합니다(Colab에서는 공개 share 링크가 자동 생성).

## 실행

```bash
pip install -r requirements.txt
pip install -r requirements_gui.txt
python gui.py                 # 로컬: http://127.0.0.1:7860
python gui.py --share         # 공개 share 링크 강제
```

Colab 에서는 `!python gui.py` 만 실행하면 됩니다(share 링크 자동).

## 탭 구성

| 탭 | 기능 |
|---|---|
| 0. 데이터셋 다운로드/정리 | HuggingFace `wchao0601/m4-sar` 다운로드(Colab) + SAR/Optical 자동 분류 → `trainA/trainB/testA/testB`(dataroot) 생성 |
| 1. 데이터셋(dataroot) | dataroot/실험이름/체크포인트·결과 폴더/gpu_ids 지정 및 스캔 |
| 2. SAR 전처리 | speckle/intensity/clipping/histogram/resize/channel 스텝을 **순서대로 추가·편집**, Before/After 미리보기, 실행, CUT dataroot export |
| 3. 기본 학습 파라미터 | CUT_mode, epochs, batch, lr, beta, load/crop size 등 |
| 4. CUT 파라미터 | netG/netF/normG/gan_mode, NCE 설정, `lambda_grad`/`lambda_color` 구조·색 손실 |
| 5. Attention | none/cbam/coord, reduction, encoder/resblocks/decoder 위치 토글 |
| 6. 학습 실행/모니터링 | `train.py` 실행, 라이브 로그 + epoch/iters/lr/손실 표시, 중단 |
| 7. 추론/테스트 | `test.py` 실행, 변환 결과(fake_B) 갤러리 |

## 설정 저장

- 각 탭의 **저장** 버튼은 모든 값을 `gui_config.json` 에 보존합니다.
- 전처리 설정(폴더/스텝 순서)은 `preproc_config.json` 에 저장됩니다.

## 동작 방식

- 학습/추론 탭은 현재 설정으로 CLI 명령을 만들어 `train.py` / `test.py` 를
  **서브프로세스**로 실행하고, 콘솔 출력을 파싱해 진행 상황을 보여줍니다.
- attention 을 켜면 `nce_layers` 가 자동 보정됩니다(아키텍처가 학습/추론에서
  일치해야 하므로 탭 4/5 설정을 동일하게 유지하세요).
- 추론(CUT, unaligned)은 `dataroot/testA` 와 `dataroot/testB` 가 모두 필요합니다.

전처리 파이프라인 설계는 `docs/README_pipeline.md`, attention 통합은
`docs/ATTENTION_PORT.md` 를 참고하세요.

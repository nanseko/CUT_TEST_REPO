# 손실 그래프 + 추가 Attention (하이브리드 / Self / ECA)

이 문서는 세 가지 추가 기능을 다룬다:
1. **epoch별 손실 그래프** 자동 저장/표시 (`util/loss_plot.py`, GUI 탭 6)
2. **CBAM + Coordinate 하이브리드** attention (`cbam_coord`)
3. **추가 attention**: Self-Attention(non-local), ECA

---

## 1. epoch별 손실 그래프 (D / G / NCE)

### 무엇을 하나
학습이 정상적으로 수렴하는지 **한눈에** 보기 위해, epoch별 평균 손실을 누적해 그래프로 그린다.
- **매 epoch 끝마다** `checkpoints_dir/<name>/loss_curve.png` + `loss_history.csv` 를 자동 갱신.
- 따라서 **400 epoch가 끝나면 최종 그래프가 체크포인트에 그대로 남는다** (요청 사항).
- 학습 **도중에도** 갱신되므로 GUI 탭 6(학습 실행/모니터링)에서 실시간으로 볼 수 있다.

### 그래프 구성
- **위 패널**: D / G / NCE 3대 주요 곡선
  - `D = (D_real + D_fake) / 2` — 판별기 손실. (원본 로그는 D_real·D_fake만 찍으므로 여기서 합성)
  - `G` = 생성기 총 손실
  - `NCE` = PatchNCE 대조손실
- **아래 패널**: 세부 손실 (G_GAN, D_real, D_fake, NCE_Y + 켜져 있으면 G_grad/G_lap/G_coherence/G_color)

### 읽는 법 (건강한 학습 vs 경고)
- **정상**: D_real·D_fake·G_GAN이 비슷한 크기대(대략 0.1~0.5)에서 등락, NCE는 완만히 감소 후 안정.
- **D 붕괴(경고)**: D_real·D_fake → 0, G_GAN → 계속 증가. G가 D를 못 속임.
- **G 붕괴/모드붕괴(경고)**: G_GAN → 0, D가 커짐.
- ⚠️ **손실값 자체는 품질 지표가 아니다**(적대적 균형값). "학습이 망가지지 않았는지" 확인용이며,
  실제 품질은 탭 8(FID/EPI)로 판단할 것.

### 설계 (왜 이렇게)
- train.py/visualizer가 이미 **모든 iteration 손실을 `loss_log.txt`에 파싱 가능한 형식**으로 남긴다.
  → 이 파일을 **단일 진실 소스**로 파싱한다(학습 스텝에 상태를 끼워넣지 않음). 덕분에:
  - GUI·CLI 학습 모두 커버, **이미 진행 중인 학습**에도 적용 가능
  - train.py 수정은 epoch 끝 **한 줄 훅**뿐(`update_loss_plot`, best-effort, 실패해도 학습 안 멈춤)
- **matplotlib은 선택적**: 없으면 PNG는 건너뛰고 **CSV는 항상** 남긴다(메시지로 안내).
  사용자 PC에 matplotlib이 없으면 `pip install matplotlib`.

### 수동 생성
```bash
python -m util.loss_plot <checkpoints_dir> <name>   # loss_log.txt -> loss_curve.png + loss_history.csv
```

---

## 2. CBAM + Coordinate 하이브리드 (`attention_type=cbam_coord`)

**직렬 결합**: 한 삽입 지점에서 **CBAM을 먼저** 적용(채널+공간 재가중) → 그 출력에 **Coordinate
Attention을 이어서** 적용(방향성 H/W 위치 인코딩). 두 attention의 강점을 순차로 쌓는다.

```
x → CBAM(채널·공간) → CoordinateAttention(방향성 위치) → 출력
```

- CBAM의 "무엇이 중요한가(채널)/어디가 중요한가(공간)"에 더해, Coord의 "가로·세로 방향 위치
  일관성"을 얹어 구조적 배치 정보를 강화.
- 파라미터/연산은 두 모듈의 합. `attention_encoder/resblocks/decoder`와 `attention_reduction`을
  기존과 동일하게 사용.

사용: 탭 5에서 `cbam_coord` 선택, 또는 CLI `--attention_type cbam_coord`.

---

## 3. 추가 attention

### Self-Attention (non-local) — `attention_type=self` ★ 구조/형태 보존
이미지 **전역의 모든 픽셀 쌍 관계**를 직접 모델링한다(SAGAN 방식). 건물/선박의 한쪽 끝 픽셀이
반대쪽 끝 정보를 참조할 수 있어 **강체(rigid) 물체의 전체 형태 일관성** 보존에 유리 — 국소적인
CBAM/Coord로는 강제할 수 없는 부분이다.

- **비용**: 메모리 O((H·W)²). **반드시 저해상도 지점(resblocks)에만** 삽입할 것.
  `attention_encoder`/`attention_decoder`(고해상도)에 켜면 매우 무거워진다.
  → 권장: `--attention_type self --attention_resblocks` (encoder/decoder는 끔).
- **초기 항등성**: 학습 가능한 잔차 스케일 `gamma`를 0으로 초기화 → **학습 시작 시점엔 항등 함수**
  (원본과 동작 동일), 학습하면서 서서히 개입. 안정적 도입.

### ECA (Efficient Channel Attention) — `attention_type=eca`
CBAM 채널 attention의 **경량화 버전**. 축소 MLP 대신 1-D conv로 채널 간 국소 관계만 포착 →
**파라미터가 거의 없다**(채널 64에서 CBAM 678 vs ECA 3). 가볍고 안정적이나, 공간/형태 단서는 없다
(순수 채널 재가중).

### 어떤 걸 언제
| type | 특징 | 형태 보존 | 비용 |
|---|---|---|---|
| `cbam` | 채널+공간 (범용) | 중 | 중 |
| `coord` | 방향성 위치 | 중 | 낮음 |
| `eca` | 경량 채널 | 낮음 | 매우 낮음 |
| `self` | 전역 픽셀 관계 | **높음** | 높음(저해상도만) |
| `cbam_coord` | CBAM→Coord 하이브리드 | 중~높음 | 중~높음 |

**형태/구조가 핵심 목표라면**: `self`(resblocks) 또는 `cbam_coord`를 우선 시도.
**어느 게 나은지 모르겠으면**: 탭 10(하이퍼파라미터 자동 탐색)에 6종이 모두 포함되어 있으니
FID/EPI로 자동 비교 가능.

---

## 검증
- `python tests/test_loss_plot.py` — loss_log 파싱, D 합성(=mean(D_real,D_fake)), 헤더 오파싱 방지,
  중간 활성화 손실의 epoch축 정렬, CSV/PNG 생성.
- `python tests/test_attention_port.py` — 6종 attention의 shape 보존, self-attn 초기 항등성,
  ECA 파라미터 경량성, 하이브리드 순서(CBAM→Coord), ResNet/HRNet 통합, backward.
- 실측 엔드투엔드: `cbam_coord` + resblocks로 소형 CUT 학습 → 매 epoch `loss_curve.png`/`loss_history.csv`
  생성 확인(D 합성 열 포함).

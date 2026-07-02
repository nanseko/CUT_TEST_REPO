# 오프라인(사내망)에서 FID용 InceptionV3 사용하기

FID 계산에는 `torchvision` 의 **InceptionV3 ImageNet 사전학습 가중치** 파일 하나가 필요합니다.

- 파일명: `inception_v3_google-0cc3c7bd.pth`  (약 104MB)
- 다운로드 URL: `https://download.pytorch.org/models/inception_v3_google-0cc3c7bd.pth`

인터넷이 되는 곳에서 이 파일을 받아 사내망 PC로 복사한 뒤, **아래 3가지 방법 중 하나**로 두면 코드가 네트워크 없이 자동으로 찾아 사용합니다.

## 1) 가중치 파일 받기 (인터넷 되는 PC에서)

방법 A — 브라우저/`curl` 로 직접 받기:
```bash
curl -L -o inception_v3_google-0cc3c7bd.pth ^
  https://download.pytorch.org/models/inception_v3_google-0cc3c7bd.pth
```

방법 B — torchvision으로 받아 캐시에서 복사:
```bash
python -c "import torchvision; torchvision.models.inception_v3(weights='IMAGENET1K_V1')"
```
받은 파일 위치:
- Windows: `C:\Users\<사용자>\.cache\torch\hub\checkpoints\inception_v3_google-0cc3c7bd.pth`
- Linux/Mac: `~/.cache/torch/hub/checkpoints/inception_v3_google-0cc3c7bd.pth`

## 2) 사내망 PC에 두기 (아래 중 하나)

**(가장 쉬움) 프로젝트 `weights/` 폴더에 두기**
```
CUT_TEST_REPO/
  weights/
    inception_v3_google-0cc3c7bd.pth
```
코드가 `weights/`, `models/`, 실행 폴더 등을 자동 탐색합니다.

**GUI에서 경로 지정**
- "⑧ 전처리 순서 자동 최적화" 탭의 **"InceptionV3 가중치 .pth 경로"** 칸에 파일 전체 경로 입력.

**환경변수로 지정**
```bash
set INCEPTION_WEIGHTS=C:\path\to\inception_v3_google-0cc3c7bd.pth   # Windows
export INCEPTION_WEIGHTS=/path/to/inception_v3_google-0cc3c7bd.pth  # Linux/Mac
```

**torch 캐시에 그대로 두기** (방법 B로 받았다면)
- 사내망 PC의 동일 캐시 경로에 복사하면 torchvision이 다운로드 없이 사용합니다.
  (`%USERPROFILE%\.cache\torch\hub\checkpoints\` 또는 `TORCH_HOME` 환경변수로 위치 변경 가능)

## 3) 탐색 우선순위

코드는 다음 순서로 로컬 가중치를 찾습니다(찾으면 **네트워크 접속 없이** 로드):
1. GUI에 입력한 경로 / `optimize_orders(inception_weights=...)`
2. 환경변수 `INCEPTION_WEIGHTS`
3. `./inception_v3_google-0cc3c7bd.pth`, `./weights/`, `./models/`, 저장소 루트/`weights`/`models`
4. (위에 없으면) torch 캐시 → 없으면 다운로드 시도

최적화 실행 시 로그에 어떤 가중치를 썼는지 표시됩니다:
```
FID용 InceptionV3 로드 중 (device=cuda, 가중치=로컬:.../weights/inception_v3_google-0cc3c7bd.pth) ...
```

## 참고

- 로컬 가중치를 못 찾고 인터넷도 안 되면 FID는 **자동 비활성화**되고 랭킹은 `composite` 로 진행됩니다(오류 없이 계속).
- 가중치 파일은 크므로 git에 커밋하지 마세요(`weights/` 는 배포 시 따로 복사).
- RTX 50xx(Blackwell)는 CUDA 12.8+ 빌드 PyTorch가 필요합니다(별도 안내 참고). FID는 GPU가 있으면 자동으로 GPU에서 계산됩니다.

# YOLOv12n OBB Fine-tuning Project

YOLOv11n OBB 모델의 파인튜닝 방식을 YOLOv12n 모델에 적용하여 Oriented Bounding Box (OBB) 검출 모델을 개발하는 프로젝트입니다.

## 📋 프로젝트 개요

이 프로젝트는 YOLOv12n 기본 검출 모델을 기반으로 DOTAv1 데이터셋을 사용하여 OBB (Oriented Bounding Box) 검출을 위한 파인튜닝을 수행합니다. YOLOv11n OBB에서 사용된 성공적인 학습 방법론을 그대로 채택하여 YOLOv12n에 적용했습니다.

### 주요 특징
- **기반 모델**: YOLOv12n (사전 훈련된 일반 검출 모델)
- **타겟 태스크**: OBB (Oriented Bounding Box) 검출
- **데이터셋**: DOTAv1 (항공 이미지 객체 검출)
- **파인튜닝 방식**: YOLOv11n OBB 검증된 하이퍼파라미터 적용

## 🗂️ 프로젝트 구조

```
Yolov12n_OBB/
├── README.md                           # 프로젝트 문서
├── train_yolov12n_obb_finetune.py     # 메인 학습 스크립트
├── data_dota.yaml                      # DOTA 데이터셋 설정 파일
├── data/                               # 데이터셋 디렉토리
│   └── DOTAv1/                        # DOTA v1.0 데이터셋
│       ├── images/
│       │   ├── train/                 # 훈련 이미지
│       │   ├── val/                   # 검증 이미지
│       │   └── test/                  # 테스트 이미지
│       └── labels/
│           ├── train/                 # 훈련 라벨 (OBB 형식)
│           ├── val/                   # 검증 라벨
│           └── test/                  # 테스트 라벨
└── trained_yolov12n_obb_finetune/     # 학습 결과 디렉토리
    ├── weights/                        # 학습된 모델 가중치
    ├── results.csv                     # 학습 결과 메트릭
    ├── results.png                     # 결과 그래프
    └── ...                            # 기타 결과 파일들
```

## 🔧 환경 설정

### 필수 요구사항
- Python 3.8+
- PyTorch 1.8+
- CUDA 지원 GPU (권장)
- Ultralytics YOLOv5/v8/v11 환경

### 설치
```bash
# Ultralytics 패키지 설치
pip install ultralytics

# 추가 의존성 (필요시)
pip install torch torchvision torchaudio
pip install opencv-python
pip install matplotlib
pip install pandas
```

## 📊 데이터셋 설정

### DOTA v1.0 데이터셋
- **클래스 수**: 15개
- **이미지 크기**: 1024x1024 (학습 시)
- **어노테이션 형식**: OBB (Oriented Bounding Box)

#### 클래스 목록
```
0: plane                1: baseball-diamond      2: bridge
3: ground-track-field   4: small-vehicle         5: large-vehicle
6: ship                 7: tennis-court          8: basketball-court
9: storage-tank         10: soccer-ball-field    11: roundabout
12: harbor              13: swimming-pool        14: helicopter
```

### 데이터셋 준비
1. DOTA v1.0 데이터셋을 다운로드
2. `data/DOTAv1/` 경로에 압축 해제
3. 이미지와 라벨이 올바른 구조로 배치되었는지 확인

## 🚀 학습 실행

### 기본 학습
```bash
python train_yolov12n_obb_finetune.py
```

### 주요 학습 설정

#### 메모리 최적화
- CUDA 메모리 할당 최적화
- 배치 크기: 6 (메모리 효율성)
- 혼합 정밀도 (AMP) 활성화
- 캐시 비활성화로 메모리 절약

#### 학습 하이퍼파라미터
```python
epochs=300                    # 학습 에폭
imgsz=1024                   # 이미지 크기 (DOTA 최적화)
batch=6                      # 배치 크기
patience=50                  # 조기 종료 대기
lr0=0.01                     # 초기 학습률
optimizer='AdamW'            # 옵티마이저
```

#### 데이터 증강 (YOLOv11n OBB 방식 적용)
```python
degrees=180.0               # 회전 증강 (OBB 필수)
scale=0.9                   # 크기 조정
flipud=0.5                  # 상하 반전
fliplr=0.5                  # 좌우 반전
mosaic=1.0                  # 모자이크 증강
mixup=0.15                  # 믹스업 증강
```

#### 손실 함수 가중치
```python
box=7.5                     # 박스 손실
cls=0.5                     # 클래스 손실
dfl=1.5                     # 분포 초점 손실
pose=12.0                   # 자세 손실 (OBB)
```

## 📈 학습 결과

학습 완료 후 `trained_yolov12n_obb_finetune/` 디렉토리에서 확인 가능한 결과:

### 모델 가중치
- `weights/best.pt`: 최고 성능 모델
- `weights/last.pt`: 마지막 에폭 모델

### 평가 메트릭
- `results.csv`: 에폭별 상세 메트릭
- `results.png`: 학습 곡선 그래프
- `confusion_matrix.png`: 혼동 행렬
- `PR_curve.png`: Precision-Recall 곡선

### 시각화 결과
- `train_batch*.jpg`: 학습 배치 샘플
- `val_batch*_pred.jpg`: 검증 예측 결과
- `val_batch*_labels.jpg`: 검증 실제 라벨

## 🎯 모델 사용법

### 학습된 모델 로드
```python
from ultralytics import YOLO

# 최고 성능 모델 로드
model = YOLO('trained_yolov12n_obb_finetune/weights/best.pt')

# 추론 실행
results = model('path/to/your/image.jpg')

# 결과 시각화
results[0].show()
```

### 배치 추론
```python
# 여러 이미지에 대한 추론
results = model(['image1.jpg', 'image2.jpg', 'image3.jpg'])

# 결과 저장
for i, result in enumerate(results):
    result.save(f'result_{i}.jpg')
```

## ⚙️ 파인튜닝 전략

### YOLOv11n OBB 방식 채택
1. **사전 훈련된 가중치 활용**: `yolov12n.pt`에서 시작
2. **OBB 헤드 초기화**: 새로운 OBB 검출 헤드 추가
3. **점진적 학습률**: 웜업과 함께 안정적 수렴
4. **DOTA 특화 증강**: 항공 이미지 특성 고려

### 메모리 최적화 기법
- 확장 가능한 세그먼트 할당
- 보수적 메모리 제한 (70%)
- cuDNN 벤치마킹 비활성화
- 결정론적 연산으로 안정성 확보


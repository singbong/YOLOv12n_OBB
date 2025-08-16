from ultralytics import YOLO
import os
import torch
import gc

def train_yolov12n_obb_finetune():
    # Set CUDA memory allocation configuration to avoid fragmentation
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'
    
    # 강제로 가비지 컬렉션 실행
    gc.collect()
    
    # Clear CUDA cache and optimize memory settings
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(0.7)  # 더 보수적인 메모리 제한
    torch.backends.cudnn.benchmark = False  # Disable cuDNN benchmarking to save memory
    torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for better performance
    torch.backends.cudnn.deterministic = True  # 메모리 사용량 안정화
    
    # Path to the pre-trained YOLOv12n weights
    # IMPORTANT: You need to have yolov12n.pt (standard detection model weights)
    # available. This might be downloadable from the official YOLOv12 source
    # or trained by yourself on a standard detection dataset like COCO.
    pretrained_weights_path = 'yolov12n.pt' # Adjust this path if necessary

    # Load the YOLOv12n-OBB model configuration, this time starting with pre-trained weights
    # The model architecture will be loaded from yolov12n-obb.yaml,
    # and then weights from yolov12n.pt will be loaded.
    # The head will likely be randomly initialized or initialized differently if it's a new OBB head.
    # 수정된 코드 (방법 2)
    model = YOLO(model='ultra/ultralytics/cfg/models/12/yolo12-obb.yaml').load(pretrained_weights_path)

    # Path to the dataset configuration file
    data_config_path = 'data_dota.yaml'



    project_name = 'trained_yolov12n_obb_finetune'
    experiment_name = 'dota_finetune_experiment_optimized'
    # It's often good to use a lower learning rate for fine-tuning
    initial_lr = 0.01  # Slightly higher for fine-tuning to adapt faster

    try:
        model.train(
            data=data_config_path,
            epochs=300,  # 학습 에폭 수
            imgsz=1024, # DOTA OBB용 이미지 크기 유지
            batch=6,    # 배치 크기를 더 줄임 (메모리 오류 방지)
            patience=50, # 조기 종료 대기 에폭
            device=[0], 
            workers=10,  # 워커 수 추가 감소
            project='./',
            name='trained_yolov12n_obb_finetune',
            task='obb',
            cache=False, # 캐시 비활성화로 메모리 절약
            max_det=100, # 최대 검출 객체 수 제한
            nms=True,    # NMS 사용 (메모리 효율성)
            close_mosaic=30, # 더 빨리 모자이크 비활성화
            hsv_h=0.015,  # Ultralytics 기본값. 색상 변화는 일반적인 강인성 향상에 도움
            hsv_s=0.7,    # Ultralytics 기본값
            hsv_v=0.4,    # Ultralytics 기본값
            degrees=180.0, # DOTA OBB 학습에 필수적. 객체의 다양한 회전각 대응
            translate=0.1, # Ultralytics 기본값. 객체의 부분적 가려짐 및 위치 변화에 대한 강인성
            scale=0.9,    # YOLOv11n-OBB DOTA 학습 값 참고. DOTA의 다양한 객체 크기 고려
            shear=0.0,    # 항공 이미지에서는 과도한 전단 변형이 불필요할 수 있음
            perspective=0.0, # 항공 이미지에서는 원근 왜곡이 크지 않으므로 기본값 유지
            flipud=0.5,   # DOTA OBB 학습에 중요. 항공 이미지에서 상하 방향성은 상대적
            fliplr=0.5,   # Ultralytics 기본값. 좌우 대칭성 학습
            mosaic=1.0,   # DOTA의 작은 객체 탐지에 매우 효과적
            mixup=0.15,   # YOLOv11n-OBB DOTA 학습 값 참고
            copy_paste=0.0, # OBB 작업에서는 일반적으로 0.0
            
            # 최적화 설정
            optimizer='AdamW', # AdamW 옵티마이저
            lr0=0.01,     # 초기 학습률
            lrf=0.01,     # 최종 학습률 계수
            momentum=0.937, # SGD 모멘텀
            weight_decay=0.0005, # 가중치 감쇠
            warmup_epochs=5, # 웜업 에폭
            warmup_momentum=0.8, # 웜업 모멘텀
            warmup_bias_lr=0.1, # 웜업 바이어스 학습률
            box=7.5,      # 박스 손실 가중치
            cls=0.5,      # 클래스 손실 가중치
            dfl=1.5,      # 분포 초점 손실 가중치
            pose=12.0,    # 자세 손실 가중치
            kobj=1.0,     # 키포인트 객체 손실 가중치
            label_smoothing=0.0, # 레이블 스무딩 값
            iou=0.65,     # NMS IoU 임계값 낮춤 (메모리 사용량 감소)
            amp=True,     # 혼합 정밀도 활성화 (메모리 효율)
            half=False,   # 하프 정밀도 비활성화 (안정성)
            val=True,     # 검증 데이터셋 평가 활성화
        )
        print("Fine-tuning finished successfully.")
        print(f"Results saved to: {project_name}/{experiment_name}")

    except Exception as e:
        print(f"An error occurred during fine-tuning: {e}")
        print("Please ensure that your model YAML (yolov12n-obb.yaml) and data YAML (data_dota.yaml) are correctly configured.")
        print(f"Ensure the pre-trained weights '{pretrained_weights_path}' exist and are compatible.")
        print("Also, check if the DOTA dataset is correctly placed and paths in data_dota.yaml are accurate.")
        print("Ensure that the Ultralytics environment is set up correctly and all dependencies are installed.")
        print("For GPU memory issues, try reducing batch_size or img_size.")

if __name__ == '__main__':
    # Example: python train_yolov12n_obb_finetune.py
    train_yolov12n_obb_finetune()

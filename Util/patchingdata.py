import pandas as pd
import numpy as np
import os
import glob

def create_overlapping_patches(csv_file_path, patch_size=1024, overlap_ratio=0.5, output_base_dir="patched_data"):
    """
    단일 CSV 파일을 overlapping 방식으로 1024 크기 패치로 분할
    Normal vs Abnormal으로 분류하여 저장
    """
    # CSV 파일 로드
    df = pd.read_csv(csv_file_path)
    total_samples = len(df)
    
    # 파일명에서 레코드 번호 추출
    filename = os.path.basename(csv_file_path)
    record_name = filename.split('_')[2]
    
    # 패칭 파라미터 계산
    stride = int(patch_size * (1 - overlap_ratio))
    num_patches = max(1, (total_samples - patch_size) // stride + 1)
    
    # 출력 폴더 생성
    normal_dir = os.path.join(output_base_dir, "Normal_beat")
    abnormal_dir = os.path.join(output_base_dir, "Abnormal_beat")
    os.makedirs(normal_dir, exist_ok=True)
    os.makedirs(abnormal_dir, exist_ok=True)
    
    # 패치 생성 및 분류
    normal_patches = 0
    abnormal_patches = 0
    
    for patch_idx in range(num_patches):
        start_idx = patch_idx * stride
        end_idx = min(start_idx + patch_size, total_samples)
        
        # 마지막 패치가 너무 작으면 스킵
        if end_idx - start_idx < patch_size * 0.8:
            continue
        
        # 패치 데이터 추출
        patch_data = df.iloc[start_idx:end_idx].copy()
        
        # 인덱스 재설정
        patch_data['sample_index'] = range(len(patch_data))
        
        # 클래스 분포 계산
        class_counts = patch_data['beat_label'].value_counts().to_dict()
        
        # Normal vs Abnormal 분류
        abnormal_classes = ['APB', 'PVC', 'LBBB', 'RBBB']
        abnormal_samples = sum(class_counts.get(cls, 0) for cls in abnormal_classes)
        
        # abnormal beat가 하나라도 있으면 Abnormal로 분류
        is_abnormal = abnormal_samples > 0
        
        # 파일 저장
        if is_abnormal:
            abnormal_patches += 1
            patch_filename = f"{record_name}_{abnormal_patches}.csv"
            patch_filepath = os.path.join(abnormal_dir, patch_filename)
        else:
            normal_patches += 1
            patch_filename = f"{record_name}_{normal_patches}.csv"
            patch_filepath = os.path.join(normal_dir, patch_filename)
        
        patch_data.to_csv(patch_filepath, index=False)

def process_all_csv_files(input_dir="./", output_base_dir="patched_data", patch_size=1024, overlap_ratio=0.5):
    """
    폴더 내 모든 MIT-BIH CSV 파일을 패칭 처리
    """
    # CSV 파일 찾기
    csv_pattern = os.path.join(input_dir, "mit_bih_*_6class_full.csv")
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        return
    
    csv_files.sort()
    
    # 출력 폴더 생성
    os.makedirs(output_base_dir, exist_ok=True)
    
    # 전체 처리
    for csv_file in csv_files:
        create_overlapping_patches(
            csv_file_path=csv_file,
            patch_size=patch_size,
            overlap_ratio=overlap_ratio,
            output_base_dir=output_base_dir
        )

# 실행
if __name__ == "__main__":
    INPUT_DIR = r"C:\Users\Ahhyun\Desktop\Workplace\Code\ECGClassification"
    OUTPUT_DIR = "patched_data"
    PATCH_SIZE = 1024
    OVERLAP_RATIO = 0.5
    
    process_all_csv_files(
        input_dir=INPUT_DIR,
        output_base_dir=OUTPUT_DIR,
        patch_size=PATCH_SIZE,
        overlap_ratio=OVERLAP_RATIO
    )
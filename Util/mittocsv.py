import numpy as np
import pandas as pd
import wfdb
from sklearn import preprocessing
import os
import glob

# 데이터 경로 설정 (실제 경로로 수정하세요)
data_path = 'C:/Users/Ahhyun/Desktop/Workplace/Dataset/ECG/mitbih/rawdata/'  # MIT-BIH 데이터가 있는 폴더 경로

def find_mit_records(data_path):
    """
    폴더에서 MIT-BIH 레코드 파일들 찾기
    """
    atr_files = glob.glob(os.path.join(data_path, "*.atr"))
    record_numbers = []
    
    for atr_file in atr_files:
        filename = os.path.basename(atr_file)
        record_num = filename.replace('.atr', '')
        record_numbers.append(record_num)
    
    try:
        record_numbers.sort(key=lambda x: int(x) if x.isdigit() else float('inf'))
    except:
        record_numbers.sort()
    
    return record_numbers

def classify_to_6_classes(symbol):
    """
    MIT-BIH annotation 기호를 순수 6개 클래스로 분류
    
    5개 주요 beat 클래스:
    - Normal: N만
    - APB: A, a, S, J  
    - PVC: V, r, E
    - LBBB: L
    - RBBB: R
    - 그 외는 baseline
    """
    
    
    class_mapping = {
        'N': 'Normal',
        'A': 'APB', 'a': 'APB', 'S': 'APB', 'J': 'APB',
        'V': 'PVC', 'r': 'PVC', 'E': 'PVC',
        'L': 'LBBB',
        'R': 'RBBB'
    }
    
    return class_mapping.get(symbol, None)

def create_single_record_csv(record_name, data_path, beat_length=260, before_r=99):
    """
    단일 레코드에 대한 6-class CSV 파일 생성
    """
    # 데이터 로드
    record = wfdb.rdrecord(data_path + record_name, smooth_frames=True)
    annotation = wfdb.rdann(data_path + record_name, 'atr')
    
    # 신호 전처리
    ecg_signal = record.p_signal[:, 0]
    processed_signal = preprocessing.scale(np.nan_to_num(ecg_signal))
    total_samples = len(processed_signal)
    
    # 5개 주요 클래스 annotation만 필터링
    main_rpeaks = []
    main_6class_labels = []
    
    for i, r_sample in enumerate(annotation.sample):
        original_symbol = annotation.symbol[i]
        class_label = classify_to_6_classes(original_symbol)
        
        if class_label is not None:
            main_rpeaks.append(r_sample)
            main_6class_labels.append(class_label)
    
    # 픽셀 레벨 라벨링
    sample_labels = ['Baseline'] * total_samples
    
    for r_peak, class_label in zip(main_rpeaks, main_6class_labels):
        beat_start = r_peak - before_r
        beat_end = r_peak + (beat_length - before_r)
        
        actual_start = max(0, beat_start)
        actual_end = min(total_samples, beat_end)
        
        for sample_idx in range(actual_start, actual_end):
            sample_labels[sample_idx] = class_label
    
    # CSV 파일 생성
    csv_filename = f'mit_bih_{record_name}_6class_full.csv'
    chunk_size = 50000
    chunks_total = (total_samples + chunk_size - 1) // chunk_size
    
    # CSV 헤더 작성
    with open(csv_filename, 'w') as f:
        f.write('sample_index,signal_value,beat_label,time_seconds\n')
    
    # 청크별 처리
    for chunk_idx in range(chunks_total):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, total_samples)
        
        chunk_data = []
        for i in range(start_idx, end_idx):
            chunk_data.append({
                'sample_index': i,
                'signal_value': processed_signal[i],
                'beat_label': sample_labels[i],
                'time_seconds': i / record.fs
            })
        
        chunk_df = pd.DataFrame(chunk_data)
        chunk_df.to_csv(csv_filename, mode='a', header=False, index=False)

def create_all_csv_files(data_path, max_records=None):
    """
    폴더 내 모든 MIT-BIH 데이터에 대해 6-class CSV 파일 생성
    """
    record_numbers = find_mit_records(data_path)
    
    if not record_numbers:
        return
    
    if max_records:
        record_numbers = record_numbers[:max_records]
    
    for record_name in record_numbers:
        create_single_record_csv(record_name, data_path)

# 실행
if __name__ == "__main__":
    create_all_csv_files(
        data_path=data_path,
        max_records=None
    )
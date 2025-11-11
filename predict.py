import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import glob
import pandas as pd

def get_segments(mask):
    """
    연속된 True 구간 찾기
    """
    segments = []
    start = None
    
    for i in range(len(mask)):
        if mask[i] and start is None:
            start = i
        elif not mask[i] and start is not None:
            segments.append((start, i))
            start = None
    
    if start is not None:
        segments.append((start, len(mask)))
    
    return segments

def visualize_ecg_segmentation(signal, true_mask, pred_mask, 
                               save_path='ecg_segmentation_result.png',
                               sample_title='ECG Segmentation Result',
                               figsize=(16, 8)):
    """
    ECG 신호와 세그멘테이션 결과를 시각화
    """
    # Tensor를 numpy로 변환
    if torch.is_tensor(signal):
        signal = signal.cpu().numpy()
    if torch.is_tensor(true_mask):
        true_mask = true_mask.cpu().numpy()
    if torch.is_tensor(pred_mask):
        pred_mask = pred_mask.cpu().numpy()
    
    # Shape 조정
    if len(signal.shape) == 2:
        signal = signal.squeeze()
    
    # 클래스 정보
    class_names = ['Baseline', 'Normal', 'APB', 'PVC', 'LBBB', 'RBBB']
    class_colors = {
        0: '#808080',  # Baseline - Gray
        1: '#2ecc71',  # Normal - Green
        2: '#3498db',  # APB - Blue
        3: '#f39c12',  # PVC - Orange/Yellow
        4: '#00d4ff',  # LBBB - Light Blue
        5: '#e74c3c'   # RBBB - Red
    }
    
    # Figure 생성
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1], hspace=0.3)
    
    # 1. ECG 신호 플롯
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(signal, 'k-', linewidth=0.8, alpha=0.9)
    ax1.set_ylabel('Amplitude', fontsize=12, fontweight='bold')
    ax1.set_title(sample_title, fontsize=14, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(0, len(signal))
    
    # 2. Ground Truth 세그멘테이션
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(signal, 'k-', linewidth=0.5, alpha=0.3)
    
    for class_idx in range(6):
        mask = (true_mask == class_idx)
        if np.any(mask):
            segments = get_segments(mask)
            for start, end in segments:
                ax2.axvspan(start, end, color=class_colors[class_idx], alpha=0.6)
                mid = (start + end) // 2
                if end - start > 30:
                    ax2.text(mid, np.max(signal) * 0.5, class_names[class_idx][0], 
                            ha='center', va='center', fontsize=9, fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    ax2.set_ylabel('Amplitude', fontsize=12, fontweight='bold')
    ax2.set_title('Ground Truth Segmentation', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim(0, len(signal))
    
    # 3. Prediction 세그멘테이션
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(signal, 'k-', linewidth=0.5, alpha=0.3)
    
    for class_idx in range(6):
        mask = (pred_mask == class_idx)
        if np.any(mask):
            segments = get_segments(mask)
            for start, end in segments:
                ax3.axvspan(start, end, color=class_colors[class_idx], alpha=0.6)
                mid = (start + end) // 2
                if end - start > 30:
                    ax3.text(mid, np.max(signal) * 0.5, class_names[class_idx][0], 
                            ha='center', va='center', fontsize=9, fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    ax3.set_ylabel('Amplitude', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Sample Index', fontsize=12, fontweight='bold')
    ax3.set_title('Predicted Segmentation', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_xlim(0, len(signal))
    
    # 범례 생성
    legend_elements = [mpatches.Patch(facecolor=class_colors[i], alpha=0.6, 
                                     edgecolor='black', label=class_names[i])
                      for i in range(6)]
    fig.legend(handles=legend_elements, loc='upper center', 
              bbox_to_anchor=(0.5, 0.98), ncol=6, fontsize=11, framealpha=0.9)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def visualize_from_csv_folder(model, csv_folder, device='cuda', output_dir='visualization_results'):
    """
    폴더 내 모든 CSV 파일들을 읽어서 시각화
    """
    os.makedirs(output_dir, exist_ok=True)
    
    model.to(device)
    model.eval()
    
    # CSV 파일 찾기
    csv_files = glob.glob(os.path.join(csv_folder, '*.csv'))
    
    # 클래스 매핑
    class_to_idx = {
        'Baseline': 0,
        'Normal': 1,
        'APB': 2,
        'PVC': 3,
        'LBBB': 4,
        'RBBB': 5
    }
    
    with torch.no_grad():
        for idx, csv_file in enumerate(csv_files):
            # CSV 로드
            df = pd.read_csv(csv_file)
            
            # 신호와 라벨 추출
            signal = df['signal_value'].values.astype(np.float32)
            labels = df['beat_label'].map(class_to_idx).values
            
            # Tensor 변환
            signal_tensor = torch.FloatTensor(signal).unsqueeze(0).unsqueeze(0).to(device)
            
            # 예측
            output = model(signal_tensor)
            prediction = output.argmax(dim=1).squeeze()
            
            # 시각화
            filename = os.path.basename(csv_file).replace('.csv', '.png')
            save_path = os.path.join(output_dir, filename)
            
            visualize_ecg_segmentation(
                signal=signal,
                true_mask=labels,
                pred_mask=prediction.cpu().numpy(),
                save_path=save_path,
                sample_title=f'ECG Segmentation - {os.path.basename(csv_file)}',
                figsize=(16, 8)
            )


if __name__ == "__main__":
    from Model.architecture import HybridNAFNet, HybridNAFNet_small
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = HybridNAFNet_small()
    
    # 체크포인트 로드
    checkpoint_path = r"checkpoint_136.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # CSV 폴더 경로
    csv_folder = r"C:\Users\Ahhyun\Desktop\Workplace\Code\ECGClassification\patched_data\Abnormal_beat"
    
    #샘플 시각화 후 저장
    visualize_from_csv_folder(
        model=model,
        csv_folder=csv_folder,
        device=device,
        num_samples=4,
        output_dir='visualization_results'
    )
    
import torch
import os
from Model.architecture import HybridNAFNet, HybridNAFNet_small


def save_parameters_to_files(model, output_dir="Parameters"):
    """
    PyTorch 모델의 파라미터를 주어진 디렉토리 구조로 저장합니다.
    특별히 qkv_dwconv와 conv2 모듈의 경우 4차원 텐서를 3차원으로 분리하여 저장합니다.
    
    Args:
        model: PyTorch 모델
        output_dir: 파라미터를 저장할 기본 디렉토리
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for name, param in model.named_parameters():
        parts = name.split('.')
        is_dwconv = any(part in ["qkv_dwconv", "conv2"] for part in parts)
        
        # qkv_dwconv 또는 conv2의 weight (4차원 텐서)
        if is_dwconv and len(param.shape) == 4 and parts[-1] == "weight":
            num_channels = param.shape[0]
            
            for i in range(num_channels):
                folder_path = output_dir
                for part in parts[:-1]:
                    folder_path = os.path.join(folder_path, part)
                
                channel_folder = os.path.join(folder_path, str(i))
                os.makedirs(channel_folder, exist_ok=True)
                
                file_path = os.path.join(channel_folder, f"{parts[-1]}.txt")
                
                with open(file_path, 'w') as f:
                    tensor_slice = param[i].detach().cpu().numpy().flatten()
                    for value in tensor_slice:
                        f.write(f"{value}\n")
            
            continue
            
        # qkv_dwconv 또는 conv2의 bias
        if is_dwconv and parts[-1] == "bias":
            num_channels = param.shape[0]
            
            for i in range(num_channels):
                folder_path = output_dir
                for part in parts[:-1]:
                    folder_path = os.path.join(folder_path, part)
                
                channel_folder = os.path.join(folder_path, str(i))
                os.makedirs(channel_folder, exist_ok=True)
                
                file_path = os.path.join(channel_folder, f"{parts[-1]}.txt")
                
                with open(file_path, 'w') as f:
                    value = param[i].item()
                    f.write(f"{value}\n")
            
            continue
        
        # 일반 파라미터
        folder_path = output_dir
        for part in parts[:-1]:
            folder_path = os.path.join(folder_path, part)
            os.makedirs(folder_path, exist_ok=True)
        
        file_name = f"{parts[-1]}.txt"
        file_path = os.path.join(folder_path, file_name)
        
        with open(file_path, 'w') as f:
            flat_tensor = param.detach().cpu().numpy().flatten()
            for value in flat_tensor:
                f.write(f"{value}\n")



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridNAFNet_small() 

    checkpoint = torch.load("model.pth", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    
    save_parameters_to_files(model)
import torch
import os

file_path = 'weights/detector.pt'

if not os.path.exists(file_path):
    print(f"'{file_path}' not exist")
else:
    data = torch.load(file_path, map_location='cpu')
    print(f"weight type: {type(data)}")

    if isinstance(data, dict):
        print("\n--- Dict ---")
        print(f"Keys: {data.keys()}")

        for i, (key, value) in enumerate(data.items()):
            if hasattr(value, 'shape'):
                print(f"  键 '{key}': 张量形状 {value.shape}")
            else:
                print(f"  键 '{key}': 类型 {type(value)}")
    
    elif isinstance(data, torch.Tensor):
        print("\n--- Tsensor ---")
        print(f"Shape: {data.shape}")
        print(f"Dtype: {data.dtype}")
    
    else:
        print("\n--- Other ---")
        print(data)

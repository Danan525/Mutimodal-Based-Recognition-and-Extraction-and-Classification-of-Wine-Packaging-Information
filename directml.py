import torch

# 检查是否有 DirectML 后端
if hasattr(torch.backends, 'directml') and torch.backends.directml.is_available():
    print("DirectML is available")
else:
    print("DirectML is not available")
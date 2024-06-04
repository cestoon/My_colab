import torch

# 清理CUDA缓存
torch.cuda.empty_cache()

# 运行垃圾收集器
import gc
gc.collect()
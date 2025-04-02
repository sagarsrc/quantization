#%%
import torch
import bitsandbytes.functional as F
#%%

A = torch.randn(16, 16)

#%%
print(F.quantize_fp4(A))
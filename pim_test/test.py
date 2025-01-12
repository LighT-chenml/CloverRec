import numpy as np

import torch

import pim_module

ln = [100] * 10
m = 64
W_list = []
for n in ln:
    low = -np.sqrt(1 / n)
    high = np.sqrt(1 / n)
    W = low + torch.rand(n, m ,dtype=torch.float32) * (high - low)
    W_list.append(W)
content = np.array(W_list).tobytes()

q = np.random.randint(0, len(content), size=10)
print(q)

storage = pim_module.PIMEmbStorage()
storage.initialize(content)

ret = storage.query_sum(len(q), q.tobytes())
print(ret)


# numpy
import numpy as np

# pytorch
import torch

import time

class EmbStorage():
    def __init__(self, m, ln):
        self.emb_l = []
        self.P_l = []
        for i in range(0, ln.size):
            n = ln[i]
            # initialize embeddings
            # W = np.random.uniform(
            #     low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, m)
            # ).astype(np.float32)
            
            low = -np.sqrt(1 / n)
            high = np.sqrt(1 / n)
            W = low + torch.rand(n, m ,dtype=torch.float32) * (high - low)
            
            self.emb_l.append(W)
            
            P = np.array([x for x in range(ln[i])])
            np.random.shuffle(P)
            self.P_l.append(torch.tensor(P))
        
        torch.set_num_threads(16)
    
    def apply_emb(self, lS_o, lS_i):
        # WARNING: notice that we are processing the batch at once. We implicitly
        # assume that the data is laid out such that:
        # 1. each embedding is indexed with a group of sparse indices,
        #   corresponding to a single lookup
        # 2. for each embedding the lookups are further organized into a batch
        # 3. for a list of embedding tables there is a list of batched lookups

        # num_threads = torch.get_num_threads()
        # print(f"num threads: {num_threads}")

        total_lookup_time = 0
        total_cal_time = 0

        ly = []
        for k, sparse_index_group_batch in enumerate(lS_i):
            sparse_offset_group_batch = lS_o[k]

            # embedding lookup
            # We are using EmbeddingBag, which implicitly uses sum operator.
            # The embeddings are represented as tall matrices, with sum
            # happening vertically across 0 axis, resulting in a row vector
            # E = emb_l[k]
            
            E = self.emb_l[k]
            batch_size = len(sparse_offset_group_batch)
            evs = []
            
            # start_time = time.time()
            
            sparse_index_group_batch = self.P_l[k][sparse_index_group_batch]
            
            # end_time = time.time()
            # total_time = end_time - start_time
            # total_time *= 1000
            # print("permutation time (ms): " + f"{total_time}")
            
            for i in range(batch_size):
                
                # start_time = time.time()
                
                start = sparse_offset_group_batch[i]
                end = sparse_offset_group_batch[i + 1] if i + 1 < batch_size else len(sparse_index_group_batch)
                
                # end_time = time.time()
                
                # start_time = time.time()
                
                # print(sparse_index_group_batch[start:end])
                
                ev = E[sparse_index_group_batch[start:end]]
                
                # end_time = time.time()
                # total_time = end_time - start_time
                # total_time *= 1000
                # total_lookup_time += total_time
                
                # start_time = time.time()
                
                # mode = "sum"
                evs.append(ev.sum(dim=0))
                
                # end_time = time.time()
                # total_time = end_time - start_time
                # total_time *= 1000
                # total_cal_time += total_time
            
            # start_time = time.time()
            
            V = torch.tensor(np.array(evs))
            ly.append(V)
            
            # end_time = time.time()
            # total_time = end_time - start_time
            # total_time *= 1000
            # print("total_time (ms): " + f"{total_time * len(lS_i)}")
            
        # print("total_lookup_time (ms): " + f"{total_lookup_time}")
        # print("total_cal_time (ms): " + f"{total_cal_time}")

        return ly
# numpy
import numpy as np

# pytorch
import torch

class EmbStorage():
    def __init__(self, m, ln):
        self.emb_l = []
        for i in range(0, ln.size):
            n = ln[i]
            # initialize embeddings
            W = np.random.uniform(
                low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, m)
            ).astype(np.float32)
            self.emb_l.append(W)
    
    def apply_emb(self, lS_o, lS_i):
        # WARNING: notice that we are processing the batch at once. We implicitly
        # assume that the data is laid out such that:
        # 1. each embedding is indexed with a group of sparse indices,
        #   corresponding to a single lookup
        # 2. for each embedding the lookups are further organized into a batch
        # 3. for a list of embedding tables there is a list of batched lookups

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
            
            for i in range(batch_size):
                if i != batch_size - 1:
                    index = sparse_index_group_batch[sparse_offset_group_batch[i]:sparse_offset_group_batch[i+1]]
                else:
                    index = sparse_index_group_batch[sparse_offset_group_batch[i]:]
                ev = E[index]
                
                # mode = "sum"
                ret = sum(ev)
                evs.append(ret)
            
            V = torch.tensor(np.array(evs))
            ly.append(V)

        # print(ly)
        return ly
# numpy
import numpy as np

# pytorch
import torch

import math

# RDMA
from connection import SKT, CM

from pyverbs.addr import AH, AHAttr, GlobalRoute
from pyverbs.cq import CQ
from pyverbs.device import Context
from pyverbs.enums import *
from pyverbs.mr import MR
from pyverbs.pd import PD
from pyverbs.qp import QP, QPCap, QPInitAttr, QPAttr
from pyverbs.wr import SGE, RecvWR, SendWR

import time

class EmbClient:
    def __init__(self):
        self.CLIENT_RECV_WR = 3
        self.CLIENT_SEND_WR = 4
    
    def read_mr(self, length, offset):
        return self.mr.read(length, offset)

    def connect(self, server_port, server_ip, mr_size, emb_size, wr_capacity):
        self.conn = CM(server_port, server_ip)

        print("New connection...")

        ctx = Context(name='mlx5_0')
        self.pd = PD(ctx)
        self.cq = CQ(ctx, wr_capacity)
        self.wr_capacity = wr_capacity

        cap = QPCap(max_send_wr=wr_capacity, max_recv_wr=wr_capacity, max_send_sge=1, max_recv_sge=1, max_inline_data=0)
        qp_init_attr = QPInitAttr(qp_type=IBV_QPT_RC, scq=self.cq, rcq=self.cq, cap=cap, sq_sig_all=True)
        self.qp = QP(self.pd, qp_init_attr)

        gid = ctx.query_gid(port_num=1, index=1)
        lid = ctx.query_port(port_num=1).lid

        # Handshake to exchange information such as QP Number
        remote_info = self.conn.handshake(gid=gid, lid=lid, qpn=self.qp.qp_num)

        ah_attr = AHAttr(dlid=remote_info['lid'], is_global=0, port_num=1)

        qa = QPAttr()
        qa.ah_attr = ah_attr
        qa.dest_qp_num = remote_info['qpn']
        qa.path_mtu = 4
        qa.max_rd_atomic = 1
        qa.max_dest_rd_atomic = 1
        qa.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE

        self.qp.to_rts(qa)
        
        self.mr = MR(self.pd, mr_size, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ)
        
        self.remote_info = self.conn.handshake(addr=self.mr.buf, rkey=self.mr.rkey)
        self.remote_rkey = self.remote_info['rkey']
        self.remote_addr = self.remote_info['addr']

    def send_request(self, offset_list, emb_size):
        
        offset_list = np.array(offset_list)
        n = len(offset_list)
        idx = 0
        ret = bytearray()
        while idx < n:
            num = min(self.wr_capacity, n - idx)
            
            for i, offset in enumerate(offset_list[idx : idx + num]):
                sgl = [SGE(self.mr.buf + i * emb_size, emb_size, self.mr.lkey)]
                wr = SendWR(self.CLIENT_SEND_WR, opcode=IBV_WR_RDMA_READ, num_sge=1, sg=sgl)
                wr.set_wr_rdma(self.remote_rkey, self.remote_addr + offset)

                self.qp.post_send(wr)

            wc_num, wc_list = self.cq.poll(num_entries=num)
        
            ret.extend(bytearray(self.read_mr(num * emb_size, 0)))
            
            idx += num
            
        return bytes(ret)

    def close(self):
        self.conn.close()

class EmbStorage():
    def __init__(self, m, ln, num_indices_per_lookup, batch_size, emb_pool_ip, emb_pool_port, wr_capacity):
        self.m = m
        self.ln = ln
        self.transmission_time = []
        self.ev_lookup_time = []
        
        self.client = EmbClient()
        mr_size = len(ln) * num_indices_per_lookup * batch_size * m * 4
        
        flag = 0
        while flag != 2:
            try:
                self.client.connect(emb_pool_port, emb_pool_ip, mr_size, m * 4, wr_capacity)
                flag = 2
            except:
                if flag == 0:
                    print("waiting for emb pool")
                    flag = 1
                    
        print("EmbStorge Finish Init!")
    
    def apply_emb(self, lS_o, lS_i):
        # WARNING: notice that we are processing the batch at once. We implicitly
        # assume that the data is laid out such that:
        # 1. each embedding is indexed with a group of sparse indices,
        #   corresponding to a single lookup
        # 2. for each embedding the lookups are further organized into a batch
        # 3. for a list of embedding tables there is a list of batched lookups
    
        start_time_g = time.time()
    
        emb_size = self.m * 4
    
        offset_list = []
        table_offset = 0
        for k, sparse_index_group_batch in enumerate(lS_i):
            offset_list.append(torch.add(sparse_index_group_batch, table_offset))
            table_offset += self.ln[k]
        offset_list = torch.cat(offset_list,dim=0) * emb_size

        start_time = time.time()

        evs_bytes = self.client.send_request(offset_list, emb_size)
        
        end_time = time.time()
        total_time = end_time - start_time
        total_time *= 1000
        print("transmission time (ms): " + f"{total_time}")
        self.transmission_time.append(total_time)
        
        
        # start_time = time.time()
        
        evs = torch.tensor(np.frombuffer(evs_bytes, dtype=np.float32)).view(int(len(evs_bytes) / emb_size), self.m)
        
        # end_time = time.time()
        # total_time = end_time - start_time
        # total_time *= 1000
        # print("convertion time (ms): " + f"{total_time}")
        
        # end_time_g = time.time()
        # total_time = end_time_g - start_time_g
        # total_time *= 1000
        # print("end to end time (ms): " + f"{total_time}")
        
        total_lookup_time = 0
        total_sum_time = 0
        
        # start_time_g = time.time()
        
        ly = []
        ev_offset = 0
        for k, sparse_index_group_batch in enumerate(lS_i):
            sparse_offset_group_batch = lS_o[k]
            
            batch_size = len(sparse_offset_group_batch)
            ev_batch = []
            
            for i in range(batch_size):
                
                start = sparse_offset_group_batch[i]
                end = sparse_offset_group_batch[i + 1] if i + 1 < batch_size else len(sparse_index_group_batch)
                
                # start_time = time.time()
                
                ev = evs[start + ev_offset : end + ev_offset]
                
                # end_time = time.time()
                # total_time = end_time - start_time
                # total_time *= 1000
                # total_lookup_time += total_time
                
                # start_time = time.time()
                
                # mode = "sum"
                ev_batch.append(ev.sum(dim=0))
                
                # end_time = time.time()
                # total_time = end_time - start_time
                # total_time *= 1000
                
                # total_sum_time += total_time
            
            V = torch.tensor(np.array(ev_batch))
            ly.append(V)
            ev_offset += len(sparse_index_group_batch)
            
        end_time_g = time.time()
        total_time = end_time_g - start_time_g
        total_time *= 1000
        print("end to end time (ms): " + f"{total_time}")
        self.ev_lookup_time.append(total_time)

        # print("lookup time (ms): " + f"{total_lookup_time}")
        # print("sum time (ms): " + f"{total_sum_time}")
        
        return ly
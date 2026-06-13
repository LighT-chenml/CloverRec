# numpy
import numpy as np

# pytorch
import torch

import math

# RDMA
import pickle
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

# client cache
from client_cache import ClientCache


def ensure_mr_capacity(mr, payload_len, label):
    required = payload_len + 8
    if required > mr.length:
        raise RuntimeError(
            f"{label} payload requires {required} bytes, "
            f"but RDMA MR is {mr.length} bytes. Increase --rdma-mr-size-mb."
        )


class EmbClient:
    def __init__(self):
        self.CLIENT_RECV_WR = 3
        self.CLIENT_SEND_WR = 4
    
    def read_mr(self, length, offset):
        return self.mr.read(length, offset)

    def send_sgl(self, payload_len):
        return [SGE(self.mr.buf, payload_len + 8, self.mr.lkey)]

    def connect(self, server_port, server_ip, wr_capacity, mr_size_mb):
        self.conn = SKT(server_port, server_ip)
        self.conn.handshake()

        print("New connection...")

        ctx = Context(name='mlx5_0')
        self.pd = PD(ctx)
        self.cq = CQ(ctx, wr_capacity)

        cap = QPCap(max_send_wr=wr_capacity, max_recv_wr=wr_capacity, max_send_sge=1, max_recv_sge=1, max_inline_data=0)
        qp_init_attr = QPInitAttr(qp_type=IBV_QPT_RC, scq=self.cq, rcq=self.cq, cap=cap, sq_sig_all=True)
        self.qp = QP(self.pd, qp_init_attr)

        gid = ctx.query_gid(1, 1)
        lid = ctx.query_port(1).lid

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
        self.conn.handshake()
        
        mr_size = mr_size_mb * 1024 * 1024
        self.mr = MR(self.pd, mr_size, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ)
        self.sgl = [SGE(self.mr.buf, self.mr.length, self.mr.lkey)]

        self.remote_info = self.conn.handshake(addr=self.mr.buf, rkey=self.mr.rkey)

    def send_request(self, input_data):
        self.conn.handshake()

        # prepare request
        header = {}
        request = pickle.dumps({'header': header, 'data': input_data})
        request_len = len(request)
        ensure_mr_capacity(self.mr, request_len, "emb_pool request")
        self.mr.write(request_len.to_bytes(8, 'little'), 8, 0)
        self.mr.write(request, request_len, 8)

        # check server recv ready
        self.conn.handshake()

        wr = SendWR(self.CLIENT_SEND_WR, opcode=IBV_WR_SEND, num_sge=1, sg=self.send_sgl(request_len))
        self.qp.post_send(wr)

        # check client send ready
        self.conn.handshake()

        wc_num, wc_list = self.cq.poll()

        wr = RecvWR(self.CLIENT_RECV_WR, num_sge=1, sg=self.sgl)
        self.qp.post_recv(wr)

        # check client recv ready
        self.conn.handshake()

        # check server send ready
        self.conn.handshake()

        wc_num, wc_list = self.cq.poll()

        # return response
        response_len = int.from_bytes(self.read_mr(8, 0), 'little')
        ensure_mr_capacity(self.mr, response_len, "emb_pool response")
        response = pickle.loads(self.read_mr(response_len, 8))
        return response

    def close(self):
        self.conn.close()

class EmbStorage():
    def __init__(self, m, ln, num_indices_per_lookup, batch_size, emb_pool_ip, emb_pool_port, wr_capacity, mr_size_mb):
        self.m = m
        self.ln = ln

        self.CPU_cal_time = []
        
        self.client = EmbClient()

        self.client_cache = ClientCache()
        self.client_cache.initialize(m, ln)
        
        flag = 0
        while flag != 2:
            try:
                self.client.connect(emb_pool_port, emb_pool_ip, wr_capacity, mr_size_mb)
                flag = 2
            except:
                if flag == 0:
                    print("waiting for emb pool")
                    flag = 1
        
        torch.set_num_threads(16)

        print("EmbStorge Finish Init!")
    
    def apply_emb(self, lS_o, lS_i):
        # WARNING: notice that we are processing the batch at once. We implicitly
        # assume that the data is laid out such that:
        # 1. each embedding is indexed with a group of sparse indices,
        #   corresponding to a single lookup
        # 2. for each embedding the lookups are further organized into a batch
        # 3. for a list of embedding tables there is a list of batched lookups

        cache_ly, lS_o, lS_i = self.client_cache.apply_emb(np.array(lS_o), np.array(lS_i))
        cache_ly = torch.tensor(cache_ly)
        lS_o = torch.tensor(lS_o)
        lS_i = torch.tensor(lS_i)

        ret = self.client.send_request({'lS_o': lS_o, 'lS_i': lS_i})

        remote_evs = torch.tensor(ret['data'])
        remote_offsets = torch.tensor(ret['offset'])
        to_cache_keys = ret['to_cache_keys']
        to_cache_values = ret['to_cache_values']
        self.client_cache.update_cache(to_cache_keys, to_cache_values)

        start_time = time.time()

        ly = []
        for k, sparse_index_group_batch in enumerate(remote_offsets):
            remote_offset = remote_offsets[k]
            batch_size = len(remote_offset) - 1
            ev_batch = []
            
            for i in range(batch_size):
                
                start = remote_offset[i]
                end = remote_offset[i + 1]

                ev = remote_evs[start: end]

                ev_batch.append(ev.sum(dim=0))

            V = torch.add(torch.tensor(np.array(ev_batch)), cache_ly[k])
            ly.append(V)

        end_time = time.time()
        total_time = end_time - start_time
        total_time *= 1000
        self.CPU_cal_time.append(total_time)

        return ly
    
    def close(self):
        self.client.close()

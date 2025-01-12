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

class EmbClient:
    def __init__(self):
        self.CLIENT_RECV_WR = 3
        self.CLIENT_SEND_WR = 4
    
    def read_mr(self, length, offset):
        return self.mr.read(length, offset)

    def connect(self, server_port, server_ip, wr_capacity):
        self.conn = SKT(server_port, server_ip)
        self.conn.handshake()

        print("New connection...")

        ctx = Context(name='mlx5_0')
        self.pd = PD(ctx)
        self.cq = CQ(ctx, wr_capacity)

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
        self.conn.handshake()
        
        mr_size = 16 * 1024 * 1024
        self.mr = MR(self.pd, mr_size, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ)
        self.sgl = [SGE(self.mr.buf, self.mr.length, self.mr.lkey)]

        self.remote_info = self.conn.handshake(addr=self.mr.buf, rkey=self.mr.rkey)

    def send_request(self, input_data):
        self.conn.handshake()

        # prepare request
        header = {}
        request = pickle.dumps({'header': header, 'data': input_data})
        request_len = len(request)
        self.mr.write(request_len.to_bytes(8, 'little'), 8, 0)
        self.mr.write(request, request_len, 8)

        # check server recv ready
        self.conn.handshake()

        wr = SendWR(self.CLIENT_SEND_WR, opcode=IBV_WR_SEND, num_sge=1, sg=self.sgl)
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
        response = pickle.loads(self.read_mr(response_len, 8))
        return response

    def close(self):
        self.conn.close()

class EmbStorage():
    def __init__(self, m, ln, num_indices_per_lookup, batch_size, emb_pool_ip, emb_pool_port, wr_capacity):
        self.m = m
        self.ln = ln
        
        self.client = EmbClient()
        
        flag = 0
        while flag != 2:
            try:
                self.client.connect(emb_pool_port, emb_pool_ip, wr_capacity)
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

        start_time = time.time()

        ly = self.client.send_request({'lS_o': lS_o, 'lS_i': lS_i})['data']

        end_time = time.time()
        total_time = end_time - start_time
        total_time *= 1000
        print("total apply_emb time (ms): " + f"{total_time}")

        return ly
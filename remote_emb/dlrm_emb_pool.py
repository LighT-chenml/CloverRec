import argparse

import numpy as np

import pickle

import torch

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

class EmbServer:
    def __init__(self, m, ln):
        self.SERVER_RECV_WR = 1
        self.SERVER_SEND_WR = 2
        
        print("m: " + f'{m}')
        print("ln: " + f'{ln}')
        
        self.content = bytearray()
        for n in ln:
            low = -np.sqrt(1 / n)
            high = np.sqrt(1 / n)
            W = low + torch.rand(n, m ,dtype=torch.float32) * (high - low)
            self.content.extend(bytearray(np.array(W).tobytes()))
        self.content = bytes(self.content)
    
    def read_mr(self, length, offset):
        return self.mr.read(length, offset)

    def check_connection_alive(self):
        try:
            sgl = [SGE(self.mr.buf + self.mr.length - 1, 1, self.mr.lkey)]
            wr = SendWR(self.SERVER_SEND_WR, opcode=IBV_WR_RDMA_READ, num_sge=1, sg=sgl)
            wr.set_wr_rdma(self.remote_info['rkey'], self.remote_info['addr'])

            self.qp.post_send(wr)
            
            wc_num, wc_list = self.cq.poll()
            
            return True
        except:
            return False

    def start_connection(self):
        
        self.conn = CM(args.emb_pool_port, None)

        print("New connection...")

        ctx = Context(name='mlx5_0')
        self.pd = PD(ctx)
        self.cq = CQ(ctx, args.rdma_wr_capacity)

        cap = QPCap(max_send_wr=args.rdma_wr_capacity, max_recv_wr=args.rdma_wr_capacity, max_send_sge=1, max_recv_sge=1, max_inline_data=0)
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

        mr_size = len(self.content) + 16
        self.mr = MR(self.pd, mr_size, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ)
        self.mr.write(self.content, len(self.content))

        self.remote_info = self.conn.handshake(addr=self.mr.buf, rkey=self.mr.rkey)

        num_iter = 0

        while True:
            ret = self.check_connection_alive()
            if ret != True:
                break

            time.sleep(0.001) # 1 ms

        self.conn.close()
        
        print("Close connection...")
        print('-' * 80)

def dash_separated_ints(value):
    vals = value.split("-")
    for val in vals:
        try:
            int(val)
        except ValueError:
            raise argparse.ArgumentTypeError(
                "%s is not a valid dash separated list of ints" % value
            )

    return value

parser = argparse.ArgumentParser(
    description="Train Deep Learning Recommendation Model (DLRM)"
)
# model related parameters
parser.add_argument("--arch-sparse-feature-size", type=int, default=2)
parser.add_argument(
    "--arch-embedding-size", type=dash_separated_ints, default="4-3-2"
)
parser.add_argument("--rdma-wr-capacity", type=int, default=16)
parser.add_argument("--emb-pool-port", type=int, default=1234)

global args
args = parser.parse_args()
ln_emb = np.fromstring(args.arch_embedding_size, dtype=int, sep="-")
ln_emb = np.asarray(ln_emb)
m_spa = args.arch_sparse_feature_size

server = EmbServer(m_spa, ln_emb)

print("start emb pool")

while True:
    server.start_connection()


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

# PIM
from pim_module import PIMEmbStorage

import time

class EmbServer:
    def __init__(self, m, ln):
        self.SERVER_RECV_WR = 1
        self.SERVER_SEND_WR = 2
        
        print("m: " + f'{m}')
        print("ln: " + f'{ln}')
        
        start_time = time.time()

        self.emb_l = []
        for n in ln:
            low = -np.sqrt(1 / n)
            high = np.sqrt(1 / n)
            W = low + torch.rand(n, m ,dtype=torch.float32) * (high - low)
            self.emb_l.append(W)

        end_time = time.time()
        total_time = end_time - start_time
        print("generate emb time (s): " + f"{total_time}")

        start_time = time.time()

        content = torch.flatten(torch.stack(self.emb_l, dim=0)).numpy().tolist()

        end_time = time.time()
        total_time = end_time - start_time
        print("convertion time (s): " + f"{total_time}")

        start_time = time.time()

        self.pim_emb_storage = PIMEmbStorage()
        self.pim_emb_storage.initialize(m, np.array(ln).tolist(), content)

        end_time = time.time()
        total_time = end_time - start_time
        print("pim module init time (s): " + f"{total_time}")
        
        self.pim_emb_storage.init_pim()
        self.pim_emb_storage.run_pim()
        self.pim_emb_storage.output_pim()

    def read_mr(self, length, offset):
        return self.mr.read(length, offset)

    def start_connection(self):
        
        self.conn = SKT(1234, None)
        self.conn.handshake()

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
        self.conn.handshake()

        mr_size = 16 * 1024 * 1024
        self.mr = MR(self.pd, mr_size, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ)
        self.sgl = [SGE(self.mr.buf, self.mr.length, self.mr.lkey)]

        self.remote_info = self.conn.handshake(addr=self.mr.buf, rkey=self.mr.rkey)

        num_iter = 0

        while True:
            ret = self.handle_request()
            if ret != True:
                break

        self.conn.close()
        
        print("Close connection...")
        print('-' * 80)

    def handle_request(self):
        # check whether connection is alive
        ret = self.conn.handshake()
        if ret != True:
            return False

        wr = RecvWR(self.SERVER_RECV_WR, num_sge=1, sg=self.sgl)
        self.qp.post_recv(wr)

        # check server recv ready (handle connection close error)
        self.conn.handshake()

        # check client send ready
        self.conn.handshake()

        wc_num, wc_list = self.cq.poll()

        # load request
        request_len = int.from_bytes(self.read_mr(8, 0), 'little')
        request_bytes = self.read_mr(request_len, 8)

        request = pickle.loads(request_bytes)
        header = request['header']
        input_data = request['data']

        lS_o = input_data['lS_o']
        lS_i = input_data['lS_i']

        start_time = time.time()

        # prepare request response
        ret = self.apply_emb(lS_o, lS_i)
        
        end_time = time.time()
        total_time = end_time - start_time
        total_time *= 1000
        print("ev lookup time (ms): " + f"{total_time}")

        header = {}
        output_data = ret
        response = pickle.dumps({'header': header, 'data': output_data})

        response_len = len(response)
        self.mr.write(response_len.to_bytes(8, 'little'), 8, 0)
        self.mr.write(response, response_len, 8)

        # check client recv ready
        self.conn.handshake()

        wr = SendWR(self.SERVER_SEND_WR, opcode=IBV_WR_SEND, num_sge=1, sg=self.sgl)
        self.qp.post_send(wr)

        # check server send ready
        self.conn.handshake()

        wc_num, wc_list = self.cq.poll()

        return True

    def apply_emb(self, lS_o, lS_i):
        ly = []

        indices = []
        for k, sparse_index_group_batch in enumerate(lS_i):
            indices.append(np.array(sparse_index_group_batch).tolist())

        ret = self.pim_emb_storage.apply_emb(np.array(lS_o).tolist(), indices)

        for k, evs in enumerate(ret):
            V = torch.tensor(np.array(evs, dtype=np.float32))
            ly.append(V)

        return ly

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

global args
args = parser.parse_args()
ln_emb = np.fromstring(args.arch_embedding_size, dtype=int, sep="-")
ln_emb = np.asarray(ln_emb)
m_spa = args.arch_sparse_feature_size

server = EmbServer(m_spa, ln_emb)

print("start emb pool")

while True:
    server.start_connection()


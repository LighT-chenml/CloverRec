#!/usr/bin/python3.8

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

SERVER_RECV_WR = 1
SERVER_SEND_WR = 2

# TODO: Error handling

print('-' * 80)
print(' ' * 25, "Python test for RDMA")

print("Running as server...")

print('-' * 80)

class RPCServer:
    def read_mr(self, length, offset):
        return self.mr.read(length, offset)

    def check_connection_alive(self):
        try:
            sgl = [SGE(self.mr.buf + self.mr.length - 1, 1, self.mr.lkey)]
            wr = SendWR(SERVER_SEND_WR, opcode=IBV_WR_RDMA_READ, num_sge=1, sg=sgl)
            wr.set_wr_rdma(self.remote_info['rkey'], self.remote_info['addr'])

            self.qp.post_send(wr)
            
            wc_num, wc_list = self.cq.poll()
            
            return True
        except:
            return False

    def start_connection(self):
        
        self.conn = CM(8000, None)

        print("New connection...")

        ctx = Context(name='mlx5_0')
        self.pd = PD(ctx)
        self.cq = CQ(ctx, 100)

        cap = QPCap(max_send_wr=16, max_recv_wr=16, max_send_sge=1, max_recv_sge=1, max_inline_data=0)
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

        mr_size = 32
        content = '123456789abcd'

        self.mr = MR(self.pd, mr_size, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ)
        self.sgl = [SGE(self.mr.buf, self.mr.length, self.mr.lkey)]
        self.mr.write(content, len(content))

        self.remote_info = self.conn.handshake(addr=self.mr.buf, rkey=self.mr.rkey)

        num_iter = 0

        while True:
            ret = self.check_connection_alive()
            if ret != True:
                break
            
            # num_iter += 1
            # print("Iter: " + f"{num_iter}")
            # print("Init Server MR Content:" + self.read_mr(self.mr.length, 0).decode())

            time.sleep(0.001) # 1 ms

        self.conn.close()
        
        print("Close connection...")
        print('-' * 80)

server = RPCServer()

while True:
    server.start_connection()
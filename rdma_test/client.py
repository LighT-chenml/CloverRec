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

CLIENT_RECV_WR = 3
CLIENT_SEND_WR = 4

# TODO: Error handling

print('-' * 80)
print(' ' * 25, "Python test for RDMA")

print("Running as client...")

print('-' * 80)

class RPCClient:
    def read_mr(self, length, offset):
        return self.mr.read(length, offset)

    def connect(self):
        self.conn = SKT(8000, '10.0.0.7')

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
        self.content = 'c' * 16

        self.mr = MR(self.pd, mr_size, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ)
        self.sgl = [SGE(self.mr.buf, self.mr.length, self.mr.lkey)]

    def send_request(self):
        self.mr.write(self.content, len(self.content))
        print("Request:" + self.read_mr(self.mr.length, 0).decode())

        # check server recv ready
        self.conn.handshake()

        wr = SendWR(CLIENT_SEND_WR, opcode=IBV_WR_SEND, num_sge=1, sg=self.sgl)
        self.qp.post_send(wr)

        # check client send ready
        self.conn.handshake()

        wc_num, wc_list = self.cq.poll()

        wr = RecvWR(CLIENT_RECV_WR, num_sge=1, sg=self.sgl)
        self.qp.post_recv(wr)

        # check client recv ready
        self.conn.handshake()

        # check server send ready
        self.conn.handshake()

        wc_num, wc_list = self.cq.poll()

        # return response
        res_len = int.from_bytes(self.read_mr(8, 0), 'little')
        return self.read_mr(res_len, 8).decode()

    def close(self):
        self.conn.close()


client = RPCClient()
client.connect()

for i in range(10):
    print("Iter: " + f"{i + 1}/{10}")
    print("Response:" + client.send_request())

client.close()
print('-' * 80)
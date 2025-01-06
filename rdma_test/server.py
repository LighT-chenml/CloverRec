#!/usr/bin/python3.8

from connection import SKT

from pyverbs.addr import AH, AHAttr, GlobalRoute
from pyverbs.cq import CQ
from pyverbs.device import Context
from pyverbs.enums import *
from pyverbs.mr import MR
from pyverbs.pd import PD
from pyverbs.qp import QP, QPCap, QPInitAttr, QPAttr
from pyverbs.wr import SGE, RecvWR, SendWR

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

    def handle_request(self):
        wr = RecvWR(SERVER_RECV_WR, num_sge=1, sg=self.sgl)
        self.qp.post_recv(wr)

        # check server recv ready (handle connection close error)
        ret = self.conn.handshake()
        if ret != True:
            return False

        # check client send ready
        self.conn.handshake()

        wc_num, wc_list = self.cq.poll()

        # load request

        print("Request:" + self.read_mr(self.mr.length, 0).decode())

        # prepare request response
        response_content = 'a' * 8
        # response_content_2 = 'b' * 8
        
        self.mr.write(response_content, len(response_content), 0)
        print("Response:" + self.read_mr(len(response_content), 0).decode())

        # check client recv ready
        self.conn.handshake()

        wr = SendWR(SERVER_SEND_WR, opcode=IBV_WR_SEND, num_sge=1, sg=self.sgl)
        self.qp.post_send(wr)

        # check server send ready
        self.conn.handshake()

        wc_num, wc_list = self.cq.poll()

        return True

    def start_connection(self):
        
        self.conn = SKT(8000, None)
        self.conn.handshake()

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
        # remote_info = conn.handshake(gid=gid, qpn=qp.qp_num)
        remote_info = self.conn.handshake(gid=gid, lid=lid, qpn=self.qp.qp_num)

        # gr = GlobalRoute(dgid=remote_info['gid'], sgid_index=args['gid_index'])
        # ah_attr = AHAttr(gr=gr, is_global=1, port_num=1)
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

        mr_size = 32
        content = 's' * 16

        self.mr = MR(self.pd, mr_size, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ)
        self.sgl = [SGE(self.mr.buf, self.mr.length, self.mr.lkey)]

        num_iter = 0

        while True:
            num_iter += 1
            print("Iter: " + f"{num_iter}")

            self.mr.write(content, len(content))
            print("Init Server MR Content:" + self.read_mr(self.mr.length, 0).decode())

            ret = self.handle_request()

            if ret == False:
                break

        # conn.handshake()
        self.conn.close()

        print('-' * 80)

server = RPCServer()

while True:
    server.start_connection()
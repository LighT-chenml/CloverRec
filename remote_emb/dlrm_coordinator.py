# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Description: an implementation of a deep learning recommendation model (DLRM)
# The model input consists of dense and sparse features. The former is a vector
# of floating point values. The latter is a list of sparse indices into
# embedding tables, which consist of vectors of floating point values.
# The selected vectors are passed to mlp networks denoted by triangles,
# in some cases the vectors are interacted through operators (Ops).
#
# output:
#                         vector of values
# model:                        |
#                              /\
#                             /__\
#                               |
#       _____________________> Op  <___________________
#     /                         |                      \
#    /\                        /\                      /\
#   /__\                      /__\           ...      /__\
#    |                          |                       |
#    |                         Op                      Op
#    |                    ____/__\_____           ____/__\____
#    |                   |_Emb_|____|__|    ...  |_Emb_|__|___|
# input:
# [ dense features ]     [sparse indices] , ..., [sparse indices]
#
# More precise definition of model layers:
# 1) fully connected layers of an mlp
# z = f(y)
# y = Wx + b
#
# 2) embedding lookup (for a list of sparse indices p=[p1,...,pk])
# z = Op(e1,...,ek)
# obtain vectors e1=E[:,p1], ..., ek=E[:,pk]
#
# 3) Operator Op can be one of the following
# Sum(e1,...,ek) = e1 + ... + ek
# Dot(e1,...,ek) = [e1'e1, ..., e1'ek, ..., ek'e1, ..., ek'ek]
# Cat(e1,...,ek) = [e1', ..., ek']'
# where ' denotes transpose operation
#
# References:
# [1] Maxim Naumov, Dheevatsa Mudigere, Hao-Jun Michael Shi, Jianyu Huang,
# Narayanan Sundaram, Jongsoo Park, Xiaodong Wang, Udit Gupta, Carole-Jean Wu,
# Alisson G. Azzolini, Dmytro Dzhulgakov, Andrey Mallevich, Ilia Cherniavskii,
# Yinghai Lu, Raghuraman Krishnamoorthi, Ansha Yu, Volodymyr Kondratenko,
# Stephanie Pereira, Xianjie Chen, Wenlin Chen, Vijay Rao, Bill Jia, Liang Xiong,
# Misha Smelyanskiy, "Deep Learning Recommendation Model for Personalization and
# Recommendation Systems", CoRR, arXiv:1906.00091, 2019

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse

# emb storage
from dlrm_emb_storage import EmbStorage

# RDMA
import socket
import pickle
import rpyc
from connection import SKT
from pyverbs.addr import AH, AHAttr, GlobalRoute
from pyverbs.cq import CQ
from pyverbs.device import Context
from pyverbs.enums import *
from pyverbs.mr import MR
from pyverbs.pd import PD
from pyverbs.qp import QP, QPCap, QPInitAttr, QPAttr
from pyverbs.wr import SGE, RecvWR, SendWR

# miscellaneous
import builtins
import datetime
import json
import sys
import time
from pathlib import Path
import os
import pandas as pd

# onnx
# The onnx import causes deprecation warnings every time workers
# are spawned during testing. So, we filter out those warnings.
import warnings

# data generation
import dlrm_data_pytorch as dp

# numpy
import numpy as np

# pytorch
import torch
import torch.nn as nn

# dataloader
try:
    from internals import fbDataLoader, fbInputBatchFormatter

    has_internal_libs = True
except ImportError:
    has_internal_libs = False

from torch._ops import ops
from torch.autograd.profiler import record_function
from torch.nn.parallel.parallel_apply import parallel_apply
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.scatter_gather import gather, scatter
from torch.nn.parameter import Parameter
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard import SummaryWriter

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    try:
        import onnx
    except ImportError as error:
        print("Unable to import onnx. ", error)

# from torchviz import make_dot
# import torch.nn.functional as Functional
# from torch.nn.parameter import Parameter

exc = getattr(builtins, "IOError", "FileNotFoundError")


def ensure_mr_capacity(mr, payload_len, label):
    required = payload_len + 8
    if required > mr.length:
        raise RuntimeError(
            f"{label} payload requires {required} bytes, "
            f"but RDMA MR is {mr.length} bytes. Increase --rdma-mr-size-mb."
        )


def dlrm_wrap(X, lS_o, lS_i, use_gpu, device, ndevices=1):
    with record_function("DLRM forward"):
        return dlrm(X, lS_o, lS_i)

# The following function is a wrapper to avoid checking this multiple times in th
# loop below.
def unpack_batch(b):
    if args.data_generation == "internal":
        return fbInputBatchFormatter(b, args.data_size)
    else:
        # Experiment with unweighted samples
        return b[0], b[1], b[2], b[3], torch.ones(b[3].size()), None

### define dlrm in PyTorch ###
class DLRM_Net(nn.Module):
    def create_emb(self, m, ln):
        emb_l = nn.ModuleList()
        v_W_l = []
        for i in range(0, ln.size):
            n = ln[i]

            # construct embedding operator
            EE = nn.EmbeddingBag(n, m, mode="sum", sparse=True)
            # initialize embeddings
            # nn.init.uniform_(EE.weight, a=-np.sqrt(1 / n), b=np.sqrt(1 / n))
            W = np.random.uniform(
                low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, m)
            ).astype(np.float32)
            # approach 1
            EE.weight.data = torch.tensor(W, requires_grad=True)
            # approach 2
            # EE.weight.data.copy_(torch.tensor(W))
            # approach 3
            # EE.weight = Parameter(torch.tensor(W),requires_grad=True)
            v_W_l.append(None)
            emb_l.append(EE)
        return emb_l, v_W_l
    
    def apply_emb(self, lS_o, lS_i, emb_l, v_W_l):
        # WARNING: notice that we are processing the batch at once. We implicitly
        # assume that the data is laid out such that:
        # 1. each embedding is indexed with a group of sparse indices,
        #   corresponding to a single lookup
        # 2. for each embedding the lookups are further organized into a batch
        # 3. for a list of embedding tables there is a list of batched lookups

        ly = []
        for k, sparse_index_group_batch in enumerate(lS_i):
            sparse_offset_group_batch = lS_o[k]

            # print(type(sparse_index_group_batch))
            # print(len(sparse_index_group_batch))
            # print(sparse_index_group_batch)
            
            # print(type(sparse_offset_group_batch))
            # print(len(sparse_offset_group_batch))
            # print(sparse_offset_group_batch)

            # embedding lookup
            # We are using EmbeddingBag, which implicitly uses sum operator.
            # The embeddings are represented as tall matrices, with sum
            # happening vertically across 0 axis, resulting in a row vector
            # E = emb_l[k]
            
            E = emb_l[k]
            V = E(
                sparse_index_group_batch,
                sparse_offset_group_batch,
                per_sample_weights=None,
            )

            ly.append(V)

        return ly

    def __init__(
        self,
        m_spa=None,
        ln_emb=None,
        ln_bot=None,
        ln_top=None,
        arch_interaction_op=None,
        arch_interaction_itself=False,
        sigmoid_bot=-1,
        sigmoid_top=-1,
        sync_dense_params=True,
        loss_threshold=0.0,
        ndevices=-1,
        qr_flag=False,
        qr_operation="mult",
        qr_collisions=0,
        qr_threshold=200,
        md_flag=False,
        md_threshold=200,
        weighted_pooling=None,
        loss_function="bce",
    ):
        super(DLRM_Net, self).__init__()

        if (
            (m_spa is not None)
            and (ln_emb is not None)
            and (ln_bot is not None)
            and (ln_top is not None)
            and (arch_interaction_op is not None)
        ):
            # save arguments
            self.ndevices = ndevices
            self.output_d = 0
            self.parallel_model_batch_size = -1
            self.parallel_model_is_not_prepared = True
            self.arch_interaction_op = arch_interaction_op
            self.arch_interaction_itself = arch_interaction_itself
            self.sync_dense_params = sync_dense_params
            self.loss_threshold = loss_threshold
            self.loss_function = loss_function
            self.weighted_pooling = weighted_pooling
            
            # create variables for QR embedding if applicable
            self.qr_flag = qr_flag
            if self.qr_flag:
                self.qr_collisions = qr_collisions
                self.qr_operation = qr_operation
                self.qr_threshold = qr_threshold
            # create variables for MD embedding if applicable
            self.md_flag = md_flag
            if self.md_flag:
                self.md_threshold = md_threshold

            # create operators
            # self.emb_l, self.v_W_l = self.create_emb(m_spa, ln_emb)
            self.emb_storage = EmbStorage(m_spa, ln_emb, args.num_indices_per_lookup, args.mini_batch_size, args.emb_pool_ip, args.emb_pool_port, args.rdma_wr_capacity)

            self.ev_lookup_time = []
            self.apply_emb_time = []

            # specify the loss function
            if self.loss_function == "mse":
                self.loss_fn = torch.nn.MSELoss(reduction="mean")
            elif self.loss_function == "bce":
                self.loss_fn = torch.nn.BCELoss(reduction="mean")
            elif self.loss_function == "wbce":
                self.loss_ws = torch.tensor(
                    np.fromstring(args.loss_weights, dtype=float, sep="-")
                )
                self.loss_fn = torch.nn.BCELoss(reduction="none")
            else:
                sys.exit(
                    "ERROR: --loss-function=" + self.loss_function + " is not supported"
                )

    def forward(self, dense_x, lS_o, lS_i):
        # single device run
        return self.sequential_forward(dense_x, lS_o, lS_i)

    def sequential_forward(self, dense_x, lS_o, lS_i):
        # process dense features (using bottom mlp), resulting in a row vector
        
        total_start_time = time.time()
        
        client.send_request(0, {'dense_x': dense_x})

        start_time = time.time()

        # process sparse features(using embeddings), resulting in a list of row vectors
        # ly = self.apply_emb(lS_o, lS_i, self.emb_l, self.v_W_l)
        ly = self.emb_storage.apply_emb(lS_o, lS_i)
        
        end_time = time.time()
        total_time = end_time - start_time
        total_time *= 1000
        print("ev lookup time (ms): " + f"{total_time}")
        self.ev_lookup_time.append(total_time)

        # obtain probability of a click (using top mlp)
        p = client.send_request(1, {'ly': ly})['data']
        
        end_time = time.time()
        total_time = end_time - total_start_time
        total_time *= 1000
        self.apply_emb_time.append(total_time)

        # clamp output if needed
        if 0.0 < self.loss_threshold and self.loss_threshold < 1.0:
            z = torch.clamp(p, min=self.loss_threshold, max=(1.0 - self.loss_threshold))
        else:
            z = p

        return z

class GPUClient:
    def __init__(self):
        self.CLIENT_RECV_WR = 3
        self.CLIENT_SEND_WR = 4
    
    def read_mr(self, length, offset):
        return self.mr.read(length, offset)

    def send_sgl(self, payload_len):
        return [SGE(self.mr.buf, payload_len + 8, self.mr.lkey)]

    def connect(self, server_ip, server_port, mr_size_mb):
        self.conn = SKT(server_port, server_ip)
        self.conn.handshake()

        print("New connection...")

        ctx = Context(name='mlx5_0')
        self.pd = PD(ctx)
        self.cq = CQ(ctx, 100)

        cap = QPCap(max_send_wr=16, max_recv_wr=16, max_send_sge=1, max_recv_sge=1, max_inline_data=0)
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

    def send_request(self, func_type, input_data):
        self.conn.handshake()
        
        # prepare request
        header = {'func_type': func_type}
        request = pickle.dumps({'header': header, 'data': input_data})
        request_len = len(request)
        ensure_mr_capacity(self.mr, request_len, "model request")
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
        ensure_mr_capacity(self.mr, response_len, "model response")
        response = pickle.loads(self.read_mr(response_len, 8))
        return response

    def close(self):
        self.conn.close()

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


def dash_separated_floats(value):
    vals = value.split("-")
    for val in vals:
        try:
            float(val)
        except ValueError:
            raise argparse.ArgumentTypeError(
                "%s is not a valid dash separated list of floats" % value
            )

    return value


def inference(
    args,
    dlrm,
    best_acc_test,
    best_auc_test,
    test_ld,
    device,
    use_gpu,
    log_iter=-1,
):
    test_accu = 0
    test_samp = 1

    arr_time_latency = []

    n_progress_indicator = 100 # 40

    if args.data_generation == "dataset":
        if (args.inference_only):
            print("==== ==== Progress bar (nWorkload: " + str(nbatches) + ") shown below:")
            progress_bar_freq = int(nbatches * args.num_indices_per_lookup / n_progress_indicator)
        
        batched_X_test = []
        batched_lS_o_test = []
        batched_lS_i_test = []
        
        for i, testBatch in enumerate(test_ld):
            # early exit if nbatches was set by the user and was exceeded
            if nbatches > 0 and i >= nbatches * args.num_indices_per_lookup:
                break

            # print progress bar
            if (i % progress_bar_freq == 0):
                print(".", end ="", flush=True)

            X_test, lS_o_test, lS_i_test, T_test, W_test, CBPP_test = unpack_batch(
                testBatch
            )
            
            batched_X_test.append(X_test)
            batched_lS_o_test.append(lS_o_test)
            batched_lS_i_test.append(lS_i_test)
            
            if (i + 1) % args.num_indices_per_lookup == 0:
                m_den = len(X_test[0])
                X_test_l = []
                for j in range(args.mini_batch_size):
                    sum = torch.zeros(m_den, dtype=torch.float32)
                    for k in range(args.num_indices_per_lookup):
                        X_test = batched_X_test[k][j]
                        sum = torch.add(sum, X_test)
                    X_test_l.append(sum)
                X_test = torch.stack(X_test_l, dim=0)
                
                lS_o_test_l = []
                lS_i_test_l = []
                for table_id in range(len(lS_i_test)):
                    indices = []
                    offsets = []
                    for j in range(args.mini_batch_size):
                        index_l = []
                        for k in range(args.num_indices_per_lookup):
                            index_l.append(batched_lS_i_test[k][table_id][j])
                        # index_l = np.unique(np.array(index_l))
                        offsets.append(len(indices))
                        indices.extend(index_l)
                    indices = np.array(indices)
                    lS_i_test_l.append(torch.tensor(indices))
                    lS_o_test_l.append(offsets)
                lS_o_test = torch.tensor(np.array(lS_o_test_l))
                lS_i_test = lS_i_test_l

                batched_X_test = []
                batched_lS_o_test = []
                batched_lS_i_test = []
                
                infer_time_start = time.time()

                # forward pass
                Z_test = dlrm_wrap(
                    X_test,
                    lS_o_test,
                    lS_i_test,
                    use_gpu,
                    device,
                    ndevices=ndevices,
                )
                
                infer_time_end = time.time()
                
                arr_time_latency.append(infer_time_end - infer_time_start)
    else:
        if (args.inference_only):
            print("==== ==== Progress bar (nWorkload: " + str(nbatches) + ") shown below:")
            progress_bar_freq = int(nbatches / n_progress_indicator) 

        for i, testBatch in enumerate(test_ld):
            # early exit if nbatches was set by the user and was exceeded
            if nbatches > 0 and i >= nbatches:
                break

            # print progress bar
            if (i % progress_bar_freq == 0):
                print(".", end ="", flush=True)

            X_test, lS_o_test, lS_i_test, T_test, W_test, CBPP_test = unpack_batch(
                testBatch
            )

            infer_time_start = time.time()

            # forward pass
            Z_test = dlrm_wrap(
                X_test,
                lS_o_test,
                lS_i_test,
                use_gpu,
                device,
                ndevices=ndevices,
            )
            
            infer_time_end = time.time()
            
            arr_time_latency.append(infer_time_end - infer_time_start)

    print("") # printing enter for the progress bar

    acc_test = test_accu / test_samp

    model_metrics_dict = {
        "nepochs": args.nepochs,
        "nbatches": nbatches,
        "nbatches_test": nbatches_test,
        "state_dict": dlrm.state_dict(),
        "test_acc": acc_test,
    }

    is_best = acc_test > best_acc_test
    if is_best:
        best_acc_test = acc_test
    print(
        " accuracy {:3.3f} %, best {:3.3f} %".format(
            acc_test * 100, best_acc_test * 100
        ),
        flush=True,
    )
    return model_metrics_dict, is_best, arr_time_latency

def try_connect():
    conn = GPUClient()

    flag = 0
    while flag != 2:
        try:
            conn.connect(args.server_ip, args.server_port, args.rdma_mr_size_mb)
            flag = 2
        except:
            if flag == 0:
                print("waiting for model server")
                flag = 1
    return conn

def run():
    ### parse arguments ###
    parser = argparse.ArgumentParser(
        description="Train Deep Learning Recommendation Model (DLRM)"
    )
    # model related parameters
    parser.add_argument("--arch-sparse-feature-size", type=int, default=2)
    parser.add_argument(
        "--arch-embedding-size", type=dash_separated_ints, default="4-3-2"
    )
    # j will be replaced with the table number
    parser.add_argument("--arch-mlp-bot", type=dash_separated_ints, default="4-3-2")
    parser.add_argument("--arch-mlp-top", type=dash_separated_ints, default="4-2-1")
    parser.add_argument(
        "--arch-interaction-op", type=str, choices=["dot", "cat"], default="dot"
    )
    parser.add_argument("--arch-interaction-itself", action="store_true", default=False)
    parser.add_argument("--weighted-pooling", type=str, default=None)
    # embedding table options
    parser.add_argument("--md-flag", action="store_true", default=False)
    parser.add_argument("--md-threshold", type=int, default=200)
    parser.add_argument("--md-temperature", type=float, default=0.3)
    parser.add_argument("--md-round-dims", action="store_true", default=False)
    parser.add_argument("--qr-flag", action="store_true", default=False)
    parser.add_argument("--qr-threshold", type=int, default=200)
    parser.add_argument("--qr-operation", type=str, default="mult")
    parser.add_argument("--qr-collisions", type=int, default=4)
    # activations and loss
    parser.add_argument("--activation-function", type=str, default="relu")
    parser.add_argument("--loss-function", type=str, default="mse")  # or bce or wbce
    parser.add_argument(
        "--loss-weights", type=dash_separated_floats, default="1.0-1.0"
    )  # for wbce
    parser.add_argument("--loss-threshold", type=float, default=0.0)  # 1.0e-7
    parser.add_argument("--round-targets", type=bool, default=False)
    # data
    parser.add_argument("--data-size", type=int, default=1)
    parser.add_argument("--num-batches", type=int, default=0)
    parser.add_argument(
        "--data-generation",
        type=str,
        choices=["random", "dataset", "internal"],
        default="random",
    )  # synthetic, dataset or internal
    parser.add_argument(
        "--rand-data-dist", type=str, default="uniform"
    )  # uniform or gaussian
    parser.add_argument("--rand-data-min", type=float, default=0)
    parser.add_argument("--rand-data-max", type=float, default=1)
    parser.add_argument("--rand-data-mu", type=float, default=-1)
    parser.add_argument("--rand-data-sigma", type=float, default=1)
    parser.add_argument("--data-trace-file", type=str, default="./input/dist_emb_j.log")
    parser.add_argument("--data-set", type=str, default="kaggle")  # or terabyte
    parser.add_argument("--raw-data-file", type=str, default="")
    parser.add_argument("--processed-data-file", type=str, default="")
    parser.add_argument("--data-randomize", type=str, default="total")  # or day or none
    parser.add_argument("--data-trace-enable-padding", type=bool, default=False)
    parser.add_argument("--zipf-parameter", type=float, default=1.5)
    parser.add_argument("--max-ind-range", type=int, default=-1)
    parser.add_argument("--data-sub-sample-rate", type=float, default=0.0)  # in [0, 1]
    parser.add_argument("--num-indices-per-lookup", type=int, default=10)
    parser.add_argument("--num-indices-per-lookup-fixed", type=bool, default=False)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--memory-map", action="store_true", default=False)
    # training
    parser.add_argument("--mini-batch-size", type=int, default=1)
    parser.add_argument("--nepochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--print-precision", type=int, default=5)
    parser.add_argument("--numpy-rand-seed", type=int, default=123)
    parser.add_argument("--sync-dense-params", type=bool, default=True)
    parser.add_argument("--optimizer", type=str, default="sgd")
    parser.add_argument(
        "--dataset-multiprocessing",
        action="store_true",
        default=False,
        help="The Kaggle dataset can be multiprocessed in an environment \
                        with more than 7 CPU cores and more than 20 GB of memory. \n \
                        The Terabyte dataset can be multiprocessed in an environment \
                        with more than 24 CPU cores and at least 1 TB of memory.",
    )
    # inference
    parser.add_argument("--inference-only", action="store_true", default=False)
    parser.add_argument("--get-cdf-lat", type=str, default="False")
    parser.add_argument("--cdf-output-dir", type=str, default="./latency/")
    # quantize
    parser.add_argument("--quantize-mlp-with-bit", type=int, default=32)
    parser.add_argument("--quantize-emb-with-bit", type=int, default=32)
    # onnx
    parser.add_argument("--save-onnx", action="store_true", default=False)
    # gpu
    parser.add_argument("--use-gpu", action="store_true", default=False)
    # distributed
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--dist-backend", type=str, default="")
    # debugging and profiling
    parser.add_argument("--print-freq", type=int, default=1)
    parser.add_argument("--test-freq", type=int, default=-1)
    parser.add_argument("--test-mini-batch-size", type=int, default=-1)
    parser.add_argument("--test-num-workers", type=int, default=-1)
    parser.add_argument("--print-time", action="store_true", default=False)
    parser.add_argument("--print-wall-time", action="store_true", default=False)
    parser.add_argument("--debug-mode", action="store_true", default=False)
    parser.add_argument("--enable-profiling", action="store_true", default=False)
    parser.add_argument("--plot-compute-graph", action="store_true", default=False)
    parser.add_argument("--tensor-board-filename", type=str, default="run_kaggle_pt")
    # store/load model
    parser.add_argument("--save-model", type=str, default="")
    parser.add_argument("--load-model", type=str, default="")
    # mlperf logging (disables other output and stops early)
    parser.add_argument("--mlperf-logging", action="store_true", default=False)
    # stop at target accuracy Kaggle 0.789, Terabyte (sub-sampled=0.875) 0.8107
    parser.add_argument("--mlperf-acc-threshold", type=float, default=0.0)
    # stop at target AUC Terabyte (no subsampling) 0.8025
    parser.add_argument("--mlperf-auc-threshold", type=float, default=0.0)
    parser.add_argument("--mlperf-bin-loader", action="store_true", default=False)
    parser.add_argument("--mlperf-bin-shuffle", action="store_true", default=False)
    # mlperf gradient accumulation iterations
    parser.add_argument("--mlperf-grad-accum-iter", type=int, default=1)
    # LR policy
    parser.add_argument("--lr-num-warmup-steps", type=int, default=0)
    parser.add_argument("--lr-decay-start-step", type=int, default=0)
    parser.add_argument("--lr-num-decay-steps", type=int, default=0)
    # rpc
    parser.add_argument("--server-ip", type=str, default='localhost')
    parser.add_argument("--server-port", type=int, default=8000)
    parser.add_argument("--emb-pool-ip", type=str, default='localhost')
    parser.add_argument("--emb-pool-port", type=int, default=8000)
    parser.add_argument("--rdma-wr-capacity", type=int, default=16)
    parser.add_argument("--rdma-mr-size-mb", type=int, default=128)

    global args
    global nbatches
    global nbatches_test
    args = parser.parse_args()

    if args.dataset_multiprocessing:
        assert sys.version_info[0] >= 3 and sys.version_info[1] > 7, (
            "The dataset_multiprocessing "
            + "flag is susceptible to a bug in Python 3.7 and under. "
            + "https://github.com/facebookresearch/dlrm/issues/172"
        )

    if args.weighted_pooling is not None:
        if args.qr_flag:
            sys.exit("ERROR: quotient remainder with weighted pooling is not supported")
        if args.md_flag:
            sys.exit("ERROR: mixed dimensions with weighted pooling is not supported")
    if args.quantize_emb_with_bit in [4, 8]:
        if args.qr_flag:
            sys.exit(
                "ERROR: 4 and 8-bit quantization with quotient remainder is not supported"
            )
        if args.md_flag:
            sys.exit(
                "ERROR: 4 and 8-bit quantization with mixed dimensions is not supported"
            )
        if args.use_gpu:
            sys.exit("ERROR: 4 and 8-bit quantization on GPU is not supported")

    ### some basic setup ###
    np.random.seed(args.numpy_rand_seed)
    np.set_printoptions(precision=args.print_precision)
    torch.set_printoptions(precision=args.print_precision)
    torch.manual_seed(args.numpy_rand_seed)

    if args.test_mini_batch_size < 0:
        # if the parameter is not set, use the training batch size
        args.test_mini_batch_size = args.mini_batch_size
    if args.test_num_workers < 0:
        # if the parameter is not set, use the same parameter for training
        args.test_num_workers = args.num_workers

    use_gpu = args.use_gpu and torch.cuda.is_available()

    device = torch.device("cpu")

    ### prepare training data ###
    ln_bot = np.fromstring(args.arch_mlp_bot, dtype=int, sep="-")
    # input data

    if args.data_generation == "dataset":
        test_data, test_ld = dp.make_criteo_data_and_loaders(args)
        nbatches = args.num_batches if args.num_batches > 0 else len(test_ld)
        nbatches_test = len(test_ld)

        ln_emb = test_data.counts
        # enforce maximum limit on number of vectors per embedding
        if args.max_ind_range > 0:
            ln_emb = np.array(
                list(
                    map(
                        lambda x: x if x < args.max_ind_range else args.max_ind_range,
                        ln_emb,
                    )
                )
            )
        else:
            ln_emb = np.array(ln_emb)
        m_den = test_data.m_den
        ln_bot[0] = m_den
    elif args.data_generation == "internal":
        if not has_internal_libs:
            raise Exception("Internal libraries are not available.")
        NUM_BATCHES = 5000
        nbatches = args.num_batches if args.num_batches > 0 else NUM_BATCHES
        train_ld, feature_to_num_embeddings = fbDataLoader(args.data_size, nbatches)
        ln_emb = np.array(list(feature_to_num_embeddings.values()))
        m_den = ln_bot[0]
    else:
        # input and target at random
        ln_emb = np.fromstring(args.arch_embedding_size, dtype=int, sep="-")
        m_den = ln_bot[0]
        test_data, test_ld = dp.make_random_data_and_loader(
            args, ln_emb, m_den
        )
        nbatches = args.num_batches if args.num_batches > 0 else len(test_ld)
        nbatches_test = len(test_ld)

    args.ln_emb = ln_emb.tolist()

    ### parse command line arguments ###
    m_spa = args.arch_sparse_feature_size
    ln_emb = np.asarray(ln_emb)
    num_fea = ln_emb.size + 1  # num sparse + num dense features

    m_den_out = ln_bot[ln_bot.size - 1]
    if args.arch_interaction_op == "dot":
        # approach 1: all
        # num_int = num_fea * num_fea + m_den_out
        # approach 2: unique
        if args.arch_interaction_itself:
            num_int = (num_fea * (num_fea + 1)) // 2 + m_den_out
        else:
            num_int = (num_fea * (num_fea - 1)) // 2 + m_den_out
    elif args.arch_interaction_op == "cat":
        num_int = num_fea * m_den_out
    else:
        sys.exit(
            "ERROR: --arch-interaction-op="
            + args.arch_interaction_op
            + " is not supported"
        )
    arch_mlp_top_adjusted = str(num_int) + "-" + args.arch_mlp_top
    ln_top = np.fromstring(arch_mlp_top_adjusted, dtype=int, sep="-")

    # sanity check: feature sizes and mlp dimensions must match
    if m_den != ln_bot[0]:
        sys.exit(
            "ERROR: arch-dense-feature-size "
            + str(m_den)
            + " does not match first dim of bottom mlp "
            + str(ln_bot[0])
        )
    if args.qr_flag:
        if args.qr_operation == "concat" and 2 * m_spa != m_den_out:
            sys.exit(
                "ERROR: 2 arch-sparse-feature-size "
                + str(2 * m_spa)
                + " does not match last dim of bottom mlp "
                + str(m_den_out)
                + " (note that the last dim of bottom mlp must be 2x the embedding dim)"
            )
        if args.qr_operation != "concat" and m_spa != m_den_out:
            sys.exit(
                "ERROR: arch-sparse-feature-size "
                + str(m_spa)
                + " does not match last dim of bottom mlp "
                + str(m_den_out)
            )
    else:
        if m_spa != m_den_out:
            sys.exit(
                "ERROR: arch-sparse-feature-size "
                + str(m_spa)
                + " does not match last dim of bottom mlp "
                + str(m_den_out)
            )
    if num_int != ln_top[0]:
        sys.exit(
            "ERROR: # of feature interactions "
            + str(num_int)
            + " does not match first dimension of top mlp "
            + str(ln_top[0])
        )

    global ndevices
    ndevices = min(ngpus, args.mini_batch_size, num_fea - 1) if use_gpu else -1

    ### construct the neural network specified above ###
    # WARNING: to obtain exactly the same initialization for
    # the weights we need to start from the same random seed.
    # np.random.seed(args.numpy_rand_seed)
    global dlrm
    dlrm = DLRM_Net(
        m_spa,
        ln_emb,
        ln_bot,
        ln_top,
        arch_interaction_op=args.arch_interaction_op,
        arch_interaction_itself=args.arch_interaction_itself,
        sigmoid_bot=-1,
        sigmoid_top=ln_top.size - 2,
        sync_dense_params=args.sync_dense_params,
        loss_threshold=args.loss_threshold,
        ndevices=ndevices,
        qr_flag=args.qr_flag,
        qr_operation=args.qr_operation,
        qr_collisions=args.qr_collisions,
        qr_threshold=args.qr_threshold,
        md_flag=args.md_flag,
        md_threshold=args.md_threshold,
        weighted_pooling=args.weighted_pooling,
        loss_function=args.loss_function,
    )

    ### prepare rpc client
    global client
    client = try_connect()

    ### main loop ###

    # training or inference
    best_acc_test = 0
    best_auc_test = 0

    with torch.autograd.profiler.profile(
        args.enable_profiling, use_cuda=use_gpu, record_shapes=True
    ) as prof:
        print("Testing for inference only")
        model_metrics_dict, is_best, arr_latency = inference(
            args,
            dlrm,
            best_acc_test,
            best_auc_test,
            test_ld,
            device,
            use_gpu,
        )
        
        seconds = 0
        for latency in arr_latency:
            seconds += latency

        avg_latency = seconds / len(arr_latency) * 1000

        print("Time elapsed (FINAL) : " + str(seconds) + " secs (" + str(int(seconds/60)) + " mins)")
        print("Throughput (Req/sec) : " + str(round(args.mini_batch_size * len(arr_latency) / seconds, 2)))
        print("Avg latency (CPU + EV_trans + network + GPU + others) (ms) : " + str(round(avg_latency, 2)))
        avg_ev_lookup_time = sum(dlrm.ev_lookup_time) / len(dlrm.ev_lookup_time)
        print("Avg ev lookup time (CPU + EV_trans) (ms) : " + str(round(avg_ev_lookup_time, 2)))
        avg_apply_emb_time = sum(dlrm.apply_emb_time) / len(dlrm.apply_emb_time)
        print("Avg apply emb time (CPU + EV_trans + network + GPU) (ms) : " + str(round(avg_apply_emb_time, 2)))
        avg_transmission_time = sum(dlrm.emb_storage.transmission_time) / len(dlrm.emb_storage.transmission_time)
        print("Avg transmission time (EV_trans) (ms) : " + str(round(avg_transmission_time, 2)))

        print("CPU cal time (ms) : " + str(round(avg_ev_lookup_time - avg_transmission_time, 2)))
        print("EV transmission time (ms) : " + str(round(avg_transmission_time, 2)))
        print("(network + GPU) time (ms) : " + str(round(avg_apply_emb_time - avg_ev_lookup_time, 2)))
        print("others time (ms) : " + str(round(avg_latency - avg_apply_emb_time, 2)))

        client.close()
        print("Close connection...")

if __name__ == "__main__":
    run()

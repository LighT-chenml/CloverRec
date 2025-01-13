#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cstddef>
#include <bits/stdc++.h>

// dpu
#include <dpu>

#define DPU_BINARY "pim_dpu"

namespace py = pybind11;

using namespace std;

class PIMEmbStorage
{
    const int MAX_BATCH_SIZE = 256;
    const int DPU_NUM = 16;
    const int TASKLET_NUM = 16;

private:
    long long emb_dim;
    vector<long long> table_sizes;
    vector<float> tables;

    dpu::DpuSet dpuset = dpu::DpuSet::allocate(DPU_NUM);

    struct EmbMetadata
    {
        int dpu_id;
        int dpu_emb_id;
        EmbMetadata() {}
        EmbMetadata(int dpu_id_, int dpu_emb_id)
        {
            dpu_id = dpu_id_;
            dpu_emb_id = dpu_emb_id;
        }
    };
    vector<EmbMetadata> emb_metadata;

public:
    void initialize(long long m, py::list ln, py::list emb_tables)
    {
        emb_dim = m;
        table_sizes = ln.cast<vector<long long>>();
        tables = emb_tables.cast<vector<float>>();

        auto dpus = dpuset.dpus();
        printf("num dpu: %ld\n", dpus.size());

        dpuset.load(DPU_BINARY);

        auto buffer = vector<vector<uint64_t>>(dpus.size(), vector<uint64_t>(1, emb_dim));
        dpuset.copy("emb_dim", buffer);

        // pending
        uint64_t total_emb_num = 0;
        for (auto size : table_sizes)
        {
            total_emb_num += size;
        }
        if (total_emb_num % (dpus.size() * TASKLET_NUM) != 0)
        {
            auto num = total_emb_num / (dpus.size() * TASKLET_NUM) + 1;
            num *= (dpus.size() * TASKLET_NUM);
            num -= total_emb_num;
            table_sizes.back() += num;
            for (int i = 0; i < num; ++i)
                tables.push_back(0.0);
        }
    }

    void init_pim()
    {
        vector<vector<float>> buffer;
        auto &dpus = dpuset.dpus();
        uint64_t emb_num = 0;
        for (auto size : table_sizes)
            emb_num += size;
        uint64_t emb_num_per_dpu = emb_num / dpus.size();

        printf("emb_num %lld\n", emb_num);
        printf("emb_num_per_dpu %lld\n", emb_num_per_dpu);

        uint64_t p = 0;
        for (int i = 0; i < dpus.size(); ++i)
        {
            auto &dpu = dpus[i];
            vector<float> a(2, 0);
            a[0] = 0;
            a[1] = emb_num_per_dpu;
            for (int j = 0; j < emb_num_per_dpu; ++j, ++p)
            {
                emb_metadata.push_back(EmbMetadata(i, j));
                for (int k = 0; k < emb_dim; ++k)
                    a.push_back(tables[p * emb_dim + k]);
            }
            buffer.push_back(a);
        }
        dpuset.copy("buffer", buffer);

        printf("finish transfer!\n");

        dpuset.exec();

        printf("finish loading!\n");
    }

    void sum_emb(vector<float> &x, float *v)
    {
        for (int i = 0; i < x.size(); ++i)
            x[i] += *(v + i);
    }

    struct IndexGroup
    {
        vector<int> dpu_ids;
        vector<vector<float>> evs;
    };

    py::list apply_emb(py::list lS_o, py::list lS_i)
    {
        vector<vector<vector<float>>> result;
        auto offsets = lS_o.cast<vector<vector<long long>>>();
        auto indices = lS_i.cast<vector<vector<long long>>>();

        vector<IndexGroup> index_groups;

        auto dpus = dpuset.dpus();
        vector<vector<uint32_t>> buffer;
        for (int i = 0; i < dpus.size(); ++i)
        {
            vector<uint32_t> a(2, 0);
            a[0] = 1;
            buffer.push_back(a);
        }

        uint32_t batch_size = 0;

        for (int i = 0, table_offset = 0; i < offsets.size(); ++i)
        {
            auto &sparse_offset_group_batch = offsets[i];
            auto &sparse_index_group_batch = indices[i];

            batch_size = sparse_offset_group_batch.size();

            for (int j = 0; j < batch_size; ++j)
            {
                IndexGroup ig;
                auto start = sparse_offset_group_batch[j];
                auto end = j + 1 < batch_size ? sparse_offset_group_batch[j + 1] : sparse_index_group_batch.size();

                int gid = index_groups.size();
                for (int k = start; k < end; ++k)
                {
                    auto index = sparse_index_group_batch[k];
                    index += table_offset;
                    auto dpu_index = emb_metadata[index];
                    buffer[dpu_index.dpu_id].push_back(gid);
                    buffer[dpu_index.dpu_id].push_back(dpu_index.dpu_emb_id);
                    ig.dpu_ids.push_back(dpu_index.dpu_id);
                }
                index_groups.push_back(ig);
            }
            table_offset += table_sizes[i];
        }

        uint32_t max_size = 0;
        for (int i = 0; i < dpus.size(); ++i)
        {
            max_size = max(max_size, (uint32_t)buffer[i].size());
        }
        for (int i = 0; i < dpus.size(); ++i)
        {
            buffer[i][1] = max_size;
            buffer[i].resize(max_size);
        }

        dpuset.copy("buffer", buffer);
        dpuset.exec();

        printf("Finish PIM calc.\n");

        vector<vector<float>> ret_buffer(dpus.size());
        for (int i = 0; i < dpus.size(); ++i)
        {
            buffer[i].resize(2);
        }
        dpuset.copy(buffer, "buffer");

        max_size = 0;
        for (int i = 0; i < dpus.size(); ++i)
        {
            max_size = max(max_size, (uint32_t)ret_buffer[i][0]);
        }
        for (int i = 0; i < dpus.size(); ++i)
        {
            ret_buffer[i].resize(max_size);
        }
        dpuset.copy(ret_buffer, "buffer");

        for (int i = 0; i < dpus.size(); ++i)
        {
            uint32_t num = (ret_buffer[i][0] - 2) / (emb_dim + 2);
            for (int j = 0; j < num; ++j)
            {
                int gid = ret_buffer[i][2 + j * (emb_dim + 2)];
                vector<float> ev;
                for (int k = 0; k < emb_dim; ++k)
                    ev.push_back(ret_buffer[i][2 + j * (emb_dim + 2) + 2 + k]);
            }
        }

        for (int i = 0; i < index_groups.size();)
        {
            vector<vector<float>> evs;
            for (int j = 0; j < batch_size; ++j, ++i)
            {
                vector<float> sum(emb_dim, 0.0);
                for (auto &ev : index_groups[i].evs)
                {
                    for (int k = 0; k < emb_dim; ++k)
                        sum[k] += ev[k];
                }
                evs.push_back(sum);
            }
            result.push_back(evs);
        }

        auto ret = py::cast(result);

        return ret;
    }

    // py::list apply_emb(py::list lS_o, py::list lS_i)
    // {
    //     vector<vector<vector<float>>> result;
    //     auto offsets = lS_o.cast<vector<vector<long long>>>();
    //     auto indices = lS_i.cast<vector<vector<long long>>>();

    //     float *table = tables.data();

    //     for (int i = 0; i < offsets.size(); ++i)
    //     {
    //         auto &sparse_offset_group_batch = offsets[i];
    //         auto &sparse_index_group_batch = indices[i];

    //         auto batch_size = sparse_offset_group_batch.size();

    //         vector<vector<float>> evs;

    //         for (int j = 0; j < batch_size; ++j)
    //         {
    //             auto start = sparse_offset_group_batch[j];
    //             auto end = j + 1 < batch_size ? sparse_offset_group_batch[j + 1] : sparse_index_group_batch.size();

    //             vector<float> sum(emb_dim, 0.0);
    //             for (int k = start; k < end; ++k)
    //             {
    //                 auto index = sparse_index_group_batch[k];
    //                 sum_emb(sum, table + index * emb_dim);
    //             }
    //             evs.push_back(sum);
    //         }
    //         result.push_back(evs);

    //         table += table_sizes[i];
    //     }

    //     auto ret = py::cast(result);

    //     return ret;
    // }
};

PYBIND11_MODULE(pim_module, m)
{
    m.doc() = "PIM Module";

    py::class_<PIMEmbStorage>(m, "PIMEmbStorage")
        .def(py::init<>())
        .def("initialize", &PIMEmbStorage::initialize, "Initialize PIMEmbStorage")
        .def("init_pim", &PIMEmbStorage::init_pim, "")
        .def("apply_emb", &PIMEmbStorage::apply_emb, "Apply Embedding");
}
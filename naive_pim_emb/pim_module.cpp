#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cstddef>
#include <bits/stdc++.h>
#include <chrono>

// dpu
#include <dpu>

#define DPU_BINARY "pim_dpu"

namespace py = pybind11;

using namespace std;

class PIMEmbStorage
{
    const int DPU_NUM = 128;
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
        EmbMetadata(int dpu_id_, int dpu_emb_id_)
        {
            dpu_id = dpu_id_;
            dpu_emb_id = dpu_emb_id_;
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

        auto buffer = vector<uint64_t>(dpus.size(), emb_dim);
        dpuset.copy("emb_dim", buffer, 8);

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
                // printf("gid %d dpu_id %d emb_index %d\n", emb_metadata.size(), i, j);
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
        auto offsets = lS_o.cast<vector<vector<long long>>>();
        auto indices = lS_i.cast<vector<vector<long long>>>();

        auto end2end_start_time = chrono::high_resolution_clock::now();

        auto start_time = chrono::high_resolution_clock::now();

        vector<IndexGroup> index_groups;
        auto dpus = dpuset.dpus();
        vector<vector<uint32_t>> buffer;
        for (int i = 0; i < dpus.size(); ++i)
        {
            vector<uint32_t> a(4, 0);
            a[0] = 1;
            buffer.push_back(a);
        }

        uint32_t batch_size = 0;

        uint32_t total_index_num = 0;

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

                total_index_num += end - start;

                int gid = index_groups.size();
                for (int k = start; k < end; ++k)
                {
                    auto index = sparse_index_group_batch[k];
                    index += table_offset;
                    auto &dpu_index = emb_metadata[index];

                    // printf("gid %d index %lu dpu_id %d emb_index %d\n", gid, index, dpu_index.dpu_id, dpu_index.dpu_emb_id);
                    
                    buffer[dpu_index.dpu_id].push_back(gid);
                    buffer[dpu_index.dpu_id].push_back(dpu_index.dpu_emb_id);
                    ig.dpu_ids.push_back(dpu_index.dpu_id);
                }
                index_groups.push_back(ig);
            }
            table_offset += table_sizes[i];
        }

        // printf("total index num: %u\n", total_index_num);

        auto end_time = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time);
        printf("Task distribution time (ms): %.2lf\n", 1.0 * duration.count() / 1000);

        start_time = chrono::high_resolution_clock::now();

        uint32_t max_size = 0;
        for (int i = 0; i < dpus.size(); ++i)
        {
            max_size = max(max_size, (uint32_t)buffer[i].size());
        }
        for (int i = 0; i < dpus.size(); ++i)
        {
            buffer[i][1] = buffer[i].size();
            buffer[i][2] = index_groups.size();
            buffer[i].resize(max_size);
        }
        dpuset.copy("buffer", buffer);

        end_time = chrono::high_resolution_clock::now();
        duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time);
        printf("Task tranfer time (ms): %.2lf\n", 1.0 * duration.count() / 1000);

        start_time = chrono::high_resolution_clock::now();

        dpuset.exec();

        end_time = chrono::high_resolution_clock::now();
        duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time);
        printf("PIM cal. time (ms): %.2lf\n", 1.0 * duration.count() / 1000);

        // dpuset.log(cout);

        printf("Finish PIM calc.\n");

        start_time = chrono::high_resolution_clock::now();

        vector<vector<float>> ret_buffer(dpus.size());
        for (int i = 0; i < dpus.size(); ++i)
        {
            ret_buffer[i].resize(index_groups.size() * emb_dim);
        }
        dpuset.copy(ret_buffer, "buffer", 8 * 1024 * 1024);

        end_time = chrono::high_resolution_clock::now();
        duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time);
        printf("Result tranfer time (ms): %.2lf\n", 1.0 * duration.count() / 1000);

        start_time = chrono::high_resolution_clock::now();

        vector<vector<vector<float>>> result;

        uint32_t total_cpu_sum_emb = 0;

        for (int i = 0; i < index_groups.size();)
        {
            vector<vector<float>> evs;
            for (int j = 0; j < batch_size; ++j, ++i)
            {
                vector<float> sum(emb_dim, 0.0);
                auto &ig = index_groups[i];
                sort(ig.dpu_ids.begin(), ig.dpu_ids.end());
                auto last = std::unique(ig.dpu_ids.begin(), ig.dpu_ids.end());
                ig.dpu_ids.erase(last, ig.dpu_ids.end());

                total_cpu_sum_emb += ig.dpu_ids.size();

                for (auto &id : ig.dpu_ids)
                {
                    for (int k = 0; k < emb_dim; ++k)
                        sum[k] += ret_buffer[id][i * emb_dim + k];
                }
                evs.push_back(sum);
            }
            result.push_back(evs);
        }

        printf("avg cpu sum emb: %.2lf\n", 1.0 * total_cpu_sum_emb / index_groups.size());

        end_time = chrono::high_resolution_clock::now();
        duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time);
        printf("CPU cal. time (ms): %.2lf\n", 1.0 * duration.count() / 1000);

        auto end2end_end_time = chrono::high_resolution_clock::now();
        auto end2end_duration = chrono::duration_cast<chrono::microseconds>(end2end_end_time - end2end_start_time);
        printf("Total apply emb time (ms): %.2lf\n", 1.0 * end2end_duration.count() / 1000);

        auto ret = py::cast(result);

        return ret;
    }
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
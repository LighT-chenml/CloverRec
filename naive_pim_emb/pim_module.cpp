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
private:
    long long emb_dim;
    vector<long long> table_sizes;
    vector<float> tables;
    
    std::vector<std::vector<float>> buffer;
    dpu::DpuSet dpuset = dpu::DpuSet::allocate();

public:
    void initialize(long long m, py::list ln, py::list emb_tables)
    {
        emb_dim = m;
        table_sizes = ln.cast<vector<long long>>();
        tables = emb_tables.cast<vector<float>>();

        auto dpus = dpuset.dpus();
        printf("num dpu: %ld\n", dpus.size());

        dpuset.load(DPU_BINARY);
    }

    void init_pim()
    {
        auto &dpus = dpuset.dpus();
        float v = 0;
        for (auto &dpu : dpus)
        {
            std::vector<float> a;
            a.push_back(v);
            a.push_back(v + 0.1);
            v += 1.0;
            buffer.push_back(a);
        }
        dpuset.copy("buffer", buffer);

        printf("finish init!\n");
    }

    void run_pim()
    {
        dpuset.exec();

        printf("finish run!\n");
    }

    void output_pim()
    {
        auto &dpus = dpuset.dpus();
        buffer.clear();
        for (auto &dpu : dpus)
        {
            std::vector<float> a(2,0);
            buffer.push_back(a);
        }
        dpuset.copy(buffer, "buffer", 2 * sizeof(float));

        for (int i=0;i<dpus.size();++i)
        {
            printf("%.2lf\n", buffer[i][0]);
        }
    }

    void sum_emb(vector<float> &x, float *v)
    {
        for (int i = 0; i < x.size(); ++i)
            x[i] += *(v + i);
    }

    py::list apply_emb(py::list lS_o, py::list lS_i)
    {
        vector<vector<vector<float>>> result;
        auto offsets = lS_o.cast<vector<vector<long long>>>();
        auto indices = lS_i.cast<vector<vector<long long>>>();

        float *table = tables.data();

        for (int i = 0; i < offsets.size(); ++i)
        {
            auto &sparse_offset_group_batch = offsets[i];
            auto &sparse_index_group_batch = indices[i];

            auto batch_size = sparse_offset_group_batch.size();

            vector<vector<float>> evs;

            for (int j = 0; j < batch_size; ++j)
            {
                auto start = sparse_offset_group_batch[j];
                auto end = j + 1 < batch_size ? sparse_offset_group_batch[j + 1] : sparse_index_group_batch.size();

                vector<float> sum(emb_dim, 0.0);
                for (int k = start; k < end; ++k)
                {
                    auto index = sparse_index_group_batch[k];
                    sum_emb(sum, table + index * emb_dim);
                }
                evs.push_back(sum);
            }
            result.push_back(evs);

            table += table_sizes[i];
        }

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
        .def("run_pim", &PIMEmbStorage::run_pim, "")
        .def("output_pim", &PIMEmbStorage::output_pim, "")
        .def("apply_emb", &PIMEmbStorage::apply_emb, "Apply Embedding");
}
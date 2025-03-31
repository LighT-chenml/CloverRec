#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <cstddef>
#include <bits/stdc++.h>
#include <chrono>

// dpu
#include <dpu>

#define DPU_BINARY "pim_dpu"

namespace py = pybind11;

using namespace std;

vector<vector<uint64_t>> numpy_to_vector_2D(py::array_t<uint64_t> &array)
{

    size_t rows = array.shape(0);
    size_t cols = array.shape(1);

    std::vector<std::vector<uint64_t>> vec(rows, std::vector<uint64_t>(cols));

    auto unchecked_array = array.unchecked<2>();
    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            vec[i][j] = unchecked_array(i, j);
        }
    }

    return vec;
}

py::array_t<float> vector_to_numpy_1D(const std::vector<float> &vec)
{
    return py::array_t<float>(
        vec.size(),
        vec.data());
}

py::array_t<float> vector_to_numpy_2D(const vector<vector<float>> &vec)
{
    size_t dim1 = vec.size();
    size_t dim2 = vec.empty() ? 0 : vec[0].size();

    vector<float> flattened;
    flattened.reserve(dim1 * dim2);
    for (const auto &row : vec)
    {
        flattened.insert(flattened.end(), row.begin(), row.end());
    }

    return py::array_t<float>(
        {dim1, dim2},          // Shape (Three dimensions)
        {dim2 * sizeof(float), // Strides for each dimension (row-major)
         sizeof(float)},
        flattened.data() // Pointer to the flat data
    );
}

py::array_t<float> vector_to_numpy_3D(const vector<vector<vector<float>>> &vec)
{
    size_t dim1 = vec.size();
    size_t dim2 = vec.empty() ? 0 : vec[0].size();
    size_t dim3 = (dim2 == 0 || vec[0].empty()) ? 0 : vec[0][0].size();

    vector<float> flattened;
    flattened.reserve(dim1 * dim2 * dim3);
    for (const auto &mat : vec)
    {
        for (const auto &row : mat)
        {
            flattened.insert(flattened.end(), row.begin(), row.end());
        }
    }

    return py::array_t<float>(
        {dim1, dim2, dim3},           // Shape (Three dimensions)
        {dim2 * dim3 * sizeof(float), // Strides for each dimension (row-major)
         dim3 * sizeof(float),
         sizeof(float)},
        flattened.data() // Pointer to the flat data
    );
}

float cal_vec_avg_float(vector<float> &vec)
{
    if (vec.size() == 0)
        return 0;

    float sum = 0.0;
    for (auto v : vec)
        sum += v;
    return sum / vec.size();
}

float cal_vec_avg_int(vector<uint32_t> &vec)
{
    if (vec.size() == 0)
        return 0;

    float sum = 0.0;
    for (auto v : vec)
        sum += v;
    return sum / vec.size();
}

uint32_t cal_vec_max(vector<uint32_t> &vec)
{
    if (vec.size() == 0)
        return 0;

    uint32_t max_v = vec[0];
    for (auto v : vec)
        max_v = max(max_v, v);
    return max_v;
}

uint32_t cal_vec_min(vector<uint32_t> &vec)
{
    if (vec.size() == 0)
        return 0;

    uint32_t min_v = vec[0];
    for (auto v : vec)
        min_v = min(min_v, v);
    return min_v;
}

class PIMEmbStorage
{
    const int DPU_NUM = 1020;
    const int TASKLET_NUM = 16;

private:
    int transfer_type;

    uint64_t emb_dim;
    vector<uint64_t> table_sizes;
    vector<float> tables;

    dpu::DpuSet dpuset = dpu::DpuSet::allocate(DPU_NUM);

    uint64_t emb_num_per_dpu;

    vector<float> PIM_input_convertion_time;
    vector<float> task_distribution_time;
    vector<float> task_tranfer_time;
    vector<float> PIM_cal_time;
    vector<float> result_transfer_time;
    vector<float> CPU_cal_time;
    vector<float> total_apply_emb_time;

    vector<vector<uint32_t>> index_nums;
    vector<vector<uint32_t>> index_group_nums;
    vector<float> CPU_sum_emb_num;

public:
    void profiling()
    {
        auto avg_PIM_input_convertion_time = cal_vec_avg_float(PIM_input_convertion_time);
        auto avg_task_distribution_time = cal_vec_avg_float(task_distribution_time);
        auto avg_task_tranfer_time = cal_vec_avg_float(task_tranfer_time);
        auto avg_PIM_cal_time = cal_vec_avg_float(PIM_cal_time);
        auto avg_result_transfer_time = cal_vec_avg_float(result_transfer_time);
        auto avg_CPU_cal_time = cal_vec_avg_float(CPU_cal_time);
        auto avg_total_apply_emb_time = cal_vec_avg_float(total_apply_emb_time);

        float avg_index_num = 0;
        float avg_max_index_num = 0;
        float avg_min_index_num = 0;

        for (auto &index_num : index_nums)
        {
            avg_index_num += cal_vec_avg_int(index_num);
            avg_max_index_num += cal_vec_max(index_num);
            avg_min_index_num += cal_vec_min(index_num);
            // printf("%d\n", cal_vec_max(index_num));
        }
        avg_index_num /= index_nums.size();
        avg_max_index_num /= index_nums.size();
        avg_min_index_num /= index_nums.size();

        float avg_index_group_num = 0;
        float avg_max_index_group_num = 0;
        float avg_min_index_group_num = 0;

        for (auto &index_group_num : index_group_nums)
        {
            avg_index_group_num += cal_vec_avg_int(index_group_num);
            avg_max_index_group_num += cal_vec_max(index_group_num);
            avg_min_index_group_num += cal_vec_min(index_group_num);
        }
        avg_index_group_num /= index_group_nums.size();
        avg_max_index_group_num /= index_group_nums.size();
        avg_min_index_group_num /= index_group_nums.size();

        auto avg_CPU_sum_emb_num = cal_vec_avg_float(CPU_sum_emb_num);

        printf("---------------------------------------------------------------------\n");

        printf("avg_PIM_input_convertion_time: %.2lf\n", avg_PIM_input_convertion_time);
        printf("avg_task_distribution_time: %.2lf\n", avg_task_distribution_time);
        printf("avg_task_tranfer_time: %.2lf\n", avg_task_tranfer_time);
        printf("avg_PIM_cal_time: %.2lf\n", avg_PIM_cal_time);
        printf("avg_result_transfer_time: %.2lf\n", avg_result_transfer_time);
        printf("avg_CPU_cal_time: %.2lf\n", avg_CPU_cal_time);
        printf("avg_total_apply_emb_time: %.2lf\n", avg_total_apply_emb_time);

        printf("\n");

        printf("avg_index_num: %.2lf\n", avg_index_num);
        printf("avg_max_index_num: %.2lf\n", avg_max_index_num);
        printf("avg_min_index_num: %.2lf\n", avg_min_index_num);
        printf("avg_index_group_num: %.2lf\n", avg_index_group_num);
        printf("avg_max_index_group_num: %.2lf\n", avg_max_index_group_num);
        printf("avg_min_index_group_num: %.2lf\n", avg_min_index_group_num);
        printf("avg_CPU_sum_emb_num: %.2lf\n", avg_CPU_sum_emb_num);

        printf("---------------------------------------------------------------------\n");
    }

    void clear_profiling_data()
    {
        vector<float>().swap(PIM_input_convertion_time);
        vector<float>().swap(task_distribution_time);
        vector<float>().swap(task_tranfer_time);
        vector<float>().swap(PIM_cal_time);
        vector<float>().swap(result_transfer_time);
        vector<float>().swap(CPU_cal_time);
        vector<float>().swap(total_apply_emb_time);

        vector<vector<uint32_t>>().swap(index_nums);
        vector<vector<uint32_t>>().swap(index_group_nums);
        vector<float>().swap(CPU_sum_emb_num);
    }

    void initialize(uint64_t m, py::array_t<uint64_t> &ln, py::array_t<float> &emb_tables)
    {
        transfer_type = 0;

        emb_dim = m;
        table_sizes.resize(ln.size());
        copy(ln.data(), ln.data() + ln.size(), table_sizes.begin());
        tables.resize(emb_tables.size());
        copy(emb_tables.data(), emb_tables.data() + emb_tables.size(), tables.begin());

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
        emb_num_per_dpu = emb_num / dpus.size();

        printf("emb_num %lld\n", emb_num);
        printf("emb_num_per_dpu %lld\n", emb_num_per_dpu);

        auto start_time = chrono::high_resolution_clock::now();

        uint64_t p = 0;
        for (int i = 0; i < dpus.size(); ++i)
        {
            auto &dpu = dpus[i];
            vector<float> a(2, 0);
            a[0] = 0;
            a[1] = emb_num_per_dpu;
            for (int j = 0; j < emb_num_per_dpu; ++j, ++p)
            {
                for (int k = 0; k < emb_dim; ++k)
                    a.push_back(tables[p * emb_dim + k]);
            }
            buffer.push_back(a);
        }

        auto end_time = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time);
        printf("Emb distribution time (ms): %.2lf\n", 1.0 * duration.count() / 1000);

        start_time = chrono::high_resolution_clock::now();

        dpuset.copy("buffer", 2 * 1024 * 1024, buffer);

        end_time = chrono::high_resolution_clock::now();
        duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time);
        printf("Emb loading time (ms): %.2lf\n", 1.0 * duration.count() / 1000);

        // vector<float>().swap(tables);
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

    py::array_t<float> apply_emb(py::array_t<uint64_t> &lS_o, py::array_t<uint64_t> &lS_i)
    {
        auto start_convertion_time = chrono::high_resolution_clock::now();

        auto offsets = numpy_to_vector_2D(lS_o);
        auto indices = numpy_to_vector_2D(lS_i);

        auto end_convertion_time = chrono::high_resolution_clock::now();
        auto convertion_duration = chrono::duration_cast<chrono::microseconds>(end_convertion_time - start_convertion_time);
        // printf("PIM module input convertion time (ms): %.2lf\n", 1.0 * convertion_duration.count() / 1000);
        PIM_input_convertion_time.push_back(1.0 * convertion_duration.count() / 1000);

        auto end2end_start_time = chrono::high_resolution_clock::now();

        auto start_time = chrono::high_resolution_clock::now();

        vector<IndexGroup> index_groups;
        auto dpus = dpuset.dpus();
        vector<vector<uint32_t>> buffer;
        for (int i = 0; i < dpus.size(); ++i)
        {
            vector<uint32_t> a(4 + offsets.size() * offsets[0].size(), 0);
            a[0] = 0;
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
                    auto dpu_id = index / emb_num_per_dpu;
                    auto dpu_emb_id = index % emb_num_per_dpu;

                    if (buffer[dpu_id][4 + gid] == 0) buffer[dpu_id][2]++;
                    buffer[dpu_id].push_back(gid);
                    buffer[dpu_id].push_back(dpu_emb_id);
                    buffer[dpu_id][4 + gid] = 1;
                    ig.dpu_ids.push_back(dpu_id);
                }
                index_groups.push_back(ig);
            }
            table_offset += table_sizes[i];
        }

        vector<uint32_t> index_group_num;
        vector<uint32_t> index_num;
        uint32_t max_size = 0;
        for (int i = 0; i < dpus.size(); ++i)
        {
            max_size = max(max_size, (uint32_t)buffer[i].size());
            index_num.push_back(((uint32_t)buffer[i].size() - 4 - index_groups.size()) / 2);

            uint32_t sum = 0;
            for (int j = 0; j < index_groups.size(); ++j)
            {
                uint32_t v = buffer[i][4 + j];
                if (v)
                {
                    buffer[i][4 + j] = sum;
                    sum += v;
                }
            }
            index_group_num.push_back(sum);
        }
        index_group_nums.push_back(index_group_num);
        index_nums.push_back(index_num);

        auto end_time = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time);
        // printf("Task distribution time (ms): %.2lf\n", 1.0 * duration.count() / 1000);
        task_distribution_time.push_back(1.0 * duration.count() / 1000);

        start_time = chrono::high_resolution_clock::now();

        if (transfer_type == 0)
        {
            for (int i = 0; i < dpus.size(); ++i)
            {
                buffer[i][1] = ((uint32_t)buffer[i].size() - 4 - index_groups.size()) / 2;
                buffer[i].resize(max_size);
            }
            dpuset.copy("buffer", buffer);
        }
        else if (transfer_type == 1)
        {
            for (int i = 0; i < dpus.size(); ++i)
            {
                buffer[i][1] = ((uint32_t)buffer[i].size() - 4 - index_groups.size()) / 2;
                dpuset.dpus()[i]->copy("buffer", buffer[i]);
            }
        }

        end_time = chrono::high_resolution_clock::now();
        duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time);
        // printf("Task tranfer time (ms): %.2lf\n", 1.0 * duration.count() / 1000);
        task_tranfer_time.push_back(1.0 * duration.count() / 1000);

        start_time = chrono::high_resolution_clock::now();

        dpuset.exec();

        end_time = chrono::high_resolution_clock::now();
        duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time);
        // printf("PIM cal. time (ms): %.2lf\n", 1.0 * duration.count() / 1000);
        PIM_cal_time.push_back(1.0 * duration.count() / 1000);

        // dpuset.log(cout);

        start_time = chrono::high_resolution_clock::now();

        vector<vector<float>> ret_buffer(dpus.size());
        if (transfer_type == 0)
        {
            max_size = 0;
            for (int i = 0; i < dpus.size(); ++i)
            {
                max_size = max(max_size, index_group_num[i]);
            }
            for (int i = 0; i < dpus.size(); ++i)
            {
                ret_buffer[i].resize(max_size * emb_dim);
            }
            dpuset.copy(ret_buffer, "buffer", 1 * 1024 * 1024);
        }
        else if (transfer_type == 1)
        {
            for (int i = 0; i < dpus.size(); ++i)
            {
                vector<vector<float>> a;
                a.push_back(vector<float>(index_group_num[i] * emb_dim));
                dpuset.dpus()[i]->copy(a, "buffer", 1 * 1024 * 1024);
                ret_buffer[i] = a[0];
            }
        }

        end_time = chrono::high_resolution_clock::now();
        duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time);
        // printf("Result transfer time (ms): %.2lf\n", 1.0 * duration.count() / 1000);
        result_transfer_time.push_back(1.0 * duration.count() / 1000);

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
                    uint32_t result_index = buffer[id][4 + i];
                    for (int k = 0; k < emb_dim; ++k)
                        sum[k] += ret_buffer[id][result_index * emb_dim + k];
                }
                evs.push_back(sum);
            }
            result.push_back(evs);
        }

        // printf("avg CPU sum emb: %.2lf\n", 1.0 * total_cpu_sum_emb / index_groups.size());
        CPU_sum_emb_num.push_back(1.0 * total_cpu_sum_emb / index_groups.size());

        end_time = chrono::high_resolution_clock::now();
        duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time);
        // printf("CPU cal. time (ms): %.2lf\n", 1.0 * duration.count() / 1000);
        CPU_cal_time.push_back(1.0 * duration.count() / 1000);

        auto end2end_end_time = chrono::high_resolution_clock::now();
        auto end2end_duration = chrono::duration_cast<chrono::microseconds>(end2end_end_time - end2end_start_time);
        // printf("Total apply emb time (ms): %.2lf\n", 1.0 * end2end_duration.count() / 1000);
        total_apply_emb_time.push_back(1.0 * end2end_duration.count() / 1000);

        // start_time = chrono::high_resolution_clock::now();

        auto ret = vector_to_numpy_3D(result);

        // end_time = chrono::high_resolution_clock::now();
        // duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time);
        // printf("PIM module result convertion time (ms): %.2lf\n", 1.0 * duration.count() / 1000);

        return ret;
    }
};

PYBIND11_MODULE(pim_module, m)
{
    m.doc() = "PIM Module";

    py::class_<PIMEmbStorage>(m, "PIMEmbStorage")
        .def(py::init<>())
        .def("profiling", &PIMEmbStorage::profiling, "Profiling")
        .def("clear_profiling_data", &PIMEmbStorage::clear_profiling_data, "Clear Profiling Data")
        .def("initialize", &PIMEmbStorage::initialize, "Initialize PIMEmbStorage")
        .def("init_pim", &PIMEmbStorage::init_pim, "Loading Embs to PIM")
        .def("apply_emb", &PIMEmbStorage::apply_emb, "Apply Embedding");
}
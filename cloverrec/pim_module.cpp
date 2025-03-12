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

struct FreqSetItem
{
    uint32_t id;
    uint32_t freq;
    FreqSetItem(uint32_t id_, uint32_t freq_)
    {
        id = id_;
        freq = freq_;
    }
    bool operator<(const FreqSetItem &t) const { return freq < t.freq || freq == t.freq && id < t.id; }
};

class PIMEmbStorage
{
    const static int DPU_NUM = 1020;
    const uint32_t TASKLET_NUM = 16;
    const float REDUNDANT_RATIO = 0.005;
    const float SPLIT_EMB_NUM_RATIO = 0.0005;
    const float MERGE_EMB_NUM_RATIO = 0.0001;

    const uint32_t SELECT_EMB_ITER = 300;
    const uint32_t SELECT_DPU_ITER = 100;

private:
    int exec_type;     // 0: sync 1: rank async 2: dpu async
    int transfer_type; // 0: all 1: rank 2: dpu

    // dpu
    dpu::DpuSet dpuset = dpu::DpuSet::allocate(DPU_NUM);
    pair<int, int> dpu_rank_ids[DPU_NUM];

    // emb data
    uint64_t emb_dim;
    uint64_t total_emb_num;
    uint64_t emb_num_per_dpu;
    uint64_t global_total_index_num;
    vector<uint64_t> table_sizes;
    vector<float> tables;

    // redundant-based load balance
    uint32_t apply_emb_num;
    uint32_t freq_base;
    uint32_t *emb_freqs;
    uint32_t dpu_freqs[DPU_NUM];
    uint32_t dpu_emb_nums[DPU_NUM];
    vector<uint32_t> dpu_redundant_emb_ids[DPU_NUM];
    uint32_t dpu_table_num[DPU_NUM];
    vector<uint32_t> dpu_table_emb_nums[DPU_NUM];

    uint32_t split_emb_num;
    uint32_t merge_emb_num;
    uint32_t max_redundant_emb_size;
    uint32_t cur_redundant_emb_size;
    struct RedundantEmb
    {
        vector<uint32_t> dpu_ids;
        vector<uint32_t> dpu_emb_ids;
    };
    unordered_map<uint32_t, RedundantEmb> redundant_embs;
    vector<uint32_t> redundant_emb_ids;

    // profiling data
    vector<float> PIM_input_convertion_time;
    vector<float> task_dispatch_time;
    vector<float> task_tranfer_time;
    vector<float> PIM_cal_time;
    vector<float> result_transfer_time;
    vector<float> CPU_cal_time;
    vector<float> select_hot_dpu_time;
    vector<float> select_cold_dpu_time;
    vector<float> select_hot_emb_time;
    vector<float> copy_emb_time;
    vector<float> split_emb_time;
    vector<float> merge_emb_time;
    vector<float> total_apply_emb_time;

    vector<vector<uint32_t>> index_nums;
    vector<vector<uint32_t>> index_group_nums;
    vector<float> CPU_sum_emb_num;

public:
    void profiling()
    {
        auto avg_PIM_input_convertion_time = cal_vec_avg_float(PIM_input_convertion_time);
        auto avg_task_dispatch_time = cal_vec_avg_float(task_dispatch_time);
        auto avg_task_tranfer_time = cal_vec_avg_float(task_tranfer_time);
        auto avg_PIM_cal_time = cal_vec_avg_float(PIM_cal_time);
        auto avg_result_transfer_time = cal_vec_avg_float(result_transfer_time);
        auto avg_CPU_cal_time = cal_vec_avg_float(CPU_cal_time);
        auto avg_select_hot_dpu_time = cal_vec_avg_float(select_hot_dpu_time);
        auto avg_select_cold_dpu_time = cal_vec_avg_float(select_cold_dpu_time);
        auto avg_select_hot_emb_time = cal_vec_avg_float(select_hot_emb_time);
        auto avg_copy_emb_time = cal_vec_avg_float(copy_emb_time);
        auto avg_split_emb_time = cal_vec_avg_float(split_emb_time);
        auto avg_merge_emb_time = cal_vec_avg_float(merge_emb_time);
        auto avg_total_apply_emb_time = cal_vec_avg_float(total_apply_emb_time);

        float avg_index_num = 0;
        float avg_max_index_num = 0;
        float avg_min_index_num = 0;

        int cnt = 0;

        for (auto &index_num : index_nums)
        {
            avg_index_num += cal_vec_avg_int(index_num);
            avg_max_index_num += cal_vec_max(index_num);
            avg_min_index_num += cal_vec_min(index_num);

            ++cnt;
            if (cnt % 10 == 0)
            {
                printf("cnt %d avg_max_index_num: %.2lf\n", cnt, avg_max_index_num / cnt);
            }
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

        // sort(emb_freqs, emb_freqs + total_emb_num);
        // for (int i = 0; i < 100; ++i)
        //     printf("%d emb_freq %u\n", i, emb_freqs[total_emb_num - 1 - i]);

        printf("---------------------------------------------------------------------\n");

        printf("avg_PIM_input_convertion_time: %.2lf\n", avg_PIM_input_convertion_time);
        printf("avg_task_dispatch_time: %.2lf\n", avg_task_dispatch_time);
        printf("avg_task_tranfer_time: %.2lf\n", avg_task_tranfer_time);
        printf("avg_PIM_cal_time: %.2lf\n", avg_PIM_cal_time);
        printf("avg_result_transfer_time: %.2lf\n", avg_result_transfer_time);
        printf("avg_CPU_cal_time: %.2lf\n", avg_CPU_cal_time);
        printf("avg_select_hot_dpu_time: %.2lf\n", avg_select_hot_dpu_time);
        printf("avg_select_cold_dpu_time: %.2lf\n", avg_select_cold_dpu_time);
        printf("avg_select_hot_emb_time: %.2lf\n", avg_select_hot_emb_time);
        printf("avg_copy_emb_time: %.2lf\n", avg_copy_emb_time);
        printf("avg_split_emb_time: %.2lf\n", avg_split_emb_time);
        printf("avg_merge_emb_time: %.2lf\n", avg_merge_emb_time);
        printf("avg_total_apply_emb_time: %.2lf\n", avg_total_apply_emb_time);

        printf("\n");

        printf("avg_index_num: %.2lf\n", avg_index_num);
        printf("avg_max_index_num: %.2lf\n", avg_max_index_num);
        printf("avg_min_index_num: %.2lf\n", avg_min_index_num);
        printf("avg_index_group_num: %.2lf\n", avg_index_group_num);
        printf("avg_max_index_group_num: %.2lf\n", avg_max_index_group_num);
        printf("avg_min_index_group_num: %.2lf\n", avg_min_index_group_num);
        printf("avg_CPU_sum_emb_num: %.2lf\n", avg_CPU_sum_emb_num);
        printf("cur_redundant_emb_size: %u\n", cur_redundant_emb_size);
        printf("split_emb_num: %u\n", split_emb_num);
        printf("merge_emb_num: %u\n", merge_emb_num);
        printf("redundant_emb_num: %u\n", (uint32_t)redundant_emb_ids.size());

        printf("---------------------------------------------------------------------\n");
    }

    void clear_profiling_data()
    {
        auto &dpus = dpuset.dpus();

        vector<float>().swap(PIM_input_convertion_time);
        vector<float>().swap(task_dispatch_time);
        vector<float>().swap(task_tranfer_time);
        vector<float>().swap(PIM_cal_time);
        vector<float>().swap(result_transfer_time);
        vector<float>().swap(CPU_cal_time);
        vector<float>().swap(select_hot_dpu_time);
        vector<float>().swap(select_cold_dpu_time);
        vector<float>().swap(select_hot_emb_time);
        vector<float>().swap(split_emb_time);
        vector<float>().swap(total_apply_emb_time);

        vector<vector<uint32_t>>().swap(index_nums);
        vector<vector<uint32_t>>().swap(index_group_nums);
        vector<float>().swap(CPU_sum_emb_num);

        global_total_index_num = 0;

        apply_emb_num = 0;
        freq_base = 100;
        redundant_embs.clear();
        vector<uint32_t>().swap(redundant_emb_ids);
        split_emb_num = 0;
        merge_emb_num = 0;
        cur_redundant_emb_size = 0;
        for (int i = 0; i < total_emb_num; ++i)
            emb_freqs[i] = 0;
        for (int i = 0; i < dpus.size(); ++i)
        {
            dpu_freqs[i] = 0;
            dpu_emb_nums[i] = emb_num_per_dpu;
            vector<uint32_t>().swap(dpu_redundant_emb_ids[i]);
            dpu_table_num[i] = 1;
            int L = i * emb_num_per_dpu;
            int R = (i + 1) * emb_num_per_dpu;
            for (int j = 0, l = 0, r = 0; j < table_sizes.size(); ++j)
            {
                r = l + table_sizes[j];
                dpu_table_emb_nums[i][j] = max(0, min(R, r) - max(L, l));
                l = r;
            }
        }
    }

    void initialize(uint64_t m, py::array_t<uint64_t> &ln, py::array_t<float> &emb_tables)
    {
        exec_type = 1;
        transfer_type = 2;

        if (exec_type != 0)
            transfer_type = exec_type;

        emb_dim = m;
        table_sizes.resize(ln.size());
        copy(ln.data(), ln.data() + ln.size(), table_sizes.begin());
        tables.resize(emb_tables.size());
        copy(emb_tables.data(), emb_tables.data() + emb_tables.size(), tables.begin());

        auto &dpus = dpuset.dpus();
        printf("num dpu: %ld\n", dpus.size());

        auto &ranks = dpuset.ranks();
        for (int i = 0, l = 0, r; i < ranks.size(); ++i, l = r)
        {
            r = l + ranks[i]->dpus().size();
            for (int j = l, k = 0; j < r; ++j, ++k)
            {
                dpu_rank_ids[j] = make_pair(i, k);
            }
        }

        dpuset.load(DPU_BINARY);

        auto buffer = vector<uint64_t>(dpus.size(), emb_dim);
        dpuset.copy("emb_dim", buffer, 8);

        // pending
        total_emb_num = 0;
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

        global_total_index_num = 0;

        max_redundant_emb_size = 1.0 * total_emb_num * REDUNDANT_RATIO;
        cur_redundant_emb_size = 0;
        split_emb_num = 0;
        merge_emb_num = 0;

        printf("max_redundant_emb_size: %u\n", max_redundant_emb_size);
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

        apply_emb_num = 0;
        freq_base = 100;
        emb_freqs = new uint32_t[total_emb_num];
        for (int i = 0; i < total_emb_num; ++i)
            emb_freqs[i] = 0;
        for (int i = 0; i < dpus.size(); ++i)
        {
            dpu_freqs[i] = 0;
            dpu_emb_nums[i] = emb_num_per_dpu;
            dpu_table_num[i] = 1;
            dpu_table_emb_nums[i].resize(table_sizes.size());
            int L = i * emb_num_per_dpu;
            int R = (i + 1) * emb_num_per_dpu;
            for (int j = 0, l = 0, r = 0; j < table_sizes.size(); ++j)
            {
                r = l + table_sizes[j];
                dpu_table_emb_nums[i][j] = max(0, min(R, r) - max(L, l));
                l = r;
            }
        }

        auto start_time = chrono::high_resolution_clock::now();

        uint64_t p = 0;
        for (int i = 0; i < dpus.size(); ++i)
        {
            vector<float> a;
            for (int j = 0; j < emb_num_per_dpu; ++j, ++p)
            {
                for (int k = 0; k < emb_dim; ++k)
                    a.push_back(tables[p * emb_dim + k]);
            }
            buffer.push_back(a);
        }

        auto end_time = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time);
        printf("Emb dispatch time (ms): %.2lf\n", 1.0 * duration.count() / 1000);

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

    uint32_t cal_table_id(uint32_t emb_id)
    {
        uint32_t sum = 0;
        for (int i = 0; i < table_sizes.size(); ++i)
        {
            sum += table_sizes[i];
            if (sum > emb_id)
                return i;
        }
        return table_sizes.size();
    }

    uint32_t cal_emb_freq(uint32_t emb_id)
    {
        auto it = redundant_embs.find(emb_id);
        if (it != redundant_embs.end())
        {
            return emb_freqs[emb_id] / (it->second).dpu_ids.size();
        }
        return emb_freqs[emb_id];
    }

    uint32_t cal_emb_delta_freq(uint32_t emb_id)
    {
        auto it = redundant_embs.find(emb_id);
        if (it != redundant_embs.end())
        {
            auto copy_num = (it->second).dpu_ids.size();
            return emb_freqs[emb_id] / copy_num - emb_freqs[emb_id] / (copy_num + 1);
        }
        return emb_freqs[emb_id] / 2;
    }

    uint32_t select_hot_emb(uint32_t dpu_id)
    {
        static std::mt19937 random_num_generation(std::random_device{}());

        uint32_t ret_emb_id = total_emb_num;
        uint32_t ret_emb_freq = 0;
        for (int i = 0; i < min(dpu_emb_nums[dpu_id], SELECT_EMB_ITER); ++i)
        {
            uint32_t p;

            if (dpu_emb_nums[dpu_id] <= SELECT_EMB_ITER)
                p = i;
            else
                p = random_num_generation() % dpu_emb_nums[dpu_id];

            uint32_t emb_id;
            if (p < emb_num_per_dpu)
            {
                emb_id = dpu_id * emb_num_per_dpu + p;
            }
            else
            {
                emb_id = dpu_redundant_emb_ids[dpu_id][p - emb_num_per_dpu];
            }

            uint32_t emb_freq = cal_emb_delta_freq(emb_id); // priority

            if (ret_emb_id == total_emb_num || ret_emb_freq < emb_freq)
            {
                ret_emb_id = emb_id;
                ret_emb_freq = emb_freq;
            }
        }
        return ret_emb_id;
    }

    bool compare_cold_dpu(uint32_t dpu_a, uint32_t dpu_b, uint32_t emb_table_id)
    {
        if (dpu_table_num[dpu_a] + (dpu_table_emb_nums[dpu_a][emb_table_id] == 0) != dpu_table_num[dpu_b] + (dpu_table_emb_nums[dpu_b][emb_table_id] == 0))
        {
            return dpu_table_num[dpu_b] + (dpu_table_emb_nums[dpu_b][emb_table_id] == 0) < dpu_table_num[dpu_a] + (dpu_table_emb_nums[dpu_a][emb_table_id] == 0);
        }
        if (dpu_table_emb_nums[dpu_a][emb_table_id] != dpu_table_emb_nums[dpu_b][emb_table_id])
        {
            return dpu_table_emb_nums[dpu_b][emb_table_id] > dpu_table_emb_nums[dpu_a][emb_table_id];
        }
        return dpu_freqs[dpu_b] < dpu_freqs[dpu_a];
    }

    uint32_t select_cold_dpu(uint32_t emb_id, uint32_t emb_table_id, uint32_t freq_limit, uint32_t add_freq)
    {
        bool static flag[DPU_NUM] = {false};
        auto &dpus = dpuset.dpus();
        auto it = redundant_embs.find(emb_id);
        if (it != redundant_embs.end())
        {
            for (auto id : (it->second).dpu_ids)
                flag[id] = true;
        }
        else
            flag[emb_id / emb_num_per_dpu] = true;

        uint32_t ret_dpu_id = dpus.size();
        for (int i = 0; i < dpus.size(); ++i)
        {
            // int p = rand() % dpus.size();
            int p = i;
            if (flag[p])
                continue;
            if (dpu_freqs[p] + add_freq >= freq_limit)
                continue;
            if (ret_dpu_id == dpus.size() || compare_cold_dpu(ret_dpu_id, p, emb_table_id))
            {
                ret_dpu_id = p;
            }
        }
        if (it != redundant_embs.end())
        {
            for (auto id : (it->second).dpu_ids)
                flag[id] = false;
        }
        else
            flag[emb_id / emb_num_per_dpu] = false;

        return ret_dpu_id;
    }

    bool compare_hot_dpu(uint32_t dpu_a, uint32_t dpu_b)
    {
        return dpu_freqs[dpu_b] > dpu_freqs[dpu_a];
    }

    uint32_t select_hot_dpu(vector<uint32_t> &dpu_ids)
    {
        auto &dpus = dpuset.dpus();
        uint32_t hot_dpu_id = dpus.size();

        for (int i = 0; i < dpu_ids.size(); ++i)
        {
            int p = dpu_ids[i];
            // int p = rand() % dpus.size();
            if (hot_dpu_id == dpus.size() || compare_hot_dpu(hot_dpu_id, p))
            {
                hot_dpu_id = p;
            }
        }

        return hot_dpu_id;
    }

    void split_emb()
    {
        auto total_start_time = chrono::high_resolution_clock::now();

        auto &dpus = dpuset.dpus();

        float sum_select_hot_dpu_time = 0;
        float sum_select_cold_dpu_time = 0;
        float sum_select_hot_emb_time = 0;
        float sum_copy_emb_time = 0;

        vector<uint32_t> dpu_ids(dpus.size());
        for (int i = 0; i < dpus.size(); ++i)
        {
            dpu_ids[i] = i;
        }

        for (int _ = 0; _ < split_emb_num; ++_)
        {
            if (cur_redundant_emb_size >= max_redundant_emb_size)
                break;

            auto start_time = chrono::high_resolution_clock::now();

            uint32_t hot_dpu_id = select_hot_dpu(dpu_ids);
            if (hot_dpu_id == dpus.size())
                continue;

            auto end_time = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time);
            sum_select_hot_dpu_time += 1.0 * duration.count() / 1000;

            start_time = chrono::high_resolution_clock::now();

            uint32_t emb_id = select_hot_emb(hot_dpu_id);
            if (emb_id == total_emb_num)
                continue;

            end_time = chrono::high_resolution_clock::now();
            duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time);
            sum_select_hot_emb_time += 1.0 * duration.count() / 1000;

            uint32_t emb_table_id = cal_table_id(emb_id);
            auto it = redundant_embs.find(emb_id);
            uint32_t delta_freq = cal_emb_delta_freq(emb_id);

            start_time = chrono::high_resolution_clock::now();

            uint32_t cold_dpu_id = select_cold_dpu(emb_id, emb_table_id, dpu_freqs[hot_dpu_id], it == redundant_embs.end() ? (emb_freqs[emb_id] / 2) : (emb_freqs[emb_id] / ((it->second).dpu_ids.size() + 1)));
            if (cold_dpu_id == dpus.size())
                continue;

            end_time = chrono::high_resolution_clock::now();
            duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time);
            sum_select_cold_dpu_time += 1.0 * duration.count() / 1000;

            // printf("hot_dpu_id %u freq %u dpu_emb_nums %u\n", hot_dpu_id, dpu_freqs[hot_dpu_id], dpu_emb_nums[hot_dpu_id]);
            // printf("emb_id %u freq %u\n", emb_id, cal_emb_freq(emb_id));
            // printf("cold_dpu_id %u freq %u dpu_emb_nums %u\n", cold_dpu_id, dpu_freqs[cold_dpu_id], dpu_emb_nums[cold_dpu_id]);

            start_time = chrono::high_resolution_clock::now();

            vector<float> buffer(emb_dim);
            for (int i = 0; i < emb_dim; ++i)
                buffer[i] = tables[emb_id * emb_dim + i];

            dpuset.dpus()[cold_dpu_id]->copy("buffer", 2 * 1024 * 1024 + dpu_emb_nums[cold_dpu_id] * emb_dim * 4, buffer);

            end_time = chrono::high_resolution_clock::now();
            duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time);
            sum_copy_emb_time += 1.0 * duration.count() / 1000;

            if (it == redundant_embs.end())
            {
                redundant_embs[emb_id] = RedundantEmb();
                redundant_embs[emb_id].dpu_ids.push_back(hot_dpu_id);
                redundant_embs[emb_id].dpu_emb_ids.push_back(emb_id % emb_num_per_dpu);
                redundant_emb_ids.push_back(emb_id);
            }
            auto &redundant_emb = redundant_embs[emb_id];
            cur_redundant_emb_size++;
            redundant_emb.dpu_ids.push_back(cold_dpu_id);
            redundant_emb.dpu_emb_ids.push_back(dpu_emb_nums[cold_dpu_id]++);
            dpu_redundant_emb_ids[cold_dpu_id].push_back(emb_id);
            if (dpu_table_emb_nums[cold_dpu_id][emb_table_id] == 0)
                dpu_table_num[cold_dpu_id]++;
            dpu_table_emb_nums[cold_dpu_id][emb_table_id]++;

            // printf("redundant_emb_num %u delta_freq %u\n", redundant_emb.dpu_ids.size(), delta_freq);

            for (auto &id : redundant_emb.dpu_ids)
            {
                if (id == cold_dpu_id)
                    continue;
                dpu_freqs[id] -= delta_freq;
            }
            dpu_freqs[cold_dpu_id] += emb_freqs[emb_id] / redundant_emb.dpu_ids.size();
        }

        select_hot_dpu_time.push_back(sum_select_hot_dpu_time);
        select_cold_dpu_time.push_back(sum_select_cold_dpu_time);
        select_hot_emb_time.push_back(sum_select_hot_emb_time);
        copy_emb_time.push_back(sum_copy_emb_time);

        auto total_end_time = chrono::high_resolution_clock::now();
        auto total_duration = chrono::duration_cast<chrono::microseconds>(total_end_time - total_start_time);
        // printf("Split emb time (ms): %.2lf\n", 1.0 * duration.count() / 1000);
        split_emb_time.push_back(1.0 * total_duration.count() / 1000);
    }

    uint32_t select_cold_redundant_emb()
    {
        static std::mt19937 random_num_generation(std::random_device{}());

        uint32_t ret_emb_id = total_emb_num;
        uint32_t ret_delta_freq = 0;

        for (int i = 0; i < min((uint32_t)redundant_emb_ids.size(), SELECT_EMB_ITER); ++i)
        {
            int p;
            if (redundant_emb_ids.size() <= SELECT_EMB_ITER)
            {
                p = i;
            }
            else
            {
                p = random_num_generation() % redundant_emb_ids.size();
            }
            uint32_t emb_id = redundant_emb_ids[p];

            uint32_t copy_num = redundant_embs[emb_id].dpu_ids.size();
            uint32_t delta_freq = emb_freqs[emb_id] / (copy_num - 1) - emb_freqs[emb_id] / copy_num;

            if (ret_emb_id == total_emb_num || ret_delta_freq > delta_freq)
            {
                ret_emb_id = emb_id;
                ret_delta_freq = delta_freq;
            }
        }

        return ret_emb_id;
    }

    void merge_emb()
    {
        auto total_start_time = chrono::high_resolution_clock::now();

        auto &dpus = dpuset.dpus();

        for (int _ = 0; _ < merge_emb_num; ++_)
        {
            if (cur_redundant_emb_size < max_redundant_emb_size * 0.95)
                break;

            uint32_t emb_id = select_cold_redundant_emb();
            if (emb_id == total_emb_num)
                continue;

            auto &redundant_emb = redundant_embs[emb_id];

            vector<uint32_t> dpu_ids = redundant_emb.dpu_ids;
            swap(dpu_ids[0], *dpu_ids.rbegin());
            dpu_ids.pop_back();

            uint32_t hot_dpu_id = select_hot_dpu(dpu_ids);

            uint32_t copy_num = redundant_embs[emb_id].dpu_ids.size();
            uint32_t delta_freq = emb_freqs[emb_id] / (copy_num - 1) - emb_freqs[emb_id] / copy_num;

            bool flag = true;
            for (auto id : redundant_emb.dpu_ids)
            {
                if (id == hot_dpu_id)
                    continue;
                if (dpu_freqs[hot_dpu_id] < dpu_freqs[id] + delta_freq)
                {
                    flag = false;
                    break;
                }
            }
            if (!flag)
                continue;

            --cur_redundant_emb_size;

            if (redundant_emb.dpu_ids.size() == 2)
            {
                for (int i = 0; i < redundant_emb_ids.size(); ++i)
                    if (redundant_emb_ids[i] == emb_id)
                    {
                        swap(redundant_emb_ids[i], *redundant_emb_ids.rbegin());
                        redundant_emb_ids.pop_back();
                        break;
                    }
                dpu_freqs[redundant_emb.dpu_ids[0]] += emb_freqs[emb_id] / 2;
                dpu_freqs[redundant_emb.dpu_ids[1]] -= emb_freqs[emb_id] / 2;
                auto it = redundant_embs.find(emb_id);
                redundant_embs.erase(it);
                continue;
            }

            for (int i = 0; i < redundant_emb.dpu_ids.size(); ++i)
                if (redundant_emb.dpu_ids[i] == hot_dpu_id)
                {
                    swap(redundant_emb.dpu_ids[i], *redundant_emb.dpu_ids.rbegin());
                    redundant_emb.dpu_ids.pop_back();
                    swap(redundant_emb.dpu_emb_ids[i], *redundant_emb.dpu_emb_ids.rbegin());
                    redundant_emb.dpu_emb_ids.pop_back();
                    break;
                }

            for (auto id : redundant_emb.dpu_ids)
            {
                dpu_freqs[id] += delta_freq;
            }
            dpu_freqs[hot_dpu_id] -= emb_freqs[emb_id] / (redundant_emb.dpu_ids.size() + 1);
        }

        auto total_end_time = chrono::high_resolution_clock::now();
        auto total_duration = chrono::duration_cast<chrono::microseconds>(total_end_time - total_start_time);
        merge_emb_time.push_back(1.0 * total_duration.count() / 1000);
    }

    struct IndexGroup
    {
        vector<int> dpu_ids;
        vector<vector<float>> evs;
    };

    py::array_t<float> apply_emb(py::array_t<uint64_t> &lS_o, py::array_t<uint64_t> &lS_i)
    {
        static std::mt19937 random_num_generation(std::random_device{}());

        auto &dpus = dpuset.dpus();
        auto &ranks = dpuset.ranks();

        apply_emb_num++;
        if (apply_emb_num % 20 == 0)
            freq_base++; // aging

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
        vector<vector<uint32_t>> buffer;
        vector<vector<vector<uint32_t>>> rank_buffer(ranks.size());
        for (int i = 0; i < dpus.size(); ++i)
        {
            vector<uint32_t> a(4 + offsets.size() * offsets[0].size(), 0);
            a[0] = 0;
            if (transfer_type == 0 || transfer_type == 2)
            {
                buffer.push_back(a);
            }
            else
            {
                auto &p = dpu_rank_ids[i];
                rank_buffer[p.first].push_back(a);
            }
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

                    auto it = redundant_embs.find(index);
                    if (it != redundant_embs.end())
                    {
                        int p = random_num_generation() % (it->second).dpu_ids.size();
                        dpu_id = (it->second).dpu_ids[p];
                        dpu_emb_id = (it->second).dpu_emb_ids[p];
                    }

                    emb_freqs[index] += freq_base;
                    dpu_freqs[dpu_id] += freq_base;
                    ig.dpu_ids.push_back(dpu_id);

                    if (transfer_type == 0 || transfer_type == 2)
                    {
                        if (buffer[dpu_id][4 + gid] == 0)
                            buffer[dpu_id][2]++;
                        buffer[dpu_id].push_back(gid);
                        buffer[dpu_id].push_back(dpu_emb_id);
                        buffer[dpu_id][4 + gid] = 1;
                    }
                    else if (transfer_type == 1)
                    {
                        auto &p = dpu_rank_ids[dpu_id];
                        if (rank_buffer[p.first][p.second][4 + gid] == 0)
                            rank_buffer[p.first][p.second][2]++;
                        rank_buffer[p.first][p.second].push_back(gid);
                        rank_buffer[p.first][p.second].push_back(dpu_emb_id);
                        rank_buffer[p.first][p.second][4 + gid] = 1;
                    }
                }
                index_groups.push_back(ig);
            }
            table_offset += table_sizes[i];
        }
        global_total_index_num += total_index_num;

        auto end_time = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time);
        // printf("Task dispatch time (ms): %.2lf\n", 1.0 * duration.count() / 1000);
        task_dispatch_time.push_back(1.0 * duration.count() / 1000);

        vector<uint32_t> index_group_num;
        vector<uint32_t> index_num;
        for (int i = 0; i < dpus.size(); ++i)
        {
            if (transfer_type == 0 || transfer_type == 2)
            {
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
            else if (transfer_type == 1)
            {
                auto &p = dpu_rank_ids[i];

                index_num.push_back(((uint32_t)rank_buffer[p.first][p.second].size() - 4 - index_groups.size()) / 2);

                uint32_t sum = 0;
                for (int j = 0; j < index_groups.size(); ++j)
                {
                    uint32_t v = rank_buffer[p.first][p.second][4 + j];
                    if (v)
                    {
                        rank_buffer[p.first][p.second][4 + j] = sum;
                        sum += v;
                    }
                }
                index_group_num.push_back(sum);
            }
        }
        index_group_nums.push_back(index_group_num);
        index_nums.push_back(index_num);

        vector<vector<float>> ret_buffer(dpus.size());
        vector<vector<vector<float>>> rank_ret_buffer(ranks.size());
        vector<vector<vector<float>>> dpu_ret_buffer(dpus.size());
        if (transfer_type == 0)
        {
            uint32_t max_size = 0, max_ret_size = 0;
            for (int i = 0; i < dpus.size(); ++i)
            {
                buffer[i][1] = index_num[i];
                max_size = max(max_size, (uint32_t)buffer[i].size());
                max_ret_size = max(max_ret_size, index_group_num[i]);
            }
            for (int i = 0; i < dpus.size(); ++i)
            {
                buffer[i].resize(max_size);
                ret_buffer[i].resize(max_ret_size * emb_dim);
            }
        }
        else if (transfer_type == 1)
        {
            for (int i = 0, l = 0, r; i < ranks.size(); ++i, l = r)
            {
                r = l + ranks[i]->dpus().size();
                uint32_t max_size = 0, max_ret_size = 0;
                for (int j = l, k = 0; j < r; ++j, ++k)
                {
                    rank_buffer[i][k][1] = ((uint32_t)rank_buffer[i][k].size() - 4 - index_groups.size()) / 2;
                    max_size = max(max_size, (uint32_t)rank_buffer[i][k].size());
                    max_ret_size = max(max_ret_size, index_group_num[j]);
                }
                for (int j = l, k = 0; j < r; ++j, ++k)
                {
                    rank_buffer[i][k].resize(max_size);
                    rank_ret_buffer[i].push_back(vector<float>(max_ret_size * emb_dim));
                }
            }
        }
        else if (transfer_type == 2)
        {
            for (int i = 0; i < dpus.size(); ++i)
            {
                buffer[i][1] = index_num[i];
                dpu_ret_buffer[i].push_back(vector<float>(index_group_num[i] * emb_dim));
            }
        }

        if (exec_type == 0)
        {
            start_time = chrono::high_resolution_clock::now();

            if (transfer_type == 0)
            {
                dpuset.copy("buffer", buffer);
            }
            else if (transfer_type == 1)
            {
                vector<dpu::DpuSetAsync> async_ranks;
                for (int i = 0, l = 0, r; i < ranks.size(); ++i, l = r)
                {
                    async_ranks.push_back(ranks[i]->async());
                    async_ranks.back().copy("buffer", rank_buffer[i]);
                }
                for (int i = 0; i < ranks.size(); ++i)
                {
                    async_ranks[i].sync();
                }
            }
            else if (transfer_type == 2)
            {
                vector<dpu::DpuSetAsync> async_dpus;
                for (int i = 0; i < dpus.size(); ++i)
                {
                    async_dpus.push_back(dpus[i]->async());
                    async_dpus.back().copy("buffer", buffer[i]);
                }
                for (int i = 0; i < dpus.size(); ++i)
                {
                    async_dpus[i].sync();
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

            if (transfer_type == 0)
            {
                dpuset.copy(ret_buffer, "buffer", 1 * 1024 * 1024);
            }
            else if (transfer_type == 1)
            {
                vector<dpu::DpuSetAsync> async_ranks;
                for (int i = 0, l = 0, r; i < ranks.size(); ++i, l = r)
                {
                    async_ranks.push_back(ranks[i]->async());
                    async_ranks.back().copy(rank_ret_buffer[i], "buffer", 1 * 1024 * 1024);
                }
                for (int i = 0; i < ranks.size(); ++i)
                {
                    async_ranks[i].sync();
                }
            }
            else if (transfer_type == 2)
            {
                vector<dpu::DpuSetAsync> async_dpus;
                for (int i = 0; i < dpus.size(); ++i)
                {
                    async_dpus.push_back(dpus[i]->async());
                    async_dpus.back().copy(dpu_ret_buffer[i], "buffer", 1 * 1024 * 1024);
                }
                for (int i = 0; i < dpus.size(); ++i)
                {
                    async_dpus[i].sync();
                }
            }

            end_time = chrono::high_resolution_clock::now();
            duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time);
            // printf("Result transfer time (ms): %.2lf\n", 1.0 * duration.count() / 1000);
            result_transfer_time.push_back(1.0 * duration.count() / 1000);
        }
        else if (exec_type == 1)
        {
            start_time = chrono::high_resolution_clock::now();

            vector<dpu::DpuSetAsync> async_ranks;
            for (int i = 0, l = 0, r; i < ranks.size(); ++i, l = r)
            {
                async_ranks.push_back(ranks[i]->async());
                async_ranks.back().copy("buffer", rank_buffer[i]);
                async_ranks.back().exec();
                async_ranks.back().copy(rank_ret_buffer[i], "buffer", 1 * 1024 * 1024);
            }
            for (int i = 0; i < ranks.size(); ++i)
            {
                async_ranks[i].sync();
            }

            end_time = chrono::high_resolution_clock::now();
            duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time);
            // printf("PIM cal. time (ms): %.2lf\n", 1.0 * duration.count() / 1000);
            PIM_cal_time.push_back(1.0 * duration.count() / 1000);
        }
        else if (exec_type == 2)
        {
            start_time = chrono::high_resolution_clock::now();

            vector<dpu::DpuSetAsync> async_dpus;
            for (int i = 0; i < dpus.size(); ++i)
            {
                async_dpus.push_back(dpus[i]->async());
                async_dpus.back().copy("buffer", buffer[i]);
                async_dpus.back().exec();
                async_dpus.back().copy(dpu_ret_buffer[i], "buffer", 1 * 1024 * 1024);
            }
            for (int i = 0; i < dpus.size(); ++i)
            {
                async_dpus[i].sync();
            }

            end_time = chrono::high_resolution_clock::now();
            duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time);
            // printf("PIM cal. time (ms): %.2lf\n", 1.0 * duration.count() / 1000);
            PIM_cal_time.push_back(1.0 * duration.count() / 1000);
        }

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
                    if (transfer_type == 0)
                    {
                        uint32_t result_index = buffer[id][4 + i];
                        auto &v = ret_buffer[id];
                        for (int k = 0; k < emb_dim; ++k)
                            sum[k] += v[result_index * emb_dim + k];
                    }
                    else if (transfer_type == 1)
                    {
                        auto &p = dpu_rank_ids[id];
                        uint32_t result_index = rank_buffer[p.first][p.second][4 + i];
                        auto &v = rank_ret_buffer[p.first][p.second];
                        for (int k = 0; k < emb_dim; ++k)
                            sum[k] += v[result_index * emb_dim + k];
                    }
                    else if (transfer_type == 2)
                    {
                        uint32_t result_index = buffer[id][4 + i];
                        auto &v = dpu_ret_buffer[id].back();
                        for (int k = 0; k < emb_dim; ++k)
                            sum[k] += v[result_index * emb_dim + k];
                    }
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

        if (split_emb_num == 0)
            split_emb_num = total_index_num * SPLIT_EMB_NUM_RATIO;
        split_emb();

        if (merge_emb_num == 0)
            merge_emb_num = total_index_num * MERGE_EMB_NUM_RATIO;
        merge_emb();

        auto end2end_end_time = chrono::high_resolution_clock::now();
        auto end2end_duration = chrono::duration_cast<chrono::microseconds>(end2end_end_time - end2end_start_time);
        // printf("Total apply emb time (ms): %.2lf\n", 1.0 * end2end_duration.count() / 1000);
        total_apply_emb_time.push_back(1.0 * end2end_duration.count() / 1000);

        float time_ratio = split_emb_time.back() / total_apply_emb_time.back() * 100;
        if (split_emb_num > 0 && time_ratio > 5)
        {
            split_emb_num--;
        }
        else if (time_ratio < 3 && cur_redundant_emb_size < max_redundant_emb_size * 0.95)
        {
            split_emb_num++;
        }

        time_ratio = merge_emb_time.back() / total_apply_emb_time.back() * 100;
        if (merge_emb_num > 0 && time_ratio > 1)
        {
            merge_emb_num--;
        }
        else if (time_ratio < 0.5 && cur_redundant_emb_size > max_redundant_emb_size * 0.95)
        {
            merge_emb_num++;
        }

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
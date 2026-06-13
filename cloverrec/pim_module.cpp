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

template <typename T>
std::vector<T> numpy_to_vector_1D(py::array_t<T> array)
{
    py::buffer_info buf = array.request();

    T *ptr = static_cast<T *>(buf.ptr);
    std::vector<T> vec(ptr, ptr + buf.size);

    return vec;
}

template <typename T>
std::vector<std::vector<T>> numpy_to_vector_2D(py::array_t<T> array)
{
    py::buffer_info buf = array.request();

    auto rows = buf.shape[0];
    auto cols = buf.shape[1];

    T *ptr = static_cast<T *>(buf.ptr);
    std::vector<std::vector<T>> result;
    result.reserve(rows);

    for (size_t i = 0; i < rows; i++)
    {
        std::vector<T> row(ptr + i * cols, ptr + (i + 1) * cols);
        result.push_back(row);
    }

    return result;
}

template <typename T>
py::array_t<T> vector_to_numpy_1D(const std::vector<T> &vec)
{
    py::array_t<T> arr(vec.size());
    if (!vec.empty())
    {
        std::memcpy(arr.mutable_data(), vec.data(), vec.size() * sizeof(T));
    }
    return arr;
}

template <typename T>
py::array_t<T> vector_to_numpy_2D(const vector<vector<T>> &vec)
{
    size_t dim1 = vec.size();
    size_t dim2 = vec.empty() ? 0 : vec[0].size();

    vector<T> flattened;
    flattened.reserve(dim1 * dim2);
    for (const auto &row : vec)
    {
        flattened.insert(flattened.end(), row.begin(), row.end());
    }

    py::array_t<T> arr({dim1, dim2});
    if (!flattened.empty())
    {
        std::memcpy(arr.mutable_data(), flattened.data(), flattened.size() * sizeof(T));
    }
    return arr;
}

template <typename T>
py::array_t<T> vector_to_numpy_3D(const vector<vector<vector<T>>> &vec)
{
    size_t dim1 = vec.size();
    size_t dim2 = vec.empty() ? 0 : vec[0].size();
    size_t dim3 = (dim2 == 0 || vec[0].empty()) ? 0 : vec[0][0].size();

    vector<T> flattened;
    flattened.reserve(dim1 * dim2 * dim3);
    for (const auto &mat : vec)
    {
        for (const auto &row : mat)
        {
            flattened.insert(flattened.end(), row.begin(), row.end());
        }
    }

    py::array_t<T> arr({dim1, dim2, dim3});
    if (!flattened.empty())
    {
        std::memcpy(arr.mutable_data(), flattened.data(), flattened.size() * sizeof(T));
    }
    return arr;
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

private:
    int exec_type;     // 0: sync 1: rank async 2: dpu async
    int transfer_type; // 0: all 1: rank 2: dpu

    float redundant_ratio;
    float split_emb_num_ratio;
    float merge_emb_num_ratio;
    uint32_t select_emb_iter;
    uint32_t aging_freq;
    uint32_t delta_freq;

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
    vector<float> task_tranfer_time;
    vector<float> PIM_cal_time;
    vector<float> result_transfer_time;
    vector<float> CPU_cal_time;
    vector<float> split_emb_time;
    vector<float> merge_emb_time;
    vector<float> total_apply_emb_time;
    vector<uint32_t> redundant_emb_size;

    vector<vector<uint32_t>> index_nums;
    vector<vector<uint32_t>> index_group_nums;

public:
    PIMEmbStorage()
    {
        redundant_ratio = 0.005;
        split_emb_num_ratio = 0.0005;
        merge_emb_num_ratio = 0.0001;
        select_emb_iter = 300;
        aging_freq = 100;
        delta_freq = 100;
    }

    void configure(float redundant_ratio_, float split_emb_num_ratio_, float merge_emb_num_ratio_, uint32_t select_emb_iter_, uint32_t aging_freq_, uint32_t delta_freq_)
    {
        redundant_ratio = redundant_ratio_;
        split_emb_num_ratio = split_emb_num_ratio_;
        merge_emb_num_ratio = merge_emb_num_ratio_;
        select_emb_iter = select_emb_iter_;
        aging_freq = aging_freq_;
        delta_freq = delta_freq_;
    }

    void profiling()
    {
        auto avg_task_tranfer_time = cal_vec_avg_float(task_tranfer_time);
        auto avg_PIM_cal_time = cal_vec_avg_float(PIM_cal_time);
        auto avg_result_transfer_time = cal_vec_avg_float(result_transfer_time);
        auto avg_CPU_cal_time = cal_vec_avg_float(CPU_cal_time);
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

        printf("---------------------------------------------------------------------\n");

        printf("avg_task_tranfer_time: %.2lf\n", avg_task_tranfer_time);
        printf("avg_PIM_cal_time: %.2lf\n", avg_PIM_cal_time);
        printf("avg_result_transfer_time: %.2lf\n", avg_result_transfer_time);
        printf("avg_CPU_cal_time: %.2lf\n", avg_CPU_cal_time);
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
        printf("cur_redundant_emb_size: %u\n", cur_redundant_emb_size);
        printf("max_redundant_emb_size: %u\n", max_redundant_emb_size);
        printf("split_emb_num: %u\n", split_emb_num);
        printf("merge_emb_num: %u\n", merge_emb_num);
        printf("redundant_emb_num: %u\n", (uint32_t)redundant_emb_ids.size());

        printf("---------------------------------------------------------------------\n");
    }

    void clear_profiling_data()
    {
        auto &dpus = dpuset.dpus();

        vector<float>().swap(task_tranfer_time);
        vector<float>().swap(PIM_cal_time);
        vector<float>().swap(result_transfer_time);
        vector<float>().swap(CPU_cal_time);
        vector<float>().swap(split_emb_time);
        vector<float>().swap(total_apply_emb_time);

        vector<vector<uint32_t>>().swap(index_nums);
        vector<vector<uint32_t>>().swap(index_group_nums);
        vector<uint32_t>().swap(redundant_emb_size);

        global_total_index_num = 0;

        apply_emb_num = 0;
        freq_base = 100;
        redundant_embs.clear();
        vector<uint32_t>().swap(redundant_emb_ids);
        split_emb_num = 0;
        merge_emb_num = 0;
        cur_redundant_emb_size = 0;
        max_redundant_emb_size = 1.0 * total_emb_num * redundant_ratio;
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
        transfer_type = 1;

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

        max_redundant_emb_size = 1.0 * total_emb_num * redundant_ratio;
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
        for (int i = 0; i < min(dpu_emb_nums[dpu_id], select_emb_iter); ++i)
        {
            uint32_t p;

            if (dpu_emb_nums[dpu_id] <= select_emb_iter)
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

            uint32_t hot_dpu_id = select_hot_dpu(dpu_ids);
            if (hot_dpu_id == dpus.size())
                continue;

            uint32_t emb_id = select_hot_emb(hot_dpu_id);
            if (emb_id == total_emb_num)
                continue;

            uint32_t emb_table_id = cal_table_id(emb_id);
            auto it = redundant_embs.find(emb_id);
            uint32_t delta_freq = cal_emb_delta_freq(emb_id);

            uint32_t cold_dpu_id = select_cold_dpu(emb_id, emb_table_id, dpu_freqs[hot_dpu_id], it == redundant_embs.end() ? (emb_freqs[emb_id] / 2) : (emb_freqs[emb_id] / ((it->second).dpu_ids.size() + 1)));
            if (cold_dpu_id == dpus.size())
                continue;

            vector<float> buffer(emb_dim);
            for (int i = 0; i < emb_dim; ++i)
                buffer[i] = tables[emb_id * emb_dim + i];

            dpuset.dpus()[cold_dpu_id]->copy("buffer", 2 * 1024 * 1024 + dpu_emb_nums[cold_dpu_id] * emb_dim * 4, buffer);

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

            for (auto &id : redundant_emb.dpu_ids)
            {
                if (id == cold_dpu_id)
                    continue;
                dpu_freqs[id] -= delta_freq;
            }
            dpu_freqs[cold_dpu_id] += emb_freqs[emb_id] / redundant_emb.dpu_ids.size();
        }

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

        for (int i = 0; i < min((uint32_t)redundant_emb_ids.size(), select_emb_iter); ++i)
        {
            int p;
            if (redundant_emb_ids.size() <= select_emb_iter)
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

    std::tuple<py::array_t<int64_t>, py::array_t<float>, py::array_t<int64_t>, py::array_t<float>> apply_emb(py::array_t<uint64_t> &lS_o, py::array_t<uint64_t> &lS_i, double host_CPU_cal_time)
    {
        static std::mt19937 random_num_generation(std::random_device{}());

        auto &dpus = dpuset.dpus();
        auto &ranks = dpuset.ranks();

        apply_emb_num++;
        if (aging_freq > 0 && apply_emb_num % aging_freq == 0)
            freq_base++; // aging

        auto offsets = numpy_to_vector_2D(lS_o);
        auto indices = numpy_to_vector_2D(lS_i);

        auto end2end_start_time = chrono::high_resolution_clock::now();

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

        vector<uint64_t> all_indices;

        for (int i = 0, table_offset = 0; i < offsets.size(); ++i)
        {
            auto &sparse_offset_group_batch = offsets[i];
            auto &sparse_index_group_batch = indices[i];

            batch_size = sparse_offset_group_batch.size() - 1;

            for (int j = 0; j < batch_size; ++j)
            {
                IndexGroup ig;
                auto start = sparse_offset_group_batch[j];
                auto end = sparse_offset_group_batch[j + 1];

                total_index_num += end - start;

                int gid = index_groups.size();
                for (int k = start; k < end; ++k)
                {
                    auto index = sparse_index_group_batch[k];
                    index += table_offset;

                    all_indices.push_back(index);

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
            auto start_time = chrono::high_resolution_clock::now();

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

            auto end_time = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time);
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
            auto start_time = chrono::high_resolution_clock::now();

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

            auto end_time = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time);
            // printf("PIM cal. time (ms): %.2lf\n", 1.0 * duration.count() / 1000);
            PIM_cal_time.push_back(1.0 * duration.count() / 1000);
        }
        else if (exec_type == 2)
        {
            auto start_time = chrono::high_resolution_clock::now();

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

            auto end_time = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time);
            // printf("PIM cal. time (ms): %.2lf\n", 1.0 * duration.count() / 1000);
            PIM_cal_time.push_back(1.0 * duration.count() / 1000);
        }

        auto start_time = chrono::high_resolution_clock::now();

        vector<vector<uint64_t>> new_offsets;

        vector<vector<float>> result;

        for (int i = 0; i < index_groups.size();)
        {
            vector<uint64_t> new_offset;
            for (int j = 0; j < batch_size; ++j, ++i)
            {
                new_offset.push_back(result.size());
                
                auto &ig = index_groups[i];
                sort(ig.dpu_ids.begin(), ig.dpu_ids.end());
                auto last = std::unique(ig.dpu_ids.begin(), ig.dpu_ids.end());
                ig.dpu_ids.erase(last, ig.dpu_ids.end());

                for (auto &id : ig.dpu_ids)
                {
                    if (transfer_type == 0)
                    {
                        uint32_t result_index = buffer[id][4 + i];
                        auto &v = ret_buffer[id];
                        result.push_back(vector<float>(v.begin() + result_index * emb_dim, v.begin() + result_index * emb_dim + emb_dim));
                    }
                    else if (transfer_type == 1)
                    {
                        auto &p = dpu_rank_ids[id];
                        uint32_t result_index = rank_buffer[p.first][p.second][4 + i];
                        auto &v = rank_ret_buffer[p.first][p.second];
                        result.push_back(vector<float>(v.begin() + result_index * emb_dim, v.begin() + result_index * emb_dim + emb_dim));
                    }
                    else if (transfer_type == 2)
                    {
                        uint32_t result_index = buffer[id][4 + i];
                        auto &v = dpu_ret_buffer[id].back();
                        result.push_back(vector<float>(v.begin() + result_index * emb_dim, v.begin() + result_index * emb_dim + emb_dim));
                    }
                }
            }
            new_offset.push_back(result.size());
            new_offsets.push_back(new_offset);
        }

        auto end_time = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time);
        // printf("CPU cal. time (ms): %.2lf\n", 1.0 * duration.count() / 1000 + host_CPU_cal_time);
        CPU_cal_time.push_back(1.0 * duration.count() / 1000 + host_CPU_cal_time);

        if (split_emb_num == 0)
            split_emb_num = total_index_num * split_emb_num_ratio;
        split_emb();

        if (merge_emb_num == 0)
            merge_emb_num = total_index_num * merge_emb_num_ratio;
        merge_emb();

        auto end2end_end_time = chrono::high_resolution_clock::now();
        auto end2end_duration = chrono::duration_cast<chrono::microseconds>(end2end_end_time - end2end_start_time);
        // printf("Total apply emb time (ms): %.2lf\n", 1.0 * end2end_duration.count() / 1000);
        total_apply_emb_time.push_back(1.0 * end2end_duration.count() / 1000);

        float time_ratio = split_emb_time.back() / total_apply_emb_time.back() * 100;
        if (split_emb_num > 0 && time_ratio > 3)
        {
            split_emb_num--;
        }
        else if (time_ratio < 1 && cur_redundant_emb_size < max_redundant_emb_size * 0.95)
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

        if (delta_freq > 0 && apply_emb_num > delta_freq * 2 && apply_emb_num % delta_freq == 0)
        {
            vector<float> a(PIM_cal_time.begin() + apply_emb_num - delta_freq * 2, PIM_cal_time.begin() + apply_emb_num - delta_freq);
            vector<float> b(PIM_cal_time.begin() + apply_emb_num - delta_freq, PIM_cal_time.begin() + apply_emb_num);
            float delta_PIM_cal_time = cal_vec_avg_float(a) - cal_vec_avg_float(b);
            a = vector<float>(CPU_cal_time.begin() + apply_emb_num - delta_freq * 2, CPU_cal_time.begin() + apply_emb_num - delta_freq);
            b = vector<float>(CPU_cal_time.begin() + apply_emb_num - delta_freq, CPU_cal_time.begin() + apply_emb_num);
            float delta_CPU_cal_time = cal_vec_avg_float(b) - cal_vec_avg_float(a);
            if (delta_PIM_cal_time < 0 || delta_PIM_cal_time < delta_CPU_cal_time)
            {
                max_redundant_emb_size = cur_redundant_emb_size;
            }
            else if (delta_PIM_cal_time > 0 && delta_PIM_cal_time > delta_CPU_cal_time)
            {
                max_redundant_emb_size += split_emb_num;
            }
        }

        redundant_emb_size.push_back(cur_redundant_emb_size);

        vector<uint64_t> to_cache_keys;
        vector<vector<float>> to_cache_values;

        if (all_indices.size() > 0)
        {
            int to_cache_num = result.size() * 0.01;
            to_cache_num = max(1, to_cache_num);

            for (int i = 0; i < to_cache_num; ++i)
            {
                int p = random_num_generation() % all_indices.size();
                int index = all_indices[p];
                to_cache_keys.push_back(index);
            }

            sort(to_cache_keys.begin(), to_cache_keys.end());
            auto last = unique(to_cache_keys.begin(), to_cache_keys.end());
            to_cache_keys.erase(last, to_cache_keys.end());

            for (auto index: to_cache_keys)
            {
                auto dpu_id = index / emb_num_per_dpu;
                auto dpu_emb_id = index % emb_num_per_dpu;

                vector<vector<float>> a;
                a.push_back(vector<float>(emb_dim, 0));

                // to be optimized
                dpus[dpu_id]->copy(a, "buffer", 2 * 1024 * 1024 + dpu_emb_id * emb_dim * 4);
                
                to_cache_values.push_back(a.back());
            }
        }

        return std::make_tuple(vector_to_numpy_2D(new_offsets), vector_to_numpy_2D(result), vector_to_numpy_1D(to_cache_keys), vector_to_numpy_2D(to_cache_values));
    }
};

PYBIND11_MODULE(pim_module, m)
{
    m.doc() = "PIM Module";

    py::class_<PIMEmbStorage>(m, "PIMEmbStorage")
        .def(py::init<>())
        .def("configure", &PIMEmbStorage::configure, "Configure adaptive CloverRec parameters")
        .def("profiling", &PIMEmbStorage::profiling, "Profiling")
        .def("clear_profiling_data", &PIMEmbStorage::clear_profiling_data, "Clear Profiling Data")
        .def("initialize", &PIMEmbStorage::initialize, "Initialize PIMEmbStorage")
        .def("init_pim", &PIMEmbStorage::init_pim, "Loading Embs to PIM")
        .def("apply_emb", &PIMEmbStorage::apply_emb, "Apply Embedding");
}

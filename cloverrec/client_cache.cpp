#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <cstddef>
#include <bits/stdc++.h>
#include <chrono>

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
    return py::array_t<T>(
        vec.size(),
        vec.data());
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

    return py::array_t<T>(
        {dim1, dim2},      // Shape (Three dimensions)
        {dim2 * sizeof(T), // Strides for each dimension (row-major)
         sizeof(T)},
        flattened.data() // Pointer to the flat data
    );
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

    return py::array_t<T>(
        {dim1, dim2, dim3},       // Shape (Three dimensions)
        {dim2 * dim3 * sizeof(T), // Strides for each dimension (row-major)
         dim3 * sizeof(T),
         sizeof(T)},
        flattened.data() // Pointer to the flat data
    );
}

template <typename Key, typename ValueType>
class LRUCache
{
public:
    explicit LRUCache(size_t capacity) : capacity_(capacity) {}

    bool contains(const Key &key)
    {
        auto it = cache_map_.find(key);
        return it != cache_map_.end();
    }

    std::vector<ValueType> get(const Key &key)
    {
        auto it = cache_map_.find(key);
        if (it == cache_map_.end())
        {
            return {};
        }

        cache_list_.splice(cache_list_.begin(), cache_list_, it->second);
        return it->second->second;
    }

    void put(const Key &key, const std::vector<ValueType> &value)
    {
        if (!capacity_) return ;

        auto it = cache_map_.find(key);
        if (it != cache_map_.end())
        {
            it->second->second = value;
            cache_list_.splice(cache_list_.begin(), cache_list_, it->second);
            return;
        }

        if (cache_map_.size() >= capacity_)
        {
            auto last = cache_list_.end();
            last--;
            cache_map_.erase(last->first);
            cache_list_.pop_back();
        }

        cache_list_.emplace_front(key, value);
        cache_map_[key] = cache_list_.begin();
    }

    int size()
    {
        return cache_map_.size();
    }

private:
    size_t capacity_;
    std::list<std::pair<Key, std::vector<ValueType>>> cache_list_;
    std::unordered_map<Key, typename std::list<std::pair<Key, std::vector<ValueType>>>::iterator> cache_map_;
};

class ClientCache
{
private:
    LRUCache<uint64_t, float> *cache;

    // emb data
    uint64_t emb_dim;
    vector<uint64_t> table_sizes;

public:
    void initialize(uint64_t m, py::array_t<uint64_t> &ln)
    {
        emb_dim = m;
        table_sizes = numpy_to_vector_1D(ln);

        uint64_t capacity = 0;
        for (int i = 0; i < table_sizes.size(); ++i)
            capacity += table_sizes[i];
        capacity *= 0.001;

        printf("client cache cacacity: %lu\n", capacity);

        cache = new LRUCache<uint64_t, float>(capacity);
    }

    std::tuple<py::array_t<float>, py::array_t<int64_t>, py::array_t<int64_t>> apply_emb(py::array_t<uint64_t> &lS_o, py::array_t<uint64_t> &lS_i)
    {
        auto start_convertion_time = chrono::high_resolution_clock::now();

        auto offsets = numpy_to_vector_2D(lS_o);
        auto indices = numpy_to_vector_2D(lS_i);

        auto end_convertion_time = chrono::high_resolution_clock::now();
        auto convertion_duration = chrono::duration_cast<chrono::microseconds>(end_convertion_time - start_convertion_time);

        uint32_t batch_size = 0;

        vector<vector<size_t>> new_offsets;
        vector<vector<size_t>> new_indices;
        vector<vector<vector<float>>> result;

        int hit_cnt = 0;
        int miss_cnt = 0;

        for (int i = 0, table_offset = 0; i < offsets.size(); ++i)
        {
            auto &sparse_offset_group_batch = offsets[i];
            auto &sparse_index_group_batch = indices[i];
            vector<size_t> new_sparse_offset_group_batch;
            vector<size_t> new_sparse_index_group_batch;

            batch_size = sparse_offset_group_batch.size();

            vector<vector<float>> evs;

            for (int j = 0; j < batch_size; ++j)
            {
                vector<float> sum(emb_dim, 0.0);

                auto start = sparse_offset_group_batch[j];
                auto end = j + 1 < batch_size ? sparse_offset_group_batch[j + 1] : sparse_index_group_batch.size();

                new_sparse_offset_group_batch.push_back(new_sparse_index_group_batch.size());

                for (int k = start; k < end; ++k)
                {
                    auto index = sparse_index_group_batch[k];
                    uint64_t key = index + table_offset;
                    if (cache->contains(key))
                    {
                        hit_cnt++;
                        auto v = cache->get(key);
                        for (int l = 0; l < emb_dim; ++l)
                            sum[l] += v[l];
                    }
                    else
                    {
                        miss_cnt++;
                        new_sparse_index_group_batch.push_back(index);
                    }
                }
                evs.push_back(sum);
            }
            table_offset += table_sizes[i];
            new_offsets.push_back(new_sparse_offset_group_batch);
            new_indices.push_back(new_sparse_index_group_batch);
            result.push_back(evs);
        }

        printf("hit: %d  miss: %d  hit rate: %.2lf\n", hit_cnt, miss_cnt, 1.0 * hit_cnt / (hit_cnt + miss_cnt));

        int max_len = 0;
        for (int i = 0; i < offsets.size(); ++i)
        {
            max_len = max(max_len, (int)new_indices[i].size());
        }
        for (int i = 0; i < offsets.size(); ++i)
        {
            int padding_len = max_len - new_indices[i].size();
            new_offsets[i].push_back(new_indices[i].size());
            for (int j = 0; j < padding_len; ++j)
            {
                new_indices[i].push_back(0);
            }
        }

        return std::make_tuple(vector_to_numpy_3D(result), vector_to_numpy_2D(new_offsets), vector_to_numpy_2D(new_indices));
    }

    void update_cache(py::array_t<uint64_t> &indices, py::array_t<float> &embs)
    {
        auto keys = numpy_to_vector_1D(indices);
        auto values = numpy_to_vector_2D(embs);

        for (int i = 0; i < keys.size(); ++i)
        {
            cache->put(keys[i], values[i]);
        }
        
        // printf("cur cache size: %d\n", cache->size());
    }
};

PYBIND11_MODULE(client_cache, m)
{
    m.doc() = "Client Cache";

    py::class_<ClientCache>(m, "ClientCache")
        .def(py::init<>())
        .def("initialize", &ClientCache::initialize, "Initialize Client Cache")
        .def("update_cache", &ClientCache::update_cache, "Update Cache")
        .def("apply_emb", &ClientCache::apply_emb, "Apply Embedding");
}
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <cstddef>
#include <cstring>
#include <bits/stdc++.h>

namespace py = pybind11;

class PIMEmbStorage {
public:
    void initialize(py::bytes input_data) {
        std::string raw_data = input_data;
        data = std::vector<uint8_t>(raw_data.begin(), raw_data.end());
    }

    int query_sum(long long n, py::bytes input_data) const {
        std::string byte_data = static_cast<std::string>(input_data);

        printf("%lld\n",n);
        uint64_t *indices = (uint64_t *)byte_data.data();
        int sum = 0; 
        for (int i=0;i<n;++i) {
            uint64_t index = *(indices + i);
            printf("%lld\n", index);
            sum += data[index];
        }
        return sum;
    }

private:
    std::vector<uint8_t> data; 
};

PYBIND11_MODULE(pim_module, m) {
    m.doc() = "PIM Module";

    py::class_<PIMEmbStorage>(m, "PIMEmbStorage")
        .def(py::init<>())
        .def("initialize", &PIMEmbStorage::initialize, "Initialize PIMEmbStorage")
        .def("query_sum", &PIMEmbStorage::query_sum, "Query the sum of data at the given indices");
}
dpu-upmem-dpurte-clang -o a_dpu a_dpu.c
g++ --std=c++11 -o a a.cpp -I/usr/include/dpu -ldpu
./a
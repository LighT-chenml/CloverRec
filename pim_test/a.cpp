#include <bits/stdc++.h>
#include <dpu>

using namespace dpu;

#define DPU_BINARY "a_dpu"

class DPUManager
{
    std::vector<std::vector<float>> buffer;
    DpuSet dpuset = DpuSet::allocate();
public:
    DPUManager()
    {
        printf("num dpu: %ld\n",dpuset.dpus().size());
        dpuset.load(DPU_BINARY);
    }
    void init_dpu()
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
    void run()
    {
        dpuset.exec();

        printf("finish run!\n");
    }
    void output()
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
};

DPUManager dpu_manager;

int main()
{
    dpu_manager.init_dpu();
    dpu_manager.run();
    dpu_manager.output();
    return 0;
}
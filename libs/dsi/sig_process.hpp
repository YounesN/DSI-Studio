#ifndef SIG_PROCESS_HPP
#define SIG_PROCESS_HPP
#include <cmath>
#include "basic_voxel.hpp"
#include "image/image.hpp"

class ProcessSignal : public BaseProcess
{
public:
    virtual void init(Voxel& voxel)
    {
    }
public:
    virtual void run(Voxel& voxel, VoxelData& data)
    {
        std::vector<float> signal(data.space.size());
        if (data.space.front() != 0.0)
        {
            for (unsigned int i = 1; i < data.space.size(); ++i)
                signal[i-1] = data.space[i]/data.space[0];
        }
        int tmp = data.voxel_index;
        voxel.signalData[tmp].resize(signal.size());
        std::copy(signal.begin(), signal.end(), voxel.signalData[data.voxel_index].begin());
    }
    virtual void end(Voxel& voxel,gz_mat_write& mat_writer)
    {
    }
};

#endif//_PROCESS_HPP

#ifndef ICABSM_PROCESS_GPU_HPP
#define ICABSM_PROCESS_GPU_HPP
#include <cmath>
#include "basic_voxel.hpp"
#include "image_model.hpp"
#include "image/image.hpp"
#include "itpp/itsignal.h"
#include "levmar/levmar.h"
#include "itbase.h"
#include <qdebug>
#include <iostream>
#include "armadillo"

#include "cuda_runtime.h"
#include "cuda.h"

using namespace std;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

extern "C"
void LaunchICABSMKernel(bool threeDim,
                        int b_count,
                        int width,
                        int height,
                        int depth,
                        float *dev_md,
                        float *dev_d0,
                        float *dev_d1,
                        float *dev_num_fibers,
                        float *dev_fr,
                        float *dev_fib_fa,
                        float *dev_fib_dir,
                        float *dev_g_dg,
                        float *dev_invg_dg,
                        float *dev_signalData,
                        char *dev_mask);

class ICABSMGPU : public BaseProcess
{

public:
    virtual void init(Voxel& voxel)
    {
        unsigned int b_count = voxel.bvalues.size() - 1;
        gpuErrchk( cudaMalloc((void **) &voxel.gpu.dev_md, voxel.dim.size() * sizeof(float)));
        gpuErrchk( cudaMalloc((void **) &voxel.gpu.dev_d0, voxel.dim.size() * sizeof(float)));
        gpuErrchk( cudaMalloc((void **) &voxel.gpu.dev_d1, voxel.dim.size() * sizeof(float)));
        gpuErrchk( cudaMalloc((void **) &voxel.gpu.dev_num_fibers, voxel.dim.size() * sizeof(float)));
        gpuErrchk( cudaMalloc((void **) &voxel.gpu.dev_fr, voxel.numberOfFibers * voxel.dim.size() * sizeof(float)));
        gpuErrchk( cudaMalloc((void **) &voxel.gpu.dev_fib_fa, voxel.dim.size() * sizeof(float)));
        gpuErrchk( cudaMalloc((void **) &voxel.gpu.dev_fib_dir, voxel.numberOfFibers * voxel.dim.size() * 3 * sizeof(float)));
        gpuErrchk( cudaMalloc((void **) &voxel.gpu.dev_g_dg, b_count * 6 * sizeof(float)));
        gpuErrchk( cudaMalloc((void **) &voxel.gpu.dev_invg_dg, b_count * 6 * sizeof(float)));
        gpuErrchk( cudaMalloc((void **) &voxel.gpu.dev_signalData, b_count * voxel.dim.size() * sizeof(float)));
        gpuErrchk( cudaMalloc((void **) &voxel.gpu.dev_matg_dg, b_count * 6 * sizeof(float)));
        gpuErrchk( cudaMalloc((void **) &voxel.gpu.dev_invg_dg, b_count * 6 * sizeof(float)));
        gpuErrchk( cudaMalloc((void **) &voxel.gpu.dev_mask, voxel.dim.size() * sizeof(char)));

        gpuErrchk( cudaMemset(voxel.gpu.dev_d0, 0, voxel.dim.size() * sizeof(float)));
        gpuErrchk( cudaMemset(voxel.gpu.dev_d1, 0, voxel.dim.size() * sizeof(float)));
        gpuErrchk( cudaMemset(voxel.gpu.dev_num_fibers, 0, voxel.dim.size() * sizeof(float)));
        gpuErrchk( cudaMemset(voxel.gpu.dev_fr, 0, voxel.numberOfFibers * voxel.dim.size() * sizeof(float)));
        gpuErrchk( cudaMemset(voxel.gpu.dev_fib_fa, 0, voxel.dim.size() * sizeof(float)));
        gpuErrchk( cudaMemset(voxel.gpu.dev_fib_dir, 0, voxel.numberOfFibers * voxel.dim.size() * 3 * sizeof(float)));
        gpuErrchk( cudaMemset(voxel.gpu.dev_g_dg, 0, b_count * 6 * sizeof(float)));
        gpuErrchk( cudaMemset(voxel.gpu.dev_invg_dg, 0, b_count * 6 * sizeof(float)));
        gpuErrchk( cudaMemset(voxel.gpu.dev_signalData, 0, b_count * voxel.dim.size() * sizeof(float)));
        gpuErrchk( cudaMemset(voxel.gpu.dev_matg_dg, 0, b_count * 6 * sizeof(float)));
        gpuErrchk( cudaMemset(voxel.gpu.dev_invg_dg, 0, b_count * 6 * sizeof(float)));
        gpuErrchk( cudaMemset(voxel.gpu.dev_mask, 0, voxel.dim.size() * sizeof(char)));

        int i,j;
        // copy signal data
        float * tmpsignalData = new float[b_count * voxel.dim.size()];
        memset(tmpsignalData, 0, b_count * voxel.dim.size() * sizeof(float));
        for(i=0; i<voxel.dim.size(); i++)
        {
            if(!voxel.signalData[i].size())
                continue;
            for(j=0; j<b_count; j++)
                tmpsignalData[i*b_count+j] = voxel.signalData[i][j];
        }
        gpuErrchk( cudaMemcpy(voxel.gpu.dev_signalData, tmpsignalData,
                   b_count * voxel.dim.w * voxel.dim.h * voxel.dim.d * sizeof(float),
                   cudaMemcpyHostToDevice));

        char * tmp_mask = new char[voxel.dim.size()];
        for(i=0; i<voxel.dim.size(); i++)
            tmp_mask[i] = voxel.image_model->mask[i];
        gpuErrchk( cudaMemcpy(voxel.gpu.dev_mask, tmp_mask, voxel.dim.size() * sizeof(char),
                   cudaMemcpyHostToDevice));
        // copy g_dg and pseudo inverse of it
        float * tmp_g_dg = new float[b_count * 6];
        float * tmp_invg_dg = new float[b_count * 6];
        voxel.matg_dg.set_size(b_count, 6);
        voxel.invg_dg.clear();
        voxel.invg_dg.resize(6*b_count);
        voxel.g_dg.clear();
        voxel.g_dg.resize(6*b_count);
        for(i=0; i<b_count; i++)
        {
            voxel.matg_dg(i,0) = voxel.g_dg[6*i+0] = voxel.bvectors[i+1][0]*voxel.bvectors[i+1][0];
            voxel.matg_dg(i,1) = voxel.g_dg[6*i+1] = voxel.bvectors[i+1][1]*voxel.bvectors[i+1][1];
            voxel.matg_dg(i,2) = voxel.g_dg[6*i+2] = voxel.bvectors[i+1][2]*voxel.bvectors[i+1][2];
            voxel.matg_dg(i,3) = voxel.g_dg[6*i+3] = voxel.bvectors[i+1][0]*voxel.bvectors[i+1][1]*2;
            voxel.matg_dg(i,4) = voxel.g_dg[6*i+4] = voxel.bvectors[i+1][0]*voxel.bvectors[i+1][2]*2;
            voxel.matg_dg(i,5) = voxel.g_dg[6*i+5] = voxel.bvectors[i+1][1]*voxel.bvectors[i+1][2]*2;
        }
        arma::mat ones;
        ones.ones(1,6);
        arma::mat bvalue(b_count,1);
        for(i=0; i<b_count; i++)
            bvalue(i, 0) = voxel.bvalues[i+1];
        bvalue = bvalue * ones;
        for(i=0;i<b_count;i++)
            for(j=0;j<6;j++)
                voxel.matg_dg(i,j)*=bvalue(i,j);
        voxel.matinvg_dg = arma::pinv(voxel.matg_dg);
        //voxel.matinvg_dg = -voxel.matinvg_dg;
        for(i=0; i<b_count; i++)
        {
            for(j=0; j<6; j++)
            {
                tmp_g_dg[6*i + j] = voxel.matg_dg(i, j);
                tmp_invg_dg[j*b_count+i] = voxel.matinvg_dg(j, i);
            }
        }
        gpuErrchk( cudaMemcpy(voxel.gpu.dev_g_dg, tmp_g_dg, b_count * 6 * sizeof(float),
                   cudaMemcpyHostToDevice));
        gpuErrchk( cudaMemcpy(voxel.gpu.dev_invg_dg, tmp_invg_dg, b_count * 6 * sizeof(float),
                   cudaMemcpyHostToDevice));
        LaunchICABSMKernel(voxel.threeDimensionalWindow,
                           voxel.bvalues.size()-1,
                           voxel.dim.width(),
                           voxel.dim.height(),
                           voxel.dim.depth(),
                           voxel.gpu.dev_md,
                           voxel.gpu.dev_d0,
                           voxel.gpu.dev_d1,
                           voxel.gpu.dev_num_fibers,
                           voxel.gpu.dev_fr,
                           voxel.gpu.dev_fib_fa,
                           voxel.gpu.dev_fib_dir,
                           voxel.gpu.dev_g_dg,
                           voxel.gpu.dev_invg_dg,
                           voxel.gpu.dev_signalData,
                           voxel.gpu.dev_mask);
        delete [] tmp_g_dg;
        delete [] tmp_invg_dg;
        delete [] tmp_mask;
        delete [] tmpsignalData;
    }
public:
    virtual void run(Voxel& voxel, VoxelData& data)
    {
    }
    virtual void end(Voxel& voxel,gz_mat_write& mat_writer)
    {
        unsigned int i=0;
        float *tmp_fib_fa = new float[voxel.dim.size()];
        float *tmp_md = new float[voxel.dim.size()];
        float *tmp_d0 = new float[voxel.dim.size()];
        float *tmp_d1 = new float[voxel.dim.size()];
        float *tmp_fib_dir = new float[voxel.dim.size() * 3];
        gpuErrchk( cudaMemcpy(tmp_fib_fa, voxel.gpu.dev_fib_fa, voxel.dim.size() * sizeof(float), cudaMemcpyDeviceToHost));
        gpuErrchk( cudaMemcpy(tmp_md, voxel.gpu.dev_md, voxel.dim.size() * sizeof(float), cudaMemcpyDeviceToHost));
        gpuErrchk( cudaMemcpy(tmp_d0, voxel.gpu.dev_d0, voxel.dim.size() * sizeof(float), cudaMemcpyDeviceToHost));
        gpuErrchk( cudaMemcpy(tmp_d1, voxel.gpu.dev_d1, voxel.dim.size() * sizeof(float), cudaMemcpyDeviceToHost));
        gpuErrchk( cudaMemcpy(tmp_fib_dir, voxel.gpu.dev_fib_dir, voxel.dim.size() * 3 * sizeof(float), cudaMemcpyDeviceToHost));

        mat_writer.write("fa0", tmp_fib_fa, 1, voxel.dim.size());
        mat_writer.write("dir0", tmp_fib_dir, 1, voxel.dim.size() * 3);
        mat_writer.write("adc", tmp_md, 1, voxel.dim.size());
        mat_writer.write("axial_dif", tmp_d0, 1, voxel.dim.size());
        mat_writer.write("radial_dif", tmp_d1, 1, voxel.dim.size());

        delete [] tmp_fib_fa;
        delete [] tmp_fib_dir;
        delete [] tmp_md;
        delete [] tmp_d0;
        delete [] tmp_d1;
    }
};


#endif//_PROCESS_HPP

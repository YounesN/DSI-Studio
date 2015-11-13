#include <cuda.h>
#include <cuda_runtime.h>

#include "basic_voxel.hpp"

__device__ void gpu_set_mat_row(float *a, float *b, int size)
{
    int i;
    for(i=0; i<size; i++)
        a[i] = b[i];
}

__global__ void icabsm_kernel(bool threeDim,
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
                              float *dev_signalData)
{
    unsigned int i=0, j=0, k=0;
    int m=0, n=0;
    float *mixedSig;
    if(threeDim)
        mixedSig = new float[19 * b_count];
    else
        mixedSig = new float[9 * b_count];
    unsigned int threadID = blockIdx.x *blockDim.x + threadIdx.x;

    /* get 3d position */
    i = threadID;
    k = (unsigned int) i / (height*width);
    i -= k * height * width;
    j = (unsigned int) i / width;
    i -= j * width;

    if(threeDim)
    {
        // 5 voxels behind current voxel
        if(k > 0 && j > 0)
            gpu_set_mat_row(mixedSig, dev_signalData+((threadID-width*height-width) * b_count), b_count);
        if(k > 0 && i > 0)
            gpu_set_mat_row(mixedSig+b_count, dev_signalData+((threadID-width*height-1)*b_count), b_count);
        if(k > 0)
            gpu_set_mat_row(mixedSig+2*b_count, dev_signalData+((threadID-width*height)*b_count), b_count);
        if(k > 0 && i < width - 1)
            gpu_set_mat_row(mixedSig+3*b_count, dev_signalData+((threadID-width*height+1)*b_count), b_count);
        if(k > 0 && j < height - 1)
            gpu_set_mat_row(mixedSig+4*b_count, dev_signalData+((threadID-width*height+width)*b_count), b_count);

        // current flat 9 voxels
        if(i > 0 && j > 0)
            gpu_set_mat_row(mixedSig+5*b_count, dev_signalData+((threadID-width-1)*b_count), b_count);
        if(j > 0)
            gpu_set_mat_row(mixedSig+6*b_count, dev_signalData+((threadID-width)*b_count), b_count);
        if(i < width - 1 && j > 0 )
            gpu_set_mat_row(mixedSig+7*b_count, dev_signalData+((threadID-width)*b_count), b_count);
        if(i > 0)
            gpu_set_mat_row(mixedSig+8*b_count, dev_signalData+((threadID-1)*b_count), b_count);
        gpu_set_mat_row(mixedSig+9*b_count, dev_signalData+(threadID*b_count), b_count);
        if(i < width - 1)
            gpu_set_mat_row(mixedSig+10*b_count, dev_signalData+((threadID+1)*b_count), b_count);
        if(i > 0 && j < height - 1)
            gpu_set_mat_row(mixedSig+11*b_count, dev_signalDat+((threadID+width-1)*b_count), b_count);
        if(j < height - 1 )
            gpu_set_mat_row(mixedSig+12*b_count, dev_signalData+((threadID+width)*b_count), b_count);
        if(i < width - 1 && j < height - 1)
            gpu_set_mat_row(mixedSig+13*b_count, dev_signalData+((threadID+width+1)*b_count), b_count);

        // 5 voxels in front of current voxel
        if(k < depth - 1 && j > 0)
            gpu_set_mat_row(mixedSig+14*b_count, dev_signalData+((threadID+width*height-width)*b_count), b_count);
        if(k < depth - 1 && i > 0)
            gpu_set_mat_row(mixedSig+15*b_count, dev_signalData+((threadID+width*height-1)*b_count), b_count);
        if(k < depth - 1)
            gpu_set_mat_row(mixedSig+16*b_count, dev_signalData+((threadID+width*height)*b_count), b_count);
        if(k < depth - 1 && i < width - 1)
            gpu_set_mat_row(mixedSig+17*b_count, dev_signalData+((threadID+width*height+1)*b_count), b_count);
        if(k < depth - 1 && j < height - 1)
            gpu_set_mat_row(mixedSig+18*b_count, dev_signalData+((threadID+width*height+width)*b_count), b_count);
    }
    else
    {

    }

    delete [] mixedSig;
}

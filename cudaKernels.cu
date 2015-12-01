#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

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
                              float *dev_signalData,
                              char *dev_mask,
                              float *dev_mixedSig);


__device__ void gpu_set_mat_row(float *a, float *b, int size)
{
    int i;
    for(i=0; i<size; i++)
        a[i] = b[i];
}

/*
 * Perform matrix multiplication
 * a: ixj -> input
 * b: jxk -> input
 * c: ixk -> output
 */
__device__ void gpu_matrix_multiplication(float *a, float *b, double *c, int i, int j, int k)
{
    memset(c, 0, i*k*sizeof(float));
    int row, col, inner;
    for(row = 0; row < i; row++)
    {
        for(col = 0; col < k; col++)
        {
            for(inner = 0; inner < j; inner ++)
            {
                c[row * i + col] += a[row * i + inner] * b[inner * j + col];
            }
        }
    }
}

/*
perform x <- c*x+s*y
perform y <- c*y-s*x
*/
__device__ void rotate(double * x_begin, int size,double * y,double c,double s)
{
    double  x_temp;
    int i=0;
    if (size != 0)
    {
        do
        {

            x_temp = x_begin[i]*c + y[i]*s;
            y[i] = y[i]*c - x_begin[i]*s;
            x_begin[i] = x_temp;

            i++;
            if (i >= size)
                break;
        }
        while (1);
    }
}

__device__ float gpu_get_fa(float l1, float l2, float l3)
{
    float ll = (l1+l2+l3)/3.0;
    if(l1 == 0.0)
        return 0.0;
    float ll1 = l1 - ll;
    float ll2 = l2 - ll;
    float ll3 = l3 - ll;
    return fmin(1.0,sqrt(1.5*(ll1*ll1+ll2*ll2+ll3*ll3)/(l1*l1+l2*l2+l3*l3)));
}

__device__ void eigen_decomposition_sym(double * A,
                                    double * V,
                                    double * d,const int row_count, const int col_count)
{
    //printf("hello from eigen_decomposition_sym\n");
    int iter;
    if(col_count == 3)
    {
        {
            double I1 = A[0] + A[4] + A[8];
            double I2 = A[0]*A[4] + A[0]*A[8]+ A[4]*A[8]-A[1]*A[1]-A[2]*A[2]-A[5]*A[5];
            double I3 = A[0]*A[4]*A[8]+2.0*A[1]*A[2]*A[5]-(A[8]*A[1]*A[1]+A[4]*A[2]*A[2]+A[0]*A[5]*A[5]);
            double I1_3 = (I1/3.0);
            double v = I1_3*I1_3-I2/3.0;
            double s = I1_3*I1_3*I1_3-I1*I2/6.0+I3/2.0;
            if(v + 1.0 == 1.0)
            {
                for(iter=0; iter<3; iter++)
                    d[iter]=0.0;
                for(iter=0; iter<9; iter++)
                    V[iter]=0.0;
                V[0] = V[4] = V[8] = 1.0;
                return;
            }
            double sqrt_v = sqrt(v);
            double tmp = s/v/sqrt_v;
            double min_part = tmp < 1.0 ? tmp : 1.0;
            double max_part = min_part < -1.0 ? -1.0 : min_part;
            double angle = acos(max_part)/3.0;
            d[0] = I1_3 + 2.0*sqrt_v*cos(angle);
            d[1] = I1_3 - 2.0*sqrt_v*cos(3.14159265358979323846/3.0+angle);
            d[2] = I1-d[0]-d[1];
        }

        for(int i = 0;i < 3;++i,V+=3)
        {
            double Ai = A[0]-d[i];
            double Bi = A[4]-d[i];
            double Ci = A[8]-d[i];
            double q1 = (A[2]*A[1]-Ai*A[5]);
            double q2 = (A[1]*A[5]-Bi*A[2]);
            double q3 = (A[2]*A[5]-Ci*A[1]);
            V[0] = q2*q3;
            V[1] = q3*q1;
            V[2] = q2*q1;
            double length = sqrt(V[0]*V[0]+V[1]*V[1]+V[2]*V[2]);
            V[0] = V[0]/length;
            V[1] = V[1]/length;
            V[2] = V[2]/length;
        }
        return;
    }
    const unsigned int size = row_count * col_count;
    const unsigned int dim = col_count;
    double *e = new double[dim+1];

    for(iter=0; iter<size; iter++)
        V[iter]=A[iter];
    for(iter=0; iter<dim; iter++)
        d[iter]=0.0;

    //void tridiagonalize(void)
    {
        //  This is derived from the Algol procedures tred2 by
        //  Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
        //  Auto. Comp., Vol.ii-Linear Algebra, and the corresponding
        //  Fortran subroutine in EISPACK.
        // Householder reduction to tridiagonal form.
        {
            double * Vrowi = V+size-dim;//n-1 row
            for (unsigned int i = dim-1;i > 1;--i,Vrowi -= dim)
            {
                double h=0,g,f;
                // Generate Householder vector.u
                // x is the lower i-1 row vector of row i
                // h = |x|^2
                for (unsigned int k = 0; k < i; k++)
                    h += Vrowi[k] * Vrowi[k];
                if (h+double(1.0) == double(1.0))
                    //if(h < la::eps<value_type>::value)
                {
                    e[i] = Vrowi[i-1];
                    continue;
                }

                f = Vrowi[i-1];
                g = std::sqrt(h);			// g = |x|
                if (f >= double(0.0))
                    g = -g;	// choose sign of g accoring to V[i][i-1]
                e[i] = g;
                h -= f * g;	// h = 1/2|u|^2=1/2(|x-|x|e|^2)=1/2(|x|^2-2*|x|*x0+|x|^2)
                //   = |x|^2-|x|*x0;
                Vrowi[i-1] -= g; // Vrowi x becomes u, u = x-|x|e
                f = double(0.0);
                {
                    double * Vrowj = V;// from the first row
                    for (unsigned int j = 0; j < i; ++j, Vrowj += dim)
                    {
                        unsigned int j_1 = j+1;
                        Vrowj[i] = Vrowi[j]/h;
                        g = double(0.0);
                        double * rowj_1 = V;
                        for (unsigned int k = 0;k < j_1;++k,rowj_1+=dim)
                            g += Vrowj[k]*Vrowi[k];
                        if (j_1 < i)
                        {
                            double * Vrowk = rowj_1+j; //row j +1 , col j
                            for (unsigned int k = j_1;k < i;++k,Vrowk += dim)
                                g += (*Vrowk)*Vrowi[k];
                        }
                        e[j] = g/h;
                        f += e[j] * Vrowi[j];
                    }
                }
                d[i] = h;
                {
                    double hh = f / (h + h);
                    double * Vrowj = V;// from the first row
                    for (unsigned int j = 0; j < i; ++j, Vrowj += dim)
                    {
                        f = Vrowi[j];
                        g = e[j]-hh * f;
                        e[j] = g;
                        for (unsigned int k = 0;k < j+1;++k)
                            Vrowj[k] -= (f * e[k] + g * Vrowi[k]);
                    }
                }
            }
        }

        e[0] = double(0.0);
        d[0] = V[0];
        e[1] = V[dim];
        d[1] = double(0.0);
        V[0] = double(1.0);
        // Accumulate transformations.
        // Also change V from column major to row major
        // Elements in V(j<i,k<i) is row major,
        // Elements in V(j>=i,k>=i) is column major,
        {
            double * Vrowi = V;// from the second row
            for (unsigned int i = 1; i < dim; ++i)
            {
                Vrowi += dim;
                if (d[i] != double(0.0))
                {
                    for (double * Vrowj = V;Vrowj != Vrowi;Vrowj += dim)
                    {
                        double g = 0;
                        for(iter = 0; iter < i; iter++)
                            g += Vrowi[iter] * Vrowj[iter];
                        {
                            double * Vrowk = V;
                            for (int k = 0;k < i;++k,Vrowk += dim)
                                Vrowj[k] -= g * Vrowk[i];
                        }
                    }
                }
                d[i] = Vrowi[i];
                Vrowi[i] = double(1.0);
                {
                    double * Vrowk = V;
                    for (int k = 0;k < i;++k,Vrowk += dim)
                        Vrowk[i] = Vrowi[k] = double(0.0);

                }
            }
        }


    }

    // Symmetric tridiagonal QL algorithm.

    //	void diagonalize(void)
    {
        using namespace std; // for hypot

        //  This is derived from the Algol procedures tql2, by
        //  Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
        //  Auto. Comp., Vol.ii-Linear Algebra, and the corresponding
        //  Fortran subroutine in EISPACK.
        ++e;
        double p,r,b,f=0;
        for (int l = 0,iter = 0;l < dim && iter < 30;++iter)
        {
            unsigned int m = l;
            // Find small subdiagonal element
            for (;m < dim-1;++m)
            {
                /*
                tst1 = ((d[m+1] > 0) ?
                std::max(tst1,la::abs(d[m]) + d[m+1]):
                std::max(tst1,la::abs(d[m]) - d[m+1]);
                if(tst1+e[m] == tst1)
                break;*/
                if (d[m]+e[m] == d[m])
                    break;

            }
            // If m == l, d[l] is an eigenvalue, go for next
            if ((int)m == l)
            {
                ++l;
                iter = 0;
                continue;
            }

            // Compute implicit shift
            p = (d[l+1]-d[l])/(e[l]*double(2.0));
            r = std::sqrt(1+p*p);
            p = d[m]-d[l]+e[l]/(p+((p<0)? -r:r));

            double s=1.0,c=1.0,g=0.0;
            int i = m-1;
            double * Vrowi = V+i*dim;
            double * Vrowi_end = Vrowi+dim;
            do
            {
                f = s*e[i];
                b = c*e[i];
                e[i+1] = r = hypot(f,p);
                if (r+f == f && r+p == p)
                {
                    d[i-1] -= g;
                    e[m] = double(0.0);
                    break;
                }
                s = f/r;
                c = p/r;
                p = d[i+1]-g;
                r = (d[i]-p)*s+c*b*double(2.0);
                g = s*r;
                d[i+1] = p+g;
                p = c*r-b;
                // Accumulate transformation.
                rotate(Vrowi,dim,Vrowi_end,c,-s);
                if (--i < l)
                    break;
                Vrowi_end = Vrowi;
                Vrowi -= dim;
            } // i loop
            while (1);

            if (r != double(0) || i < l)
            {
                e[l] = p;
                e[m] = double(0.0);
                d[l] -= g;
            }
        } // l loop


        // Sort eigenvalues and corresponding vectors.
        double * Vrowi = V;
        for (unsigned int i = 0; i < dim-1; ++i,Vrowi += dim)
        {
            double max = d[i];
            unsigned int k = i;
            for(int j=i+1; j<dim; j++)
            {
                if(d[j]>max)
                {
                    max = d[j];
                    k=j;
                }
            }
            if (k != i)
            {
                double tmp = d[k];
                d[k] = d[i];
                d[i] = tmp;
                for(int iter=0; iter<dim; iter++)
                {
                    tmp = Vrowi[iter];
                    Vrowi[iter] = V[k*dim+iter];
                    V[k*dim+iter] = tmp;
                }
            }
        }
    }

    delete [] e;
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
                        char *dev_mask,
                        float *dev_mixedSig)
{
    /*dim3 blockDim(4, 4, 4);
    dim3 gridDim((width + blockDim.z - 1)/ blockDim.z,
                 (height + blockDim.y - 1) / blockDim.y,
                 (depth + blockDim.z - 1) / blockDim.z);*/
    int threadPerBlock = 128;
    int blocksPerGrid = width * height * depth / threadPerBlock;
    icabsm_kernel<<<blocksPerGrid, threadPerBlock>>>(threeDim,
                                                     b_count,
                                                     width,
                                                     height,
                                                     depth,
                                                     dev_md,
                                                     dev_d0,
                                                     dev_d1,
                                                     dev_num_fibers,
                                                     dev_fr,
                                                     dev_fib_fa,
                                                     dev_fib_dir,
                                                     dev_g_dg,
                                                     dev_invg_dg,
                                                     dev_signalData,
                                                     dev_mask,
                                                     dev_mixedSig);
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));
    cudaDeviceSynchronize();
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
                              float *dev_signalData,
                              char *dev_mask,
                              float *dev_mixedSig)
{
    unsigned int i=0, j=0, k=0;
    int m=0, n=0;
    int blockId = blockIdx.x + blockIdx.y * gridDim.x
                  + gridDim.x * gridDim.y * blockIdx.z;
    int threadID = blockId * (blockDim.x * blockDim.y * blockDim.z)
                   + (threadIdx.z * (blockDim.x * blockDim.y))
                   + (threadIdx.y * blockDim.x) + threadIdx.x;

    if(!dev_mask[threadID])
        return;

    int mixedSigSize;
    if(threeDim)
        mixedSigSize = 19;
    else
        mixedSigSize = 9;

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
            gpu_set_mat_row(dev_mixedSig+threadID*b_count*mixedSigSize, dev_signalData+((threadID-width*height-width) * b_count), b_count);
        __syncthreads();
        if(k > 0 && i > 0)
            gpu_set_mat_row(dev_mixedSig+threadID*b_count*mixedSigSize+b_count, dev_signalData+((threadID-width*height-1)*b_count), b_count);
        if(k > 0)
            gpu_set_mat_row(dev_mixedSig+threadID*b_count*mixedSigSize+2*b_count, dev_signalData+((threadID-width*height)*b_count), b_count);
        if(k > 0 && i < width - 1)
            gpu_set_mat_row(dev_mixedSig+threadID*b_count*mixedSigSize+3*b_count, dev_signalData+((threadID-width*height+1)*b_count), b_count);
        if(k > 0 && j < height - 1)
            gpu_set_mat_row(dev_mixedSig+threadID*b_count*mixedSigSize+4*b_count, dev_signalData+((threadID-width*height+width)*b_count), b_count);
        //printf("%f\n", *(mixedSig+4*b_count));

        // current flat 9 voxels
        if(i > 0 && j > 0)
            gpu_set_mat_row(dev_mixedSig+threadID*b_count*mixedSigSize+5*b_count, dev_signalData+((threadID-width-1)*b_count), b_count);
        if(j > 0)
            gpu_set_mat_row(dev_mixedSig+threadID*b_count*mixedSigSize+6*b_count, dev_signalData+((threadID-width)*b_count), b_count);
        if(i < width - 1 && j > 0 )
            gpu_set_mat_row(dev_mixedSig+7*b_count, dev_signalData+((threadID-width+1)*b_count), b_count);
        if(i > 0)
            gpu_set_mat_row(dev_mixedSig+8*b_count, dev_signalData+((threadID-1)*b_count), b_count);
        gpu_set_mat_row(dev_mixedSig+threadID*b_count*mixedSigSize+9*b_count, dev_signalData+(threadID*b_count), b_count);
        if(i < width - 1)
            gpu_set_mat_row(dev_mixedSig+threadID*b_count*mixedSigSize+10*b_count, dev_signalData+((threadID+1)*b_count), b_count);
        if(i > 0 && j < height - 1)
            gpu_set_mat_row(dev_mixedSig+threadID*b_count*mixedSigSize+11*b_count, dev_signalData+((threadID+width-1)*b_count), b_count);
        if(j < height - 1 )
            gpu_set_mat_row(dev_mixedSig+threadID*b_count*mixedSigSize+12*b_count, dev_signalData+((threadID+width)*b_count), b_count);
        if(i < width - 1 && j < height - 1)
            gpu_set_mat_row(dev_mixedSig+threadID*b_count*mixedSigSize+13*b_count, dev_signalData+((threadID+width+1)*b_count), b_count);

        // 5 voxels in front of current voxel
        if(k < depth - 1 && j > 0)
            gpu_set_mat_row(dev_mixedSig+threadID*b_count*mixedSigSize+14*b_count, dev_signalData+((threadID+width*height-width)*b_count), b_count);
        if(k < depth - 1 && i > 0)
            gpu_set_mat_row(dev_mixedSig+threadID*b_count*mixedSigSize+15*b_count, dev_signalData+((threadID+width*height-1)*b_count), b_count);
        if(k < depth - 1)
            gpu_set_mat_row(dev_mixedSig+threadID*b_count*mixedSigSize+16*b_count, dev_signalData+((threadID+width*height)*b_count), b_count);
        if(k < depth - 1 && i < width - 1)
            gpu_set_mat_row(dev_mixedSig+threadID*b_count*mixedSigSize+17*b_count, dev_signalData+((threadID+width*height+1)*b_count), b_count);
        if(k < depth - 1 && j < height - 1)
            gpu_set_mat_row(dev_mixedSig+threadID*b_count*mixedSigSize+18*b_count, dev_signalData+((threadID+width*height+width)*b_count), b_count);
    }
    else
    {
        if(i > 0 && j > 0)
            gpu_set_mat_row(dev_mixedSig+threadID*b_count*mixedSigSize, dev_signalData+((threadID-width-1)*b_count), b_count);
        if(j > 0)
            gpu_set_mat_row(dev_mixedSig+threadID*b_count*mixedSigSize+b_count, dev_signalData+((threadID-width)*b_count), b_count);
        if(i < width - 1 && j > 0)
            gpu_set_mat_row(dev_mixedSig+threadID*b_count*mixedSigSize+2*b_count, dev_signalData+((threadID-width+1)*b_count), b_count);
        if(i > 0)
            gpu_set_mat_row(dev_mixedSig+threadID*b_count*mixedSigSize+3*b_count, dev_signalData+((threadID-1)*b_count), b_count);
        gpu_set_mat_row(dev_mixedSig+threadID*b_count*mixedSigSize+4*b_count, dev_signalData+(threadID*b_count), b_count);
        if(i < width - 1)
            gpu_set_mat_row(dev_mixedSig+threadID*b_count*mixedSigSize+5*b_count, dev_signalData+((threadID+1)*b_count), b_count);
        if(i > 0 && j < height - 1)
            gpu_set_mat_row(dev_mixedSig+threadID*b_count*mixedSigSize+6*b_count, dev_signalData+((threadID+width-1)*b_count), b_count);
        if(j < height - 1 )
            gpu_set_mat_row(dev_mixedSig+threadID*b_count*mixedSigSize+7*b_count, dev_signalData+((threadID+width)*b_count), b_count);
        if(i < width - 1 && j < height - 1)
            gpu_set_mat_row(dev_mixedSig+threadID*b_count*mixedSigSize+8*b_count, dev_signalData+((threadID+width+1)*b_count), b_count);
    }

    dev_fib_fa[threadID] = 0.5;
    dev_fib_dir[3*threadID] = 0.5;
    dev_fib_dir[3*threadID+1] = 0.5;
    dev_fib_dir[3*threadID+2] = 0.5;
    dev_md[threadID] = 0.5;
    dev_d0[threadID] =  0.5;
    dev_d1[threadID] = 0.5;

    // DTI
    /*float *signal = new float [b_count-1];
    float *space = new float [b_count];
    double tensor[9];
    double V[9],d[3];
    double tensor_param[6];
    memcpy((void*)space, (void*)(dev_signalData+(threadID * b_count)), sizeof(float) * b_count);
    if (space[0] != 0.0)
    {
        float logs0 = (float)log(fmax(double(1.0),double(space[0])));
        for (unsigned int i = 1; i < b_count; ++i)
            signal[i-1] = (float)fmax(double(0.0),double(logs0)-log(fmax(1.0,(double)space[i])));
    }
    gpu_matrix_multiplication(dev_invg_dg, signal, tensor_param, 6, 64, 1);
    unsigned int tensor_index[9] = {0,3,4,3,1,5,4,5,2};
    for(unsigned int index = 0; index < 9; index++)
        tensor[index] = tensor_param[tensor_index[index]];
    eigen_decomposition_sym(tensor,V,d,3, 3);
    if (d[1] < 0.0)
    {
        d[1] = 0.0;
        d[2] = 0.0;
    }
    if (d[2] < 0.0)
        d[2] = 0.0;
    if (d[0] < 0.0)
    {
        d[0] = 0.0;
        d[1] = 0.0;
        d[2] = 0.0;
    }

    dev_fib_dir[3*threadID] = V[0];
    dev_fib_dir[3*threadID+1] = V[1];
    dev_fib_dir[3*threadID+2] = V[2];
    dev_fib_fa[threadID] = gpu_get_fa(d[0], d[1], d[2]);
    printf("fib_fa: %f", dev_fib_fa[threadID]);
    dev_md[threadID] = (d[0]+d[1]+d[2])/3.0;
    dev_d0[threadID] =  d[0];
    dev_d1[threadID] = (d[1]+d[2])/2.0;


    if (data.fa[0] < voxel.FAth)
    {

        for(n=0; n<b_count; n++)
        {
            x[n] = mixedSig(center_voxel,n);
            xstar[n] = mixedSig(center_voxel,n);
            mydata.x[n] = mixedSig(center_voxel,n);
            weight0[n]= (b_count-2+1)/(x[n]*x[n]);
        }

            par[0] = 0.5f;
            par[1] = lambda;
            mydata.n = 0; maxIteration = niter*1;

            result = slevmar_blec_dif(&cost_function, par, x, 2, b_count, p_min0, p_max0, A0, B1, 1, weight0, maxIteration, opts, info, NULL, NULL, (void*)&mydata);

            e[ica_num] = info[1];
            SC[ica_num] = logf(e[ica_num]/b_count)+1*logf(b_count)/b_count;

            V[0] = V[1] = V[2] = 0;
            std::copy(V, V+3, voxel.fib_dir.begin() + data.voxel_index * 3);
            std::copy(V, V+3, voxel.fib_dir.begin() + voxel.dim.size() * 3 + data.voxel_index * 3);
            std::copy(V, V+3, voxel.fib_dir.begin() + 2*voxel.dim.size() * 3 + data.voxel_index * 3);
            voxel.fr[data.voxel_index] = 0;
            voxel.fr[voxel.dim.size()+data.voxel_index] = 0;
            voxel.fr[2*voxel.dim.size()+data.voxel_index] = 0;

    }
    else
    {
      for(ica_num = 1; ica_num <=3; ica_num++)
      {
        float eigenvectors[9]; // BSM
        float fractions[4]; // BSM
        numOfIC = ica_num;

            if(ica_num == 1 || ica_num == 3)
                g = FICA_NONLIN_POW3;
            else
                g = FICA_NONLIN_TANH;

            itpp::Fast_ICA fi(mixedSig);
            fi.set_approach(approach);
            fi.set_nrof_independent_components(numOfIC);
            fi.set_non_linearity(g);
            fi.set_fine_tune(finetune);
            fi.set_a1(a1);
            fi.set_a2(a2);
            fi.set_mu(mu);
            fi.set_epsilon(epsilon);
            fi.set_sample_size(sampleSize);
            fi.set_stabilization(stabilization);
            fi.set_max_num_iterations(maxNumIterations);
            fi.set_max_fine_tune(maxFineTune);
            fi.set_first_eig(firstEig);
            fi.set_last_eig(lastEig);
            fi.set_pca_only(PCAonly);

            try{
                fi.separate();
            }
            catch(...)
            {
                qDebug() << "Exception!!!.";
            }

            icasig = fi.get_independent_components();
            icasig_no_log.set_size(icasig.rows(), icasig.cols());
            icasig_no_log = icasig;
            mixing_matrix = fi.get_mixing_matrix();

            if(icasig.rows() < ica_num) // Add zero if the number of ICA signal is less than number of fibers
            {
                icasig.set_size(ica_num, icasig.cols());
                itpp::vec tmp(icasig.cols());
                tmp.zeros();
                for(m = icasig.rows(); m < ica_num-1; m++)
                {
                    icasig.set_col(m, tmp);
                }
            }

            for(m = 0; m < icasig.rows(); m++)
            {
                double sum = 0;
                for(n=0;n<icasig.cols();n++)
                    sum+=icasig.get(m,n);
                if(sum<0)
                {
                    for(n=0;n<icasig.cols();n++)
                        icasig.set(m, n, icasig.get(m,n)*-1);
                    for(n=0;n<mixing_matrix.rows();n++)
                        mixing_matrix.set(n, m, mixing_matrix.get(n, m)*-1);
                }
                double min_value = icasig.get(m, 0);
                double max_value = icasig.get(m, 0);
                for(n=0;n<icasig.cols();n++)
                {
                    if(icasig.get(m,n)>max_value)
                        max_value = icasig.get(m, n);
                    if(icasig.get(m,n)<min_value)
                        min_value = icasig.get(m,n);
                }
                for(n=0;n<icasig.cols();n++)
                {
                    double tmp = icasig.get(m, n);
                    double t = tmp - min_value;
                    double b = max_value - min_value;
                    double f = (t/b)*0.8+.1;
                    icasig.set(m,n, f);
                    icasig_no_log.set(m,n, f);
                    icasig.set(m,n, std::log(icasig.get(m,n)));
                }
            }

            for(m=0; m<icasig.rows(); m++)
            {
                get_mat_row(icasig, signal, m);
                arma::mat matsignal(icasig.cols(),1);
                set_arma_col(matsignal, signal, 0);
                tensor_param = voxel.matinvg_dg * matsignal;

                unsigned int tensor_index[9] = {0,3,4,3,1,5,4,5,2};
                for(unsigned int index = 0; index < 9; index++)
                    tensor[index] = tensor_param(tensor_index[index],0);

                image::matrix::eigen_decomposition_sym(tensor,V,d,image::dim<3,3>());
                if (d[1] < 0.0)
                {
                    d[1] = 0.0;
                    d[2] = 0.0;
                }
                if (d[2] < 0.0)
                    d[2] = 0.0;
                if (d[0] < 0.0)
                {
                    d[0] = 0.0;
                    d[1] = 0.0;
                    d[2] = 0.0;
                }
                std::copy(V, V+3, voxel.fib_dir.begin() + m*voxel.dim.size() * 3 + data.voxel_index * 3);
                eigenvectors[m * 3 + 0] = V[0]; // we need this for BSM
                eigenvectors[m * 3 + 1] = V[1];
                eigenvectors[m * 3 + 2] = V[2];
            }

            icasig = icasig_no_log;

            get_mat_row(mixing_matrix, w, center_voxel);
            float sum = 0;
            for(m=0; m<icasig.rows(); m++)
                sum += abs(w[m]);

            for(m=0; m<icasig.rows(); m++)
            {
                fractions[m+1] = abs(w[m])/sum;
                voxel.fr[m*voxel.dim.size()+data.voxel_index] = fractions[m+1];
            }
            icasig.clear();
            icasig_no_log.clear();

            for(n=0; n<b_count; n++)
            {
                x[n] = mixedSig(center_voxel,n);
                xstar[n] = mixedSig(center_voxel,n);
                mydata.x[n] = mixedSig(center_voxel,n);
                weight1[n]= (b_count-6+1)/(x[n]*x[n]);
                weight2[n]= (b_count-10+1)/(x[n]*x[n]);
                weight3[n]= (b_count-14+1)/(x[n]*x[n]);
            }

            switch(ica_num)
            {
            case 1: // one stick

                sum=0.0f;
                for(n=0; n<2; n++)
                {
                    if (n==0)
                        par[n] =  ((float) rand()/(RAND_MAX));
                    else
                        par[n] =  2.0f + ((float) rand()/(RAND_MAX)); //0.25f; // divide 1 between four isotropic fractions
                        sum+=par[n];
                }

                for(n=0; n<2; n++)
                    par[n] =  par[n]/sum; //0.25f; // divide 1 between four isotropic fractions

                for(n=0; n<3; n++)
                {    par[n+2] = eigenvectors[n];
                     p_min1[n+2] = std::min<float>(par[n+2]*0.9f,par[n+2]*1.1f);
                     p_max1[n+2] = std::max<float>(par[n+2]*0.9f,par[n+2]*1.1f);
                }

                par[5] = lambda + (((float) rand()/(RAND_MAX))-0.5f)/10;

                mydata.n = 1; maxIteration = niter*6;

                result = slevmar_blec_dif(&cost_function, par, x, 6, b_count, p_min1, p_max1, A1, B1, 1, weight1, maxIteration, opts, info, NULL, NULL, (void*)&mydata);

                e[ica_num] = info[1];
                SC[ica_num] = logf(e[ica_num]/b_count)+6*logf(b_count)/b_count;
                if(SC_min > SC[ica_num])
                {
                    SC_min = SC[ica_num];
                    min_index = ica_num;
                    F_best[0] = std::abs<float>(par[1]);
                    for(n=0; n<3; n++)
                        V_best[n]=par[n+2];
                }
                break;
            case 2: // two sticks
                sum=0.0f;
                for(n=0; n<3; n++)
                {
                    if (n == 0)
                        par[n] =  ((float) rand()/(RAND_MAX));
                    else
                        par[n] =  2.0f + ((float) rand()/(RAND_MAX)); //0.25f; // divide 1 between four isotropic fractions
                        sum+=par[n];
                }

                for(n=0; n<3; n++)
                    par[n] =  par[n]/sum; //0.25f; // divide 1 between four isotropic fractions

                for(n=0; n<6; n++)
                {    par[n+3] = eigenvectors[n];
                    p_min3[n+3] = std::min<float>(par[n+3]*0.9f,par[n+3]*1.1f);
                    p_max3[n+3] = std::max<float>(par[n+3]*0.9f,par[n+3]*1.1f);
                }

                par[9] = lambda  + (((float) rand()/(RAND_MAX))-0.5f)/10;

                mydata.n = 2; maxIteration = niter*10;

                result = slevmar_blec_dif(&cost_function, par, x, 10, b_count, p_min2, p_max2, A2, B1, 1, weight2, maxIteration, opts, info, NULL, NULL, (void*)&mydata);

                e[ica_num] = info[1];
                SC[ica_num] = logf(e[ica_num]/b_count)+10*logf(b_count)/b_count;
                if(SC_min > SC[ica_num])
                {
                    SC_min = SC[ica_num];
                    min_index = ica_num;
                    F_best[0] = std::abs<float>(par[1]);
                    F_best[1] = std::abs<float>(par[2]);
                    for(n=0; n<6; n++)
                        V_best[n]=par[n+3];
                }
                break;
            case 3:
                sum=0.0f;
                for(n=0; n<4; n++)
                {
                    if (n == 0)
                        par[n] =  ((float) rand()/(RAND_MAX));
                    else
                        par[n] =  2.0f + ((float) rand()/(RAND_MAX)); //0.25f; // divide 1 between four isotropic fractions
                        sum+=par[n];
                }

                for(n=0; n<4; n++)
                    par[n] =  par[n]/sum; //0.25f; // divide 1 between four isotropic fractions

                for(n=0; n<9; n++) // copy eigenvectors to par.
                {    par[n+4] = eigenvectors[n];
                    p_min3[n+4] = std::min<float>(par[n+4]*0.9f,par[n+4]*1.1f);
                    p_max3[n+4] = std::max<float>(par[n+4]*0.9f,par[n+4]*1.1f);
                }

                par[13] = lambda + (((float) rand()/(RAND_MAX))-0.5f)/10;

                mydata.n = 3; maxIteration = niter*14;

                result = slevmar_blec_dif(&cost_function, par, x, 14, b_count, p_min3, p_max3, A3, B1, 1, weight3, maxIteration, opts, info, NULL, NULL, (void*)&mydata);

                e[ica_num] = info[1];
                SC[ica_num] = logf(e[ica_num]/b_count)+14*logf(b_count)/b_count;
                if(SC_min > SC[ica_num])
                {
                    SC_min = SC[ica_num];
                    min_index = ica_num;
                    F_best[0] = std::abs<float>(par[1]);
                    F_best[1] = std::abs<float>(par[2]);
                    F_best[2] = std::abs<float>(par[3]);
                    for(n=0; n<9; n++)
                        V_best[n]=par[n+4];
                }
                break;
            }
        }

        switch(min_index)
        {
         case 1:
            V[0] = V[1] = V[2] = 0;
            std::copy(V_best, V_best+3, voxel.fib_dir.begin() + data.voxel_index * 3);
            std::copy(V, V+3, voxel.fib_dir.begin() + voxel.dim.size() * 3 + data.voxel_index * 3);
            std::copy(V, V+3, voxel.fib_dir.begin() + 2*voxel.dim.size() * 3 + data.voxel_index * 3);
            voxel.fr[data.voxel_index] = F_best[0];
            voxel.fr[voxel.dim.size()+data.voxel_index] = 0;
            voxel.fr[2*voxel.dim.size()+data.voxel_index] = 0;
            break;
        case 2:
            V[0] = V[1] = V[2] = 0;
            std::copy(V_best, V_best+3, voxel.fib_dir.begin() + data.voxel_index * 3);
            std::copy(V_best+3, V_best+6, voxel.fib_dir.begin() + voxel.dim.size() * 3 + data.voxel_index * 3);
            std::copy(V, V+3, voxel.fib_dir.begin() + 2*voxel.dim.size() * 3 + data.voxel_index * 3);
            voxel.fr[data.voxel_index] = F_best[0];
            voxel.fr[voxel.dim.size()+data.voxel_index] = F_best[1];
            voxel.fr[2*voxel.dim.size()+data.voxel_index] = 0;
            break;
        case 3:
            std::copy(V_best, V_best+3, voxel.fib_dir.begin() + data.voxel_index * 3);
            std::copy(V_best+3, V_best+6, voxel.fib_dir.begin() + voxel.dim.size() * 3 + data.voxel_index * 3);
            std::copy(V_best+6, V_best+9, voxel.fib_dir.begin() + 2*voxel.dim.size() * 3 + data.voxel_index * 3);
            voxel.fr[data.voxel_index] = F_best[0];
            voxel.fr[voxel.dim.size()+data.voxel_index] = F_best[1];
            voxel.fr[2*voxel.dim.size()+data.voxel_index] = F_best[2];
            break;
        }
        num_fibers[data.voxel_index] = min_index;
    }*/

    //delete [] signal;
    //delete [] space;
    //delete [] mixedSig;
}

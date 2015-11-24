#include <cuda.h>
#include <cuda_runtime.h>
#include "cudaKernels.h"
#include "basic_voxel.hpp"

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
__device__ void gpu_matrix_multiplication(float *a, float *b, float *c, int i, int j, int k)
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

void LaunchICABSMKernel(Voxel &voxel)
{
    icabsm_kernel<<<1, voxel.dim.size()>>>(voxel.threeDimensionalWindow,
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
                                           voxel.gpu.dev_signalData);
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
            gpu_set_mat_row(mixedSig+11*b_count, dev_signalData+((threadID+width-1)*b_count), b_count);
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
        if(i > 0 && j > 0)
            gpu_set_mat_row(mixedSig, dev_signalData+((threadID-width-1)*b_count), b_count);
        if(j > 0)
            gpu_set_mat_row(mixedSig+b_count, dev_signalData+((threadID-width)*b_count), b_count);
        if(i < width - 1 && j > 0)
            gpu_set_mat_row(mixedSig+2*b_count, dev_signalData+((threadID-width+1)*b_count), b_count);
        if(i > 0)
            gpu_set_mat_row(mixedSig+3*b_count, dev_signalData+((threadID-1)*b_count), b_count);
        gpu_set_mat_row(mixedSig+4*b_count, dev_signalData+(threadID*b_count), b_count);
        if(i < width - 1)
            gpu_set_mat_row(mixedSig+5*b_count, dev_signalData+((threadID+1)*b_count), b_count);
        if(i > 0 && j < height - 1)
            gpu_set_mat_row(mixedSig+6*b_count, dev_signalData+((threadID+width-1)*b_count), b_count);
        if(j < height - 1 )
            gpu_set_mat_row(mixedSig+7*b_count, dev_signalData+((threadID+width)*b_count), b_count);
        if(i < width - 1 && j < height - 1)
            gpu_set_mat_row(mixedSig+8*b_count, dev_signalData+((threadID+width+1)*b_count), b_count);
    }

    // DTI
    float *signal = new float [b_count-1];
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
        tensor[index] = tensor_param(tensor_index[index],0);
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
    dev_fib_fa[threadID] = gpu_get_fa(d[0], d[1], d[2]);
    dev_md[threadID] = (d[0]+d[1]+d[2])/3.0;
    dev_d0[threadID] =  d[0];
    dev_d1[threadID] = (d[1]+d[2])/2.0;

    delete [] signal;
    delete [] space;
    delete [] mixedSig;
}

#ifndef ICA_PROCESS_HPP
#define ICA_PROCESS_HPP
#include <cmath>
#include "basic_voxel.hpp"
#include "image/image.hpp"
#include "itpp/itsignal.h"
#include "itbase.h"
#include <qdebug>
#include <iostream>
using namespace std;

using namespace itpp;

class ICABSM : public BaseProcess
{
private:
    int approach, numOfIC, g, initState;
    bool finetune, stabilization, PCAonly;
    double a1, a2, mu, epsilon, sampleSize;
    int maxNumIterations, maxFineTune;
    int firstEig, lastEig;
    mat mixedSig, icasig;
    mat mixing_matrix;
    std::vector<float> d0;
    std::vector<float> d1;
    std::vector<float> md;
    std::vector<double> iKtK; // 6-by-6
    std::vector<unsigned int> iKtK_pivot;
    std::vector<double> Kt;
    unsigned int b_count;
    float get_fa(float l1, float l2, float l3)
    {
        float ll = (l1+l2+l3)/3.0;
        if(l1 == 0.0)
            return 0.0;
        float ll1 = l1 - ll;
        float ll2 = l2 - ll;
        float ll3 = l3 - ll;
        return std::min(1.0, std::sqrt(1.5*(ll1*ll2+ll2*ll2+ll3*ll3)/(l1*l1+l2*l2+l3*l3)));
    }

    void set_mat_row(mat &m, std::vector<float>& v, int r)
    {
        int i;
        int size = v.size();
        for(i = 0; i < size; i++)
            m.set(r, i, v[i]);
    }

    void get_mat_row(mat &m, std::vector<float>& v, int r)
    {
        int i;
        int size = m.cols();
        v.resize(size);
        for(i = 0; i< size; i++)
            v[i] = m.get(r, i);
    }

public:
    virtual void init(Voxel& voxel)
    {
        md.clear();
        md.resize(voxel.dim.size());
        d0.clear();
        d0.resize(voxel.dim.size());
        d1.clear();
        d1.resize(voxel.dim.size());

        approach = FICA_APPROACH_SYMM;
        numOfIC = 3;
        g = FICA_NONLIN_TANH;
        initState = FICA_INIT_RAND;
        finetune = false;
        stabilization = false;
        PCAonly = false;
        a1 = 1;
        a2 = 1;
        mu = 1;
        epsilon = 0.0001f;
        sampleSize = 1;
        maxNumIterations = 1000;
        maxFineTune = 5;
        firstEig = 1;
        lastEig = 3;
        mixedSig.set_size(9, 65);
        mixedSig.zeros();
        voxel.fr.clear();
        voxel.fr.resize(numOfIC* voxel.dim.size());
        voxel.fib_fa.clear();
        voxel.fib_fa.resize(voxel.dim.size());
        voxel.fib_dir.clear();
        voxel.fib_dir.resize(voxel.dim.size()*3*numOfIC);

        b_count = voxel.bvalues.size()-1;
        std::vector<image::vector<3> > b_data(b_count);
        //skip null
        std::copy(voxel.bvectors.begin()+1,voxel.bvectors.end(),b_data.begin());
        for(unsigned int index = 0; index < b_count; ++index)
            b_data[index] *= std::sqrt(voxel.bvalues[index+1]);

        Kt.resize(6*b_count);
        {
            unsigned int qmap[6]		= {0  ,4  ,8  ,1  ,2  ,5  };
            double qweighting[6]= {1.0,1.0,1.0,2.0,2.0,2.0};
            //					  bxx,byy,bzz,bxy,bxz,byz
            for (unsigned int i = 0,k = 0; i < b_data.size(); ++i,k+=6)
            {
                //qq = q qT
                std::vector<float> qq(3*3);
                image::matrix::product_transpose(b_data[i].begin(),b_data[i].begin(),qq.begin(),
                                               image::dyndim(3,1),image::dyndim(3,1));

                /*
                      q11 q15 q19 2*q12 2*q13 2*q16
                      q21 q25 q29 2*q22 2*q23 2*q26
                K  = | ...                         |
                */
                for (unsigned int col = 0,index = i; col < 6; ++col,index+=b_count)
                    Kt[index] = qq[qmap[col]]*qweighting[col];
            }
        }
        iKtK.resize(6*6);
        iKtK_pivot.resize(6);
        image::matrix::product_transpose(Kt.begin(),Kt.begin(),iKtK.begin(),
                                       image::dyndim(6,b_count),image::dyndim(6,b_count));
        image::matrix::lu_decomposition(iKtK.begin(),iKtK_pivot.begin(),image::dyndim(6,6));

    }
public:
    virtual void run(Voxel& voxel, VoxelData& data)
    {
        unsigned int i=0,j=0,k=0,m=0,n=0;
        /* get 3d position */
        unsigned int index = data.voxel_index;
        k = (unsigned int) index / (voxel.dim.h * voxel.dim.w);
        index -= k * voxel.dim.h * voxel.dim.w;

        j = (unsigned int) index / voxel.dim.w;
        index -= j * voxel.dim.w;

        i = (unsigned int) index / 1;

        if(i > 0 && j > 0)
            set_mat_row(mixedSig, voxel.signalData[data.voxel_index-voxel.dim.w-1], 0);

        if(j > 0)
            set_mat_row(mixedSig, voxel.signalData[data.voxel_index-voxel.dim.w], 1);

        if(i < voxel.dim.w - 1 && j > 0 )
            set_mat_row(mixedSig, voxel.signalData[data.voxel_index-voxel.dim.w+1], 2);

        if(i > 0)
            set_mat_row(mixedSig, voxel.signalData[data.voxel_index-1], 3);

        set_mat_row(mixedSig, voxel.signalData[data.voxel_index], 4);

        if(i < voxel.dim.w - 1)
            set_mat_row(mixedSig, voxel.signalData[data.voxel_index+1], 5);

        if(i > 0 && j < voxel.dim.h - 1)
            set_mat_row(mixedSig, voxel.signalData[data.voxel_index+voxel.dim.w-1], 6);

        if(j < voxel.dim.h - 1 )
            set_mat_row(mixedSig, voxel.signalData[data.voxel_index+voxel.dim.w], 7);

        if(i < voxel.dim.w - 1 && j < voxel.dim.h - 1)
            set_mat_row(mixedSig, voxel.signalData[data.voxel_index+voxel.dim.w+1], 8);

        Fast_ICA fi(mixedSig);
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
        mixing_matrix = fi.get_mixing_matrix();

        for(m = 0; m < icasig.rows(); m++)
        {
            double sum = 0;
            double min_value = icasig.get(m, 0);
            double max_value = icasig.get(m, 0);
            for(n=0;n<icasig.cols();n++)
            {
                sum+=icasig.get(m,n);
                if(icasig.get(m,n)>max_value)
                    max_value = icasig.get(m, n);
                if(icasig.get(m,n)<min_value)
                    min_value = icasig.get(m,n);
            }
            if(sum<0)
            {
                for(n=0;n<icasig.cols();n++)
                    icasig.set(m, n, icasig.get(m,n)*-1);
                for(n=0;n<mixing_matrix.rows();n++)
                    mixing_matrix.set(n, m, mixing_matrix.get(n, m)*-1);
            }
            for(n=0;n<icasig.cols();n++)
            {
                double tmp = icasig.get(m, n);
                double t = tmp - min_value;
                double b = max_value - min_value;
                double f = (t/b)*0.8+1;
                icasig.set(m,n, f);
                icasig.set(m,n, std::max<float>(0.0, std::log(std::max<float>(1.0, icasig.get(m,n)))));
            }
        }

        double KtS[6], tensor_param[6];
        double tensor[9];
        double V[9],d[3];
        std::vector<float> signal(icasig.cols());
        std::vector<float> w(icasig.rows());

        for(m=0; m<icasig.rows(); m++)
        {
            get_mat_row(icasig, signal, m);

            image::matrix::product(Kt.begin(),signal.begin(),KtS,image::dyndim(6,b_count),image::dyndim(b_count,1));
            image::matrix::lu_solve(iKtK.begin(),iKtK_pivot.begin(),KtS,tensor_param,image::dyndim(6,6));
            unsigned int tensor_index[9] = {0,3,4,3,1,5,4,5,2};
            for (unsigned int index = 0; index < 9; ++index)
                tensor[index] = tensor_param[tensor_index[index]];

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
        }
        get_mat_row(mixing_matrix, w, 4);
        float sum = 0;
        for(m=0; m<icasig.rows(); m++)
            sum += abs(w[m]);

        for(m=0; m<icasig.rows(); m++)
            voxel.fr[m*voxel.dim.size()+data.voxel_index] = abs(w[m])/sum;

        if (data.space.front() != 0.0)
        {
            float logs0 = std::log(std::max<float>(1.0,data.space.front()));
            for (unsigned int i = 1; i < data.space.size(); ++i)
                signal[i-1] = std::max<float>(0.0,logs0-std::log(std::max<float>(1.0,data.space[i])));
        }

        image::matrix::product(Kt.begin(),signal.begin(),KtS,image::dyndim(6,b_count),image::dyndim(b_count,1));
        image::matrix::lu_solve(iKtK.begin(),iKtK_pivot.begin(),KtS,tensor_param,image::dyndim(6,6));


        unsigned int tensor_index[9] = {0,3,4,3,1,5,4,5,2};
        for (unsigned int index = 0; index < 9; ++index)
            tensor[index] = tensor_param[tensor_index[index]];

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
        data.fa[0] = voxel.fib_fa[data.voxel_index] = get_fa(d[0],d[1],d[2]);
        md[data.voxel_index] = 1000.0*(d[0]+d[1]+d[2])/3.0;
        d0[data.voxel_index] = 1000.0*d[0];
        d1[data.voxel_index] = 1000.0*(d[1]+d[2])/2.0;
    }
    virtual void end(Voxel& voxel,gz_mat_write& mat_writer)
    {
        mat_writer.write("dir0",&*voxel.fib_dir.begin(), 1, voxel.fib_dir.size()/3);
        mat_writer.write("dir1",&*voxel.fib_dir.begin()+voxel.dim.size()*3, 1, voxel.fib_dir.size()/3);
        mat_writer.write("dir2",&*voxel.fib_dir.begin()+2*voxel.dim.size()*3, 1, voxel.fib_dir.size()/3);
        mat_writer.write("fa0",&*voxel.fr.begin(), 1, voxel.fr.size()/3);
        mat_writer.write("fa1",&*voxel.fr.begin()+voxel.dim.size(), 1, voxel.fr.size()/3);
        mat_writer.write("fa2",&*voxel.fr.begin()+2*voxel.dim.size(), 1, voxel.fr.size()/3);
        mat_writer.write("gfa",&*voxel.fib_fa.begin(), 1, voxel.fib_fa.size());
        mat_writer.write("adc",&*md.begin(),1,md.size());
        mat_writer.write("axial_dif",&*d0.begin(),1,d0.size());
        mat_writer.write("radial_dif",&*d1.begin(),1,d1.size());
    }
};

#endif//_PROCESS_HPP

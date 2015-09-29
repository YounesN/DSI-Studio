#ifndef ICA_PROCESS_HPP
#define ICA_PROCESS_HPP
#include <cmath>
#include "basic_voxel.hpp"
#include "image/image.hpp"
#include "itpp/itsignal.h"
#include "itbase.h"
#include <qdebug>
#include <iostream>
#include "armadillo"

using namespace std;

class ICABSM : public BaseProcess
{
private:
    int approach, numOfIC, g, initState;
    bool finetune, stabilization, PCAonly;
    double a1, a2, mu, epsilon, sampleSize;
    int maxNumIterations, maxFineTune;
    int firstEig, lastEig;
    itpp::mat mixedSig, icasig;
    itpp::mat mixing_matrix;
    std::vector<float> d0;
    std::vector<float> d1;
    std::vector<float> md;
    unsigned int b_count;
    float get_fa(float l1, float l2, float l3)
    {
        float ll = (l1+l2+l3)/3.0;
        if(l1 == 0.0)
            return 0.0;
        float ll1 = l1 - ll;
        float ll2 = l2 - ll;
        float ll3 = l3 - ll;
        return std::min(1.0,std::sqrt(1.5*(ll1*ll1+ll2*ll2+ll3*ll3)/(l1*l1+l2*l2+l3*l3)));
    }

    void set_mat_row(itpp::mat &m, std::vector<float>& v, int r)
    {
        int i;
        int size = v.size();
        for(i = 0; i < size; i++)
            m.set(r, i, v[i]);
    }

    void get_mat_row(itpp::mat &m, std::vector<float>& v, int r)
    {
        int i;
        int size = m.cols();
        v.resize(size);
        for(i = 0; i< size; i++)
            v[i] = m.get(r, i);
    }
    void set_arma_col(arma::mat &m, std::vector<float> v, int r)
    {
        int i;
        int size = v.size();
        for(i=0; i<size; i++)
            m(i,r) = v[i];
    }

public:
    virtual void init(Voxel& voxel)
    {
        int m,n;
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
        mixedSig.set_size(9, 64);
        mixedSig.zeros();
        voxel.fr.clear();
        voxel.fr.resize(numOfIC* voxel.dim.size());
        voxel.fib_fa.clear();
        voxel.fib_fa.resize(voxel.dim.size());
        voxel.fib_dir.clear();
        voxel.fib_dir.resize(voxel.dim.size()*3*numOfIC);

        b_count = voxel.bvalues.size()-1;
        voxel.g_dg.clear();
        voxel.g_dg.resize(6*b_count);
        voxel.invg_dg.clear();
        voxel.invg_dg.resize(6*b_count);
        voxel.matg_dg.set_size(b_count, 6);

        for(m=0; m<b_count; m++)
        {
            voxel.matg_dg(m,0) = voxel.g_dg[6*m+0] = voxel.bvectors[m+1][0]*voxel.bvectors[m+1][0];
            voxel.matg_dg(m,1) = voxel.g_dg[6*m+1] = voxel.bvectors[m+1][1]*voxel.bvectors[m+1][1];
            voxel.matg_dg(m,2) = voxel.g_dg[6*m+2] = voxel.bvectors[m+1][2]*voxel.bvectors[m+1][2];
            voxel.matg_dg(m,3) = voxel.g_dg[6*m+3] = voxel.bvectors[m+1][0]*voxel.bvectors[m+1][1]*2;
            voxel.matg_dg(m,4) = voxel.g_dg[6*m+4] = voxel.bvectors[m+1][0]*voxel.bvectors[m+1][2]*2;
            voxel.matg_dg(m,5) = voxel.g_dg[6*m+5] = voxel.bvectors[m+1][1]*voxel.bvectors[m+1][2]*2;
        }
        arma::mat ones;
        ones.ones(1,6);
        arma::mat bvalue(64,1);
        for(m=0; m<b_count; m++)
            bvalue(m, 0) = voxel.bvalues[m+1];
        bvalue = bvalue * ones;
        for(m=0;m<b_count;m++)
            for(n=0;n<6;n++)
                voxel.matg_dg(m,n)*=bvalue(m,n);
        voxel.matinvg_dg = arma::pinv(voxel.matg_dg);
        voxel.matinvg_dg = -voxel.matinvg_dg;
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

        arma::mat tensor_param(6,1);
        double tensor[9];
        double V[9],d[3];
        std::vector<float> signal(icasig.cols());
        std::vector<float> w(icasig.rows());

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

        arma::mat matsignal(icasig.cols(),1);
        set_arma_col(matsignal, signal, 0);
        arma::mat pos_invg_dg = -voxel.matinvg_dg;
        tensor_param = pos_invg_dg * matsignal;

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
        data.fa[0] = voxel.fib_fa[data.voxel_index] = get_fa(d[0], d[1], d[2]);
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

        ofstream out;
        out.open("out.txt", ios::app);
        for(int i=0;i<voxel.fib_fa.size(); i++)
            out << voxel.fib_fa[i] << std::endl;
        out.close();

        mat_writer.write("gfa",&*voxel.fib_fa.begin(), 1, voxel.fib_fa.size());
        mat_writer.write("adc",&*md.begin(),1,md.size());
        mat_writer.write("axial_dif",&*d0.begin(),1,d0.size());
        mat_writer.write("radial_dif",&*d1.begin(),1,d1.size());
    }
};

#endif//_PROCESS_HPP

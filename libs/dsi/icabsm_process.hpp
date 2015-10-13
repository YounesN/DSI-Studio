#ifndef ICABSM_PROCESS_HPP
#define ICABSM_PROCESS_HPP
#include <cmath>
#include "basic_voxel.hpp"
#include "image/image.hpp"
#include "itpp/itsignal.h"
#include "levmar/levmar.h"
#include "itbase.h"
#include <qdebug>
#include <iostream>
#include "armadillo"

using namespace std;

struct ad {
    int n;
    Voxel *voxel;
    float *x;
};

void cost_function(float *p, float *hx, int m, int n, void *adata);

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

    // BSM variables
    float par[14];
    float p_min0[1];
    float p_max0[1];
    float p_min1[6];
    float p_max1[6];
    float p_min2[10];
    float p_max2[10];
    float p_min3[14];
    float p_max3[14];
    float fractions[4];
    float lambda;
    float eigenvectors[9];
    float info[LM_INFO_SZ];

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
    float error_value(float *x, float *xstar, int size)
    {
        int i=0;
        float ret = 0.0f;
        for(i=0; i<size; i++)
            ret += (x[i]-xstar[i])*(x[i]-xstar[i]);
        return ret;
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

        p_min0[0] = 0.0010f;
        p_max0[0] = 0.0020f;
        p_min1[0] = 0.05f; p_min1[1] = 0.05f; p_min1[2] = -1.0f; p_min1[3] = -1.0f; p_min1[4] = -1.0f; p_min1[5] = 0.0010f;
        p_max1[0] = 0.95f; p_max1[1] = 0.95f; p_max1[2] = 1.0f; p_max1; p_max1[3] = 1.0f; p_max1[4] = 1.0f; p_max1[5] = 0.0020f;
        p_min2[0] = 0.05f; p_min2[1] = 0.05f; p_min2[2] = 0.05f; p_min2[3] = -1.0f; p_min2[4] = -1.0f; p_min2[5] = -1.0f; p_min2[6] = -1.0f;
        p_min2[7] = -1.0f; p_min2[8] = -1.0f; p_min2[9] = 0.0010f;
        p_max2[0] = 0.95f; p_max2[1] = 0.95f; p_max2[2] = 0.95f; p_max2[3] = 1.0f; p_max2[4] = 1.0f; p_max2[5] = 1.0f; p_max2[6] = 1.0f;
        p_max2[7] = 1.0f; p_max2[8] = 1.0f; p_max2[9] = 0.0020f;
        p_min3[0] = 0.05f; p_min3[1] = 0.05f; p_min3[2] = 0.05f; p_min3[3] = 0.05f; p_min3[4] = -1.0f; p_min3[5] = -1.0f; p_min3[6] = -1.0f;
        p_min3[7] = -1.0f; p_min3[8] = -1.0f; p_min3[9] = -1.0f; p_min3[10] = -1.0f; p_min3[11] = -1.0f; p_min3[12] = -1.0f; p_min3[13] = 0.0010f;
        p_max3[0] = 0.95f; p_max3[1] = 0.95f; p_max3[2] = 0.95f; p_max3[3] = 0.95f; p_max3[4] = 1.0f; p_max3[5] = 1.0f; p_max3[6] = 1.0f;
        p_max3[7] = 1.0f; p_max3[8] = 1.0f; p_max3[9] = 1.0f; p_max3[10] = 1.0f; p_max3[11] = 1.0f; p_max3[12] = 1.0f; p_max3[13] = 0.0020f;

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

        lambda = 0.0015f;

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
        // ICA
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
            eigenvectors[m * 3 + 0] = V[0]; // we need this for BSM
            eigenvectors[m * 3 + 1] = V[1];
            eigenvectors[m * 3 + 2] = V[2];
        }
        get_mat_row(mixing_matrix, w, 4);
        float sum = 0;
        for(m=0; m<icasig.rows(); m++)
            sum += abs(w[m]);

        for(m=0; m<icasig.rows(); m++)
        {
            fractions[m+1] = abs(w[m])/sum;
            voxel.fr[m*voxel.dim.size()+data.voxel_index] = fractions[m+1];
        }

        // BSM
        ad mydata;
        mydata.voxel = &voxel;
        mydata.n = icasig.rows();
        float opts[LM_OPTS_SZ];
        opts[0] = 1E-3;
        opts[1] = opts[2] = 1E-15;
        opts[3] = 1E-15;
        opts[4] = 1E-3;
        int b_count = voxel.bvalues.size()-1;
        float *x = new float[b_count];
        float *xstar = new float[b_count];
        mydata.x = new float[b_count];
        for(m=0; m<b_count; m++)
        {
            x[m] = mixedSig(4,m);
            xstar[m] = mixedSig(4,m);
            mydata.x[m] = mixedSig(4,m);
        }
        float e[4]={0};
        float e_min=1E+15;
        int e_min_index = -1;
        float V_best[9];

        ofstream out;
        out.open("output.txt", ios::app);
        out << "===============================================================================" << std::endl;
        out << "Voxel :" << i << "x" << j << "x" << k << std::endl;
        out << "Parameters before BSM: " << std::endl;
        int result;

        for(m = 0;m <= icasig.rows(); m++)
        {
            switch(m)
            {
            case 0: // zero sticks
                par[0] = lambda;
                out << par[0] << std::endl;
                result = slevmar_bc_dif(&cost_function, par, x, 1, b_count, p_min0, p_max0, NULL, 100, opts, info, NULL, NULL, (void*)&mydata);
                out << "Parameters after BSM: " << std::endl;
                out << par[0] << std::endl;
                e[m] = (info[1]>=0)?info[1]:-1*info[1];
                if(e_min > e[m])
                {
                    e_min = e[m];
                    e_min_index = m;
                }
                break;
            case 1: // one stick
                fractions[1] *= 0.8;
                fractions[0] = 1-fractions[1];
                par[0] = fractions[0];
                par[1] = fractions[1];
                for(m=0; m<3; m++)
                    par[m+2] = eigenvectors[m];
                par[5] = lambda;
                for(m=0; m<6; m++)
                    out << par[m] << " ";
                out << std::endl;
                result = slevmar_bc_dif(&cost_function, par, x, 6, b_count, p_min1, p_max1, NULL, 100, opts, info, NULL, NULL, (void*)&mydata);
                out << "Parameters after BSM: " << std::endl;
                for(m=0; m<6; m++)
                    out << par[m] << " ";
                out << std::endl;
                e[m] = (info[1]>=0)?info[1]:-1*info[1];
                if(e_min > e[m])
                {
                    e_min = e[m];
                    e_min_index = m;
                    for(n=0; n<3; n++)
                        V_best[n]=par[n+2];
                }
                break;
            case 2: // two sticks
                fractions[1] *= 0.8;
                fractions[2] *= 0.8;
                fractions[0] = 1-fractions[0]-fractions[1];
                for(m=0; m<3; m++) // copy fractions to par.
                    par[m] = fractions[m];
                for(m=0; m<6; m++)
                    par[m+3] = eigenvectors[m];
                par[9] = lambda;
                for(m=0; m<10; m++)
                    out << par[m] << " ";
                out << std::endl;
                result = slevmar_bc_dif(&cost_function, par, x, 10, b_count, p_min2, p_max2, NULL, 100, opts, info, NULL, NULL, (void*)&mydata);
                out << "Parameters after BSM: " << std::endl;
                for(m=0; m<10; m++)
                    out << par[m] << " ";
                out << std::endl;
                e[m] = (info[1]>=0)?info[1]:-1*info[1];
                if(e_min > e[m])
                {
                    e_min = e[m];
                    e_min_index = m;
                    for(n=0; n<6; n++)
                        V_best[n]=par[n+3];
                }
                break;
            case 3: // three sticks
                fractions[1] *= 0.8;
                fractions[2] *= 0.8;
                fractions[3] *= 0.8;
                fractions[0] = 1-fractions[1]-fractions[2]-fractions[3];
                for(m=0; m<4; m++) // copy fractions to par.
                    par[m] = fractions[m];
                for(m=0; m<9; m++) // copy eigenvectors to par.
                    par[m+4] = eigenvectors[m];
                par[13] = lambda;
                for(m=0; m<14; m++)
                    out << par[m] << " ";
                out << std::endl;
                result = slevmar_bc_dif(&cost_function, par, x, 14, b_count, p_min3, p_max3, NULL, 100, opts, info, NULL, NULL, (void*)&mydata);
                out << "Parameters after BSM: " << std::endl;
                for(m=0; m<14; m++)
                    out << par[m] << " ";
                out << std::endl;
                e[m] = (info[1]>=0)?info[1]:-1*info[1];
                if(e_min > e[m])
                {
                    e_min = e[m];
                    e_min_index = m;
                    for(n=0; n<9; n++)
                        V_best[n]=par[n+4];
                }
                break;
            }
        }

        switch(e_min_index)
        {
        case 0:
            V[0] = V[1] = V[2] = 0;
            std::copy(V, V+3, voxel.fib_dir.begin() + data.voxel_index * 3);
            std::copy(V, V+3, voxel.fib_dir.begin() + voxel.dim.size() * 3 + data.voxel_index * 3);
            std::copy(V, V+3, voxel.fib_dir.begin() + 2*voxel.dim.size() * 3 + data.voxel_index * 3);
            break;
        case 1:
            V[0] = V[1] = V[2] = 0;
            std::copy(V_best, V_best+3, voxel.fib_dir.begin() + data.voxel_index * 3);
            std::copy(V, V+3, voxel.fib_dir.begin() + voxel.dim.size() * 3 + data.voxel_index * 3);
            std::copy(V, V+3, voxel.fib_dir.begin() + 2*voxel.dim.size() * 3 + data.voxel_index * 3);
            break;
        case 2:
            V[0] = V[1] = V[2] = 0;
            std::copy(V_best, V_best+3, voxel.fib_dir.begin() + data.voxel_index * 3);
            std::copy(V_best+3, V_best+6, voxel.fib_dir.begin() + voxel.dim.size() * 3 + data.voxel_index * 3);
            std::copy(V, V+3, voxel.fib_dir.begin() + 2*voxel.dim.size() * 3 + data.voxel_index * 3);
            break;
        case 3:
            V[0] = V[1] = V[2] = 0;
            std::copy(V_best, V_best+3, voxel.fib_dir.begin() + data.voxel_index * 3);
            std::copy(V_best+3, V_best+6, voxel.fib_dir.begin() + voxel.dim.size() * 3 + data.voxel_index * 3);
            std::copy(V_best+6, V_best+9, voxel.fib_dir.begin() + 2*voxel.dim.size() * 3 + data.voxel_index * 3);
            break;
        }

        // DTI
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

        delete [] x;
        delete [] xstar;
        delete [] mydata.x;
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

void flat_matrix(arma::mat &mat, std::vector<float>& v)
{
    int m, n, i, j;
    m = mat.n_rows;
    n = mat.n_cols;
    for(i=0; i<m; i++)
    {
        for(j=0; j<n; j++)
        {
            v.push_back(mat(i,j));
        }
    }
}

void cost_function(float *p, float *hx, int m, int n, void *adata)
{
    ad * data = (ad *)adata;
    float lamb = 0.0f;
    float evectors[9] = {0};
    float Dm[9] = {0};
    float Vm[9] = {0};
    float tmp1[9] = {0};
    float T[9] = {0};
    float Ds[18] = {0};
    float Diso[6] = {0};
    std::vector<float> BDi;
    std::vector<float> BDs;
    std::vector<float> expBDs;
    std::vector<float> rhs;
    std::vector<float> S;
    std::vector<float> B;
    int i = 0, j = 0;
    int N = data->n;
    unsigned int b_count = data->voxel->matinvg_dg.n_cols;

    BDi.clear();
    BDi.resize(b_count);
    BDs.clear();
    BDs.resize(b_count*N);
    expBDs.clear();
    expBDs.resize(b_count*N);
    rhs.clear();
    rhs.resize(b_count);
    S.clear();
    S.resize(b_count);
    B.clear();
    for(i=0;i<3*N;i++)
        evectors[i]=p[i+1+N];

    if(N>=1)
    {
        lamb = p[N*4+1];
        for(i=0; i<N; i++)
        {
            memset(Dm, 0, 9*sizeof(float));
            for(j=0; j<3; j++)
                Dm[3*j] = evectors[i*3+j];
            memset(Vm, 0, 9*sizeof(float));
            Vm[0] = lamb;
            memset(T, 0, 9*sizeof(float));
            image::matrix::product(Dm, Vm, tmp1, image::dyndim(3, 3), image::dyndim(3, 3));
            image::matrix::product_transpose(tmp1, Dm, T, image::dyndim(3, 3), image::dyndim(3, 3));
            Ds[0*N+i] = T[0*3+0];
            Ds[1*N+i] = T[1*3+1];
            Ds[2*N+i] = T[2*3+2];
            Ds[3*N+i] = T[0*3+1];
            Ds[4*N+i] = T[0*3+2];
            Ds[5*N+i] = T[1*3+2];
        }
        memset(Dm, 0, 9*sizeof(float));
        Dm[0] = Dm[4] = Dm[8] = 1.0f;
        memset(Vm, 0, 9*sizeof(float));
        Vm[0] = Vm[4] = Vm[8] = lamb;
        image::matrix::product(Dm, Vm, tmp1, image::dyndim(3, 3), image::dyndim(3, 3));
        image::matrix::product_transpose(tmp1, Dm, T, image::dyndim(3, 3), image::dyndim(3, 3));
        int tensor_index[6] = {0, 4, 8, 1, 2, 5};
        for(i=0; i<6; i++)
            Diso[i] = T[tensor_index[i]];
        arma::mat tmp = data->voxel->matg_dg;
        flat_matrix(tmp, B);
        image::matrix::product(B.data(), Diso, BDi.data(), image::dyndim(b_count, 6), image::dyndim(6, 1));
        image::matrix::product(B.data(), Ds, BDs.data(), image::dyndim(b_count, 6), image::dyndim(6, N));
        for(i=0; i<BDs.size(); i++)
            expBDs[i] = expf(-1*BDs[i]);
        image::matrix::product(expBDs.data(), p+1, rhs.data(), image::dyndim(b_count, N), image::dyndim(N, 1));
        for(i=0; i<BDi.size(); i++)
        {
            hx[i] = p[0]*expf(-1*BDi[i])+rhs[i];
            data->x[i] = hx[i];
        }
    }
    else
    {
        lamb = p[0];
        memset(Dm, 0, 9*sizeof(float));
        Dm[0] = Dm[4] = Dm[8] = 1.0f;
        memset(Vm, 0, 9*sizeof(float));
        Vm[0] = Vm[4] = Vm[8] = lamb;
        image::matrix::product(Dm, Vm, tmp1, image::dyndim(3, 3), image::dyndim(3, 3));
        image::matrix::product_transpose(tmp1, Dm, T, image::dyndim(3, 3), image::dyndim(3, 3));
        int tensor_index[6] = {0, 4, 8, 1, 2, 5};
        for(i=0; i<6; i++)
            Diso[i] = T[tensor_index[i]];
        arma::mat tmp = data->voxel->matg_dg;
        flat_matrix(tmp, B);
        image::matrix::product(B.data(), Diso, BDi.data(), image::dyndim(b_count, 6), image::dyndim(6, 1));
        for(i=0; i<BDi.size(); i++)
        {
            hx[i] = expf(-1*BDi[i]);
            data->x[i] = hx[i];
        }
    }
}

#endif//_PROCESS_HPP

#ifndef ICAIDSI_PROCESS_HPP
#define ICAIDSI_PROCESS_HPP
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

struct UserDataIDSI {
    int n;
    std::vector<std::vector<float>> d_gradient;
    std::vector<float> *b_gradient;
    float *x;
    ofstream *out;
};

void cost_function_idsi(float *p, float *hx, int m, int n, void *adata);

void printToFile(ofstream &out, float *data, int size, string message)
{
    int i;
    out << "------------------" << message << "------------------" << std::endl;
    for(i=0; i<size; i++)
    {
        out << data[i] << " ";
        if((i+1)%10==0)
            out << std::endl;
    }
    out << std::endl;
}

class ICAIDSI : public BaseProcess
{
private:
    int approach, numOfIC, g, initState;
    bool finetune, stabilization, PCAonly;
    double a1, a2, mu, epsilon, sampleSize;
    int maxNumIterations, maxFineTune;
    int firstEig, lastEig;
    std::vector<float> d0;
    std::vector<float> d1;
    std::vector<float> md;
    std::vector<float> num_fibers;

    // IDSI boundry variables
    float p_min0[4];
    float p_max0[4];
    float p_min1[10];
    float p_max1[10];
    float p_min2[16];
    float p_max2[16];
    float p_min3[22];
    float p_max3[22];
    float A0[4];
    float A1[10];
    float A2[16];
    float A3[22];
    float B1[1];

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
    void set_arma_col(arma::mat &m, std::vector<float> &v, int r)
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

    void setFloatValue(float *arr, float value, int size)
    {
        int i;
        for(i=0; i<size; i++)
            arr[i] = value;
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
        num_fibers.clear();
        num_fibers.resize(voxel.dim.size());

        float min_fraction = 0.05f; // minimum value of fractions
        float max_fraction = 0.95;  // maximum value of fractions
        float min_lambda = 1.0f; // minimum value of lambda
        float max_lambda = 2.0f; // maximum value of lambda
        float min_vector = -1.0f;   // minimum value of eigenvector
        float max_vector = 1.0f;    // maximum value of eigenvector

        // setFloatValue ( array to copy, the value, size of array )
        setFloatValue(p_min0, min_fraction, 4);
        setFloatValue(p_max0, max_fraction, 4);

        setFloatValue(p_min1, min_fraction, 5);
        setFloatValue(p_min1+5, min_vector, 3);
        setFloatValue(p_min1+8, min_lambda, 2);
        setFloatValue(p_max1, max_fraction, 5);
        setFloatValue(p_max1+5, max_vector, 3);
        setFloatValue(p_max1+8, max_lambda, 2);

        setFloatValue(p_min2, min_fraction, 6);
        setFloatValue(p_min2+6, min_vector, 6);
        setFloatValue(p_min2+12, min_lambda, 4);
        setFloatValue(p_max2, max_fraction, 6);
        setFloatValue(p_max2+6, max_vector, 6);
        setFloatValue(p_max2+12, max_lambda, 4);

        setFloatValue(p_min3, min_fraction, 7);
        setFloatValue(p_min3+7, min_vector, 9);
        setFloatValue(p_min3+16, min_lambda, 6);
        setFloatValue(p_max3, max_fraction, 7);
        setFloatValue(p_max3+7, max_vector, 9);
        setFloatValue(p_max3+16, max_lambda, 6);

        // first initialize with zero
        memset(A0, 0, 4 * sizeof(float));
        memset(A1, 0, 10 * sizeof(float));
        memset(A2, 0, 16 * sizeof(float));
        memset(A3, 0, 22 * sizeof(float));

        for(m=0; m<4; m++)
            A0[m] = 1.0f;
        for(m=0; m<5; m++)
            A1[m] = 1.0f;
        for(m=0; m<6; m++)
            A2[m] = 1.0f;
        for(m=0; m<7; m++)
            A3[m] = 1.0f;

        B1[0] = 1.0f;

        approach = FICA_APPROACH_SYMM;
        g = FICA_NONLIN_TANH;
        initState = FICA_INIT_RAND;
        finetune = false;
        stabilization = false;
        PCAonly = false;
        a1 = 1;
        a2 = 1;
        mu = 1;
        numOfIC = 3;
        epsilon = 0.0001f;
        sampleSize = 1;
        maxNumIterations = 1000;
        maxFineTune = 5;
        firstEig = 1;
        lastEig = 3;

        voxel.fr.clear();
        voxel.fr.resize(numOfIC* voxel.dim.size());
        voxel.ia.clear();
        voxel.ia.resize(4 * voxel.dim.size());
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
        arma::mat bvalue(b_count,1);
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
        ofstream out;
        out.open("output.txt", ios::app);

        bool threeDim = false;
        int center_voxel = 4;

        if(threeDim)
            center_voxel = 9;

        int pi, pj;
        int ica_num = 0;
        //int stop_count = 0;
        //int stop_flag = 0;
        itpp::mat mixedSig, icasig;
        itpp::mat icasig_no_log;
        itpp::mat mixing_matrix;

        if(threeDim)
            mixedSig.set_size(19, b_count);
        else
            mixedSig.set_size(9, b_count);
        mixedSig.zeros();
        // ICA
        unsigned int i=0,j=0,k=0,m=0,n=0;
        /* get 3d position */
        i = data.voxel_index;
        k = (unsigned int) i / (voxel.dim.h * voxel.dim.w);
        i -= k * voxel.dim.h * voxel.dim.w;

        j = (unsigned int) i / voxel.dim.w;
        i -= j * voxel.dim.w;

        out << "====================================================" << std::endl;
        out << "i= " << i << ", j= " << j << ", k= " << k << std::endl;

        if(threeDim)
        {
            // 5 voxels behind current voxel
            if(k > 0 && j > 0)
                set_mat_row(mixedSig, voxel.signalData[data.voxel_index-voxel.dim.w*voxel.dim.h-voxel.dim.w], 0);
            if(k > 0 && i > 0)
                set_mat_row(mixedSig, voxel.signalData[data.voxel_index-voxel.dim.w*voxel.dim.h-1], 1);
            if(k > 0)
                set_mat_row(mixedSig, voxel.signalData[data.voxel_index-voxel.dim.w*voxel.dim.h], 2);
            if(k > 0 && i < voxel.dim.w - 1)
                set_mat_row(mixedSig, voxel.signalData[data.voxel_index-voxel.dim.w*voxel.dim.h+1], 3);
            if(k > 0 && j < voxel.dim.h - 1)
                set_mat_row(mixedSig, voxel.signalData[data.voxel_index-voxel.dim.w*voxel.dim.h+voxel.dim.w], 4);

            // current flat 9 voxels
            if(i > 0 && j > 0)
                set_mat_row(mixedSig, voxel.signalData[data.voxel_index-voxel.dim.w-1], 5);
            if(j > 0)
                set_mat_row(mixedSig, voxel.signalData[data.voxel_index-voxel.dim.w], 6);
            if(i < voxel.dim.w - 1 && j > 0 )
                set_mat_row(mixedSig, voxel.signalData[data.voxel_index-voxel.dim.w+1], 7);
            if(i > 0)
                set_mat_row(mixedSig, voxel.signalData[data.voxel_index-1], 8);
            set_mat_row(mixedSig, voxel.signalData[data.voxel_index], 9);
            if(i < voxel.dim.w - 1)
                set_mat_row(mixedSig, voxel.signalData[data.voxel_index+1], 10);
            if(i > 0 && j < voxel.dim.h - 1)
                set_mat_row(mixedSig, voxel.signalData[data.voxel_index+voxel.dim.w-1], 11);
            if(j < voxel.dim.h - 1 )
                set_mat_row(mixedSig, voxel.signalData[data.voxel_index+voxel.dim.w], 12);
            if(i < voxel.dim.w - 1 && j < voxel.dim.h - 1)
                set_mat_row(mixedSig, voxel.signalData[data.voxel_index+voxel.dim.w+1], 13);

            // 5 voxels in front of current voxel
            if(k < voxel.dim.d - 1 && j > 0)
                set_mat_row(mixedSig, voxel.signalData[data.voxel_index+voxel.dim.w*voxel.dim.h-voxel.dim.w], 14);
            if(k < voxel.dim.d - 1 && i > 0)
                set_mat_row(mixedSig, voxel.signalData[data.voxel_index+voxel.dim.w*voxel.dim.h-1], 15);
            if(k < voxel.dim.d - 1)
                set_mat_row(mixedSig, voxel.signalData[data.voxel_index+voxel.dim.w*voxel.dim.h], 16);
            if(k < voxel.dim.d - 1 && i < voxel.dim.w - 1)
                set_mat_row(mixedSig, voxel.signalData[data.voxel_index+voxel.dim.w*voxel.dim.h+1], 17);
            if(k < voxel.dim.d - 1 && j < voxel.dim.h - 1)
                set_mat_row(mixedSig, voxel.signalData[data.voxel_index+voxel.dim.w*voxel.dim.h+voxel.dim.w], 18);
        }
        else
        {
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
        }

        // BSM variables
        float par[22];
        float lambda = 0.0015;
        float info[LM_INFO_SZ];
        int b_count = voxel.bvalues.size()-1;
        int min_index = -1;
        float opts[LM_OPTS_SZ];
        opts[0] = 1E-6;    // mu
        opts[1] = 1E-15;
        opts[2] = 1E-15;   // |dp|^2
        opts[3] = 1E-15;   // |e|^2
        opts[4] = 1E-15;   // delta, step used in difference approximation to the Jacobian
        float *x = new float[b_count];
        float *weight0 = new float[b_count];
        float *weight1 = new float[b_count];
        float *weight2 = new float[b_count];
        float *weight3 = new float[b_count];
        UserDataIDSI mydata;
        mydata.x = new float[b_count];
        mydata.d_gradient.clear();
        mydata.out = &out;
        for(m=0; m<b_count; m++)
        {
            image::vector<3> tmp = voxel.bvectors[m+1];
            std::vector<float> vtmp;
            vtmp.push_back(tmp[0]);
            vtmp.push_back(tmp[1]);
            vtmp.push_back(tmp[2]);
            mydata.d_gradient.push_back(vtmp);
        }
        mydata.b_gradient = &voxel.bvalues;
        float e[4] = {0};
        float SC[4] = {0};
        float SC_min = 1E+15;
        float V_best[9];
        float F_best[3];
        float IA_best[4];
        int result;
        int maxIteration;
        int niter = 20;

        // ICA variables
        arma::mat tensor_param(6,1);
        double tensor[9];
        double V[9],d[3];
        std::vector<float> signal(b_count);
        std::vector<float> w(3);

        for(ica_num = 0; ica_num <=3; ica_num++)
        {
            float eigenvectors[9]; // BSM
            float fractions[4]; // BSM
            numOfIC = ica_num;

            if(ica_num!=0) // We do not run ICA for zero sticks, we run only BSM.
            {
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

            } // if (ica_num!=0)

            for(n=0; n<b_count; n++)
            {
                x[n] = mixedSig(center_voxel, n);
                mydata.x[n] = mixedSig(center_voxel, n);
                weight0[n]= (b_count-4+1)/(x[n]*x[n]);
                weight1[n]= (b_count-10+1)/(x[n]*x[n]);
                weight2[n]= (b_count-16+1)/(x[n]*x[n]);
                weight3[n]= (b_count-22+1)/(x[n]*x[n]);

            }
            switch(ica_num)
            {
            case 0: // zero source
                for(n=0; n<4; n++)
                    par[n] = 0.25f; // divide 1 between four isotropic fractions
                mydata.n = 0;
                maxIteration = niter * 4;
                result = slevmar_blec_dif(&cost_function_idsi, par,
                                          x, 4, b_count, p_min0, p_max0, A0, B1,
                                          1, weight0, maxIteration, opts, info, NULL,
                                          NULL, (void*)&mydata);
                //printToFile(out, par, 4, "Parameters");
                e[ica_num] = info[1];
                SC[ica_num] = logf(e[ica_num]/b_count)+1*logf(b_count)/b_count;
                //printToFile(out, x, b_count, "X(0)");
                //printToFile(out, mydata.x, b_count, "New X(0)");
                if(SC_min > SC[ica_num])
                {
                    SC_min = SC[ica_num];
                    min_index = ica_num;
                    for(n=0; n<4; n++)
                        IA_best[n] = par[n];
                }
                break;
            case 1: // one ica source
                for(m=0; m<5; m++)
                    par[m] = 1.0f/5;
                for(m=0; m<3; m++)
                {
                    par[m+5] = eigenvectors[m];
                    p_min1[m+5] = std::min<float>(par[m+5]*0.9f,par[m+5]*1.1f);
                    p_max1[m+5] = std::max<float>(par[m+5]*0.9f,par[m+5]*1.1f);
                }
                for(m=8; m<10; m++)
                    par[m] = lambda;
                mydata.n = 1;
                maxIteration = niter * 10;
                result = slevmar_blec_dif(&cost_function_idsi, par,
                                          x, 10, b_count, p_min1, p_max1, A1, B1,
                                          1, weight1, maxIteration, opts, info, NULL,
                                          NULL, (void*)&mydata);
                //printToFile(out, par, 10, "Parameters");
                e[ica_num] = info[1];
                SC[ica_num] = logf(e[ica_num]/b_count)+1*logf(b_count)/b_count;
                //printToFile(out, x, b_count, "X(1)");
                //printToFile(out, mydata.x, b_count, "New X(1)");
                if(SC_min > SC[ica_num])
                {
                    SC_min = SC[ica_num];
                    min_index = ica_num;
                    F_best[0] = std::abs<float>(par[0]);
                    for(n=0; n<4; n++)
                        IA_best[n] = par[n+1];
                    for(n=0; n<3; n++)
                        V_best[n]=par[n+5];
                }
                break;
            case 2: // two ica sources
                for(m=0; m<6; m++)
                    par[m] = 1.0f/6;
                for(m=0; m<6; m++)
                {
                    par[m+6] = eigenvectors[m];
                    p_min2[m+6] = std::min<float>(par[m+6]*0.9f,par[m+6]*1.1f);
                    p_max2[m+6] = std::max<float>(par[m+6]*0.9f,par[m+6]*1.1f);
                }
                for(m=12; m<16; m++)
                    par[m] = lambda;
                mydata.n = 2;
                maxIteration = niter * 16;
                printToFile(out, p_min2, 16, "p_min2");
                printToFile(out, p_max2, 16, "p_max2");
                result = slevmar_blec_dif(&cost_function_idsi, par,
                                          x, 16, b_count, p_min2, p_max2, A2, B1,
                                          1, weight2, maxIteration, opts, info, NULL,
                                          NULL, (void*)&mydata);
                //printToFile(out, par, 16, "Parameters");
                e[ica_num] = info[1];
                SC[ica_num] = logf(e[ica_num]/b_count)+1*logf(b_count)/b_count;
                //printToFile(out, x, b_count, "X(2)");
                //printToFile(out, mydata.x, b_count, "New X(2)");
                if(SC_min > SC[ica_num])
                {
                    SC_min = SC[ica_num];
                    min_index = ica_num;
                    F_best[0] = std::abs<float>(par[0]);
                    F_best[1] = std::abs<float>(par[1]);
                    for(n=0; n<4; n++)
                        IA_best[n] = par[n+2];
                    for(n=0; n<6; n++)
                        V_best[n] = par[n+6];
                }
                break;
            case 3: // three ica sources
                for(m=0; m<7; m++)
                    par[m] = 1.0f/7;
                for(m=0; m<9; m++)
                {
                    par[m+7] = eigenvectors[m];
                    p_min3[m+7] = std::min<float>(par[m+7]*0.9f,par[m+7]*1.1f);
                    p_max3[m+7] = std::max<float>(par[m+7]*0.9f,par[m+7]*1.1f);
                }
                for(m=16; m<22; m++)
                    par[m] = lambda;
                mydata.n = 3;
                maxIteration = niter * 22;
                result = slevmar_blec_dif(&cost_function_idsi, par,
                                          x, 22, b_count, p_min3, p_max3, A3, B1,
                                          1, weight3, maxIteration, opts, info, NULL,
                                          NULL, (void*)&mydata);
                e[ica_num] = info[1];
                SC[ica_num] = logf(e[ica_num]/b_count)+1*logf(b_count)/b_count;
                //printToFile(out, x, b_count, "X(3)");
                //printToFile(out, mydata.x, b_count, "New X(3)");
                if(SC_min > SC[ica_num])
                {
                    SC_min = SC[ica_num];
                    min_index = ica_num;
                    F_best[0] = std::abs<float>(par[1]);
                    F_best[1] = std::abs<float>(par[2]);
                    F_best[2] = std::abs<float>(par[3]);
                    for(n=0; n<4; n++)
                        IA_best[n] = par[n+3];
                    for(n=0; n<9; n++)
                        V_best[n] = par[n+7];
                }
                break;
            }
        }
        switch(min_index)
        {
        case 0:
            V[0] = V[1] = V[2] = 0;
            std::copy(V, V+3, voxel.fib_dir.begin() + data.voxel_index * 3);
            std::copy(V, V+3, voxel.fib_dir.begin() + voxel.dim.size() * 3 + data.voxel_index * 3);
            std::copy(V, V+3, voxel.fib_dir.begin() + 2*voxel.dim.size() * 3 + data.voxel_index * 3);
            voxel.fr[data.voxel_index] = 0;
            voxel.fr[voxel.dim.size()+data.voxel_index] = 0;
            voxel.fr[2*voxel.dim.size()+data.voxel_index] = 0;
            voxel.ia[data.voxel_index] = IA_best[0];
            voxel.ia[voxel.dim.size()+data.voxel_index] = IA_best[1];
            voxel.ia[2*voxel.dim.size()+data.voxel_index] = IA_best[2];
            voxel.ia[3*voxel.dim.size()+data.voxel_index] = IA_best[3];
            break;
        case 1:
            V[0] = V[1] = V[2] = 0;
            std::copy(V_best, V_best+3, voxel.fib_dir.begin() + data.voxel_index * 3);
            std::copy(V, V+3, voxel.fib_dir.begin() + voxel.dim.size() * 3 + data.voxel_index * 3);
            std::copy(V, V+3, voxel.fib_dir.begin() + 2*voxel.dim.size() * 3 + data.voxel_index * 3);
            voxel.fr[data.voxel_index] = F_best[0];
            voxel.fr[voxel.dim.size()+data.voxel_index] = 0;
            voxel.fr[2*voxel.dim.size()+data.voxel_index] = 0;
            voxel.ia[data.voxel_index] = IA_best[0];
            voxel.ia[voxel.dim.size()+data.voxel_index] = IA_best[1];
            voxel.ia[2*voxel.dim.size()+data.voxel_index] = IA_best[2];
            voxel.ia[3*voxel.dim.size()+data.voxel_index] = IA_best[3];
            break;
        case 2:
            V[0] = V[1] = V[2] = 0;
            std::copy(V_best, V_best+3, voxel.fib_dir.begin() + data.voxel_index * 3);
            std::copy(V_best+3, V_best+6, voxel.fib_dir.begin() + voxel.dim.size() * 3 + data.voxel_index * 3);
            std::copy(V, V+3, voxel.fib_dir.begin() + 2*voxel.dim.size() * 3 + data.voxel_index * 3);
            voxel.fr[data.voxel_index] = F_best[0];
            voxel.fr[voxel.dim.size()+data.voxel_index] = F_best[1];
            voxel.fr[2*voxel.dim.size()+data.voxel_index] = 0;
            voxel.ia[data.voxel_index] = IA_best[0];
            voxel.ia[voxel.dim.size()+data.voxel_index] = IA_best[1];
            voxel.ia[2*voxel.dim.size()+data.voxel_index] = IA_best[2];
            voxel.ia[3*voxel.dim.size()+data.voxel_index] = IA_best[3];
            break;
        case 3:
            std::copy(V_best, V_best+3, voxel.fib_dir.begin() + data.voxel_index * 3);
            std::copy(V_best+3, V_best+6, voxel.fib_dir.begin() + voxel.dim.size() * 3 + data.voxel_index * 3);
            std::copy(V_best+6, V_best+9, voxel.fib_dir.begin() + 2*voxel.dim.size() * 3 + data.voxel_index * 3);
            voxel.fr[data.voxel_index] = F_best[0];
            voxel.fr[voxel.dim.size()+data.voxel_index] = F_best[1];
            voxel.fr[2*voxel.dim.size()+data.voxel_index] = F_best[2];
            voxel.ia[data.voxel_index] = IA_best[0];
            voxel.ia[voxel.dim.size()+data.voxel_index] = IA_best[1];
            voxel.ia[2*voxel.dim.size()+data.voxel_index] = IA_best[2];
            voxel.ia[3*voxel.dim.size()+data.voxel_index] = IA_best[3];
            break;
        }
        num_fibers[data.voxel_index] = min_index;

        // DTI
        if (data.space.front() != 0.0)
        {
            float logs0 = std::log(std::max<float>(1.0,data.space.front()));
            for (unsigned int i = 1; i < data.space.size(); ++i)
                signal[i-1] = std::max<float>(0.0,logs0-std::log(std::max<float>(1.0,data.space[i])));
        }

        arma::mat matsignal(b_count,1);
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
        delete [] mydata.x;
        delete [] weight0;
        delete [] weight1;
        delete [] weight2;
        delete [] weight3;
        out.close();

    }
    virtual void end(Voxel& voxel,gz_mat_write& mat_writer)
    {
        mat_writer.write("dir0",&*voxel.fib_dir.begin(), 1, voxel.fib_dir.size()/3);
        mat_writer.write("dir1",&*voxel.fib_dir.begin()+voxel.dim.size()*3, 1, voxel.fib_dir.size()/3);
        mat_writer.write("dir2",&*voxel.fib_dir.begin()+2*voxel.dim.size()*3, 1, voxel.fib_dir.size()/3);
        mat_writer.write("fa0",&*voxel.fr.begin(), 1, voxel.fr.size()/3);
        mat_writer.write("fa1",&*voxel.fr.begin()+voxel.dim.size(), 1, voxel.fr.size()/3);
        mat_writer.write("fa2",&*voxel.fr.begin()+2*voxel.dim.size(), 1, voxel.fr.size()/3);
        mat_writer.write("ia0",&*voxel.ia.begin(), 1, voxel.ia.size()/4);
        mat_writer.write("ia1",&*voxel.ia.begin()+voxel.dim.size(), 1, voxel.ia.size()/4);
        mat_writer.write("ia2",&*voxel.ia.begin()+2*voxel.dim.size(), 1, voxel.ia.size()/4);
        mat_writer.write("ia3",&*voxel.ia.begin()+3*voxel.dim.size(), 1, voxel.ia.size()/4);
        mat_writer.write("gfa",&*voxel.fib_fa.begin(), 1, voxel.fib_fa.size());
        mat_writer.write("adc",&*md.begin(),1,md.size());
        mat_writer.write("axial_dif",&*d0.begin(),1,d0.size());
        mat_writer.write("radial_dif",&*d1.begin(),1,d1.size());
        mat_writer.write("numfiber", &*num_fibers.begin(), 1, num_fibers.size());
    }
};

void cost_function_idsi(float *p, float *hx, int m, int n, void *adata)
{
    // define variables
    int i, j;
    int l;
    UserDataIDSI * data = (UserDataIDSI *) adata;
    int N = data->n;
    int L = 4;
    float bin_start = 0.0f;
    float bin_end = 3.0E-3f;
    arma::mat theta(1, n);
    arma::mat d_gradients(3,n);
    arma::mat eigenvectors(3, 3);
    arma::mat lambdas(3, 2);
    arma::mat filledwithones(1, n, arma::fill::ones);
    arma::mat D(L, 1);
    arma::mat s_aniso;
    arma::mat s_iso(L, n);
    arma::mat b_gradients(1, n);
    arma::mat iso_part(1, n, arma::fill::zeros);
    arma::mat anis_part(1, n, arma::fill::zeros);
    arma::mat f_iso(1, L);
    arma::mat f_aniso;
    if(N!=0)
        s_aniso.set_size(N, n);
    if(N!=0)
        f_aniso.set_size(1, N);

    //printToFile(*data->out, p, 22, "P");

    // assign values
    for(i=0; i<n; i++)
        b_gradients(0, i) = (*data->b_gradient)[i+1];
    for(i=0; i<L; i++)
        D(i, 0) = i*((bin_end-bin_start)/(L-1));
    //*data->out << "=======d-gradients==========" << std:: endl;
    for(i=0; i<3; i++)
    {
        for(j=0; j<n; j++)
        {
            d_gradients(i,j) = data->d_gradient[j][i];
            //*data->out << d_gradients(i, j) << " ";
        }
        //*data->out << std::endl;
    }

    switch(N)
    {
    case 0:
        for(i=0; i<L; i++)
            f_iso(0, i) = p[i];
        break;
    case 1:
        for(i=0; i<3; i++)
            eigenvectors(0, i) = p[i+5];
        lambdas(0, 0) = abs(p[8]/1000);
        lambdas(0, 1) = abs(p[9]/1000);
        for(i=0; i<L; i++)
            f_iso(0, i) = p[i+1];
        f_aniso(0, 0) = p[0];
        break;
    case 2:
        for(i=0; i<3; i++)
            eigenvectors(0, i) = p[i+6];
        for(i=0; i<3; i++)
            eigenvectors(1, i) = p[i+9];
        lambdas(0, 0) = abs(p[12]/1000);
        lambdas(0, 1) = abs(p[14]/1000);
        lambdas(1, 0) = abs(p[13]/1000);
        lambdas(1, 1) = abs(p[15]/1000);
        for(i=0; i<L; i++)
            f_iso(0, i) = p[i+2];
        f_aniso(0, 0) = p[0];
        f_aniso(0, 1) = p[1];
        break;
    case 3:
        for(i=0; i<3; i++)
            eigenvectors(0, i) = p[i+7];
        for(i=0; i<3; i++)
            eigenvectors(1, i) = p[i+10];
        for(i=0; i<3; i++)
            eigenvectors(2, i) = p[i+13];
        lambdas(0, 0) = abs(p[16]/1000);
        lambdas(0, 1) = abs(p[19]/1000);
        lambdas(1, 0) = abs(p[17]/1000);
        lambdas(1, 1) = abs(p[20]/1000);
        lambdas(2, 0) = abs(p[18]/1000);
        lambdas(2, 1) = abs(p[21]/1000);
        for(i=0; i<L; i++)
            f_iso(0, i) = p[i+3];
        f_aniso(0, 0) = p[0];
        f_aniso(0, 1) = p[1];
        f_aniso(0, 2) = p[2];
        break;
    }

    for(l=0; l<N; l++) // main loop
    {
        arma::mat eigenrow(eigenvectors.row(l)); // eigenvectors = Nx3, eigenrow = 1x3
        eigenrow = eigenrow.t(); // eigenrow = 3x1
        arma::mat scaledeigenrow = eigenrow * filledwithones; // filledwithones = 1xn (e.g. n=64), scaledeigenrow = 3xn
        arma::mat scaledtimesgradient(1,n); // d_gradients = 3xn, scaledtimesgradient = 1xn
        for(i=0; i<scaledeigenrow.n_cols; i++)
            scaledtimesgradient(0, i) = scaledeigenrow(0, i) * d_gradients(0,i) +
                    scaledeigenrow(1, i) * d_gradients(1, i) +
                    scaledeigenrow(2, i) * d_gradients(2, i);
        theta = arma::acos<arma::mat>(scaledtimesgradient).t();
        for(i=0; i<n; i++)
        {
            s_aniso(l, i) = expf(-1*b_gradients(0, i)*lambdas(l, 1)) *
                    expf(-1*b_gradients(0, i) * (lambdas(l, 0) - lambdas(l, 1)) *
                         cos(theta(i,0)) * cos(theta(i,0)));
        }
    } // main loop
    s_iso = arma::exp(-1 * (D * b_gradients));
    for(i=0; i<L; i++)
    {
        for(j=0; j<n ; j++)
        {
            iso_part(0, j) += f_iso(0, i) * s_iso(i, j);
        }
    }
    if(N==0)
    {
        for(i=0; i<n; i++)
        {
            hx[i] = iso_part(0, i);
            data->x[i] = hx[i];
        }
    }
    else
    {
        anis_part = f_aniso * s_aniso;
        for(i=0; i<n; i++)
        {
            hx[i] = anis_part(0, i) + iso_part(0, i);
            data->x[i] = hx[i];
        }
    }
    printToFile(*data->out, data->x, n, "difference in x-value");
}

#endif//_PROCESS_HPP

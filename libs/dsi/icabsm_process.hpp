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

struct UserData {
    int n;
    arma::mat matd_g;
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
    std::vector<float> d0;
    std::vector<float> d1;
    std::vector<float> md;
    std::vector<float> num_fibers;

    // BSM variables
    float p_min0[2];
    float p_max0[2];
    float p_min1[6];
    float p_max1[6];
    float p_min2[10];
    float p_max2[10];
    float p_min3[14];
    float p_max3[14];
    float A0[2];
    float A1[6];
    float A2[10];
    float A3[14];
    float B1[1];
    float stick_d_min;
    float stick_d_max;
    float ball_d_min;
    float ball_d_max;


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

        stick_d_min = 0.10f; stick_d_max = 0.20f;
        ball_d_min  = 0.05f; ball_d_max  = 0.30f;
        p_min0[0] = 0.05f; p_min0[1] = ball_d_min;
        p_max0[0] = 0.95f; p_max0[1] = ball_d_max;
        p_min1[0] = 0.05f; p_min1[1] = 0.05f; p_min1[2] = -1.0f;  p_min1[3] = -1.0f; p_min1[4] = -1.0f; p_min1[5] = stick_d_min;   //0.0010f
        p_max1[0] = 0.95f; p_max1[1] = 0.95f; p_max1[2] = 1.0f;   p_max1[3] = 1.0f;  p_max1[4] = 1.0f;  p_max1[5] = stick_d_max;   //0.0020f
        p_min2[0] = 0.05f; p_min2[1] = 0.05f; p_min2[2] = 0.05f;  p_min2[3] = -1.0f; p_min2[4] = -1.0f; p_min2[5] = -1.0f; p_min2[6] = -1.0f; p_min2[7] = -1.0f; p_min2[8] = -1.0f; p_min2[9] = stick_d_min;
        p_max2[0] = 0.95f; p_max2[1] = 0.95f; p_max2[2] = 0.95f;  p_max2[3] = 1.0f;  p_max2[4] = 1.0f;  p_max2[5] = 1.0f;  p_max2[6] = 1.0f;  p_max2[7] = 1.0f;  p_max2[8] = 1.0f;  p_max2[9] = stick_d_max;
        p_min3[0] = 0.05f; p_min3[1] = 0.05f; p_min3[2] = 0.05f;  p_min3[3] = 0.05f; p_min3[4] = -1.0f; p_min3[5] = -1.0f; p_min3[6] = -1.0f; p_min3[7] = -1.0f; p_min3[8] = -1.0f; p_min3[9] = -1.0f; p_min3[10] = -1.0f; p_min3[11] = -1.0f; p_min3[12] = -1.0f; p_min3[13] = stick_d_min;
        p_max3[0] = 0.95f; p_max3[1] = 0.95f; p_max3[2] = 0.95f;  p_max3[3] = 0.95f; p_max3[4] = 1.0f;  p_max3[5] = 1.0f;  p_max3[6] = 1.0f;  p_max3[7] = 1.0f;  p_max3[8] = 1.0f;  p_max3[9] = 1.0f;  p_max3[10] = 1.0f;  p_max3[11] = 1.0f;  p_max3[12] = 1.0f;  p_max3[13] = stick_d_max;

        A0[0] = 1.0f; A0[1] = 0.0f;
        A1[0] = 1.0f; A1[1] = 1.0f; A1[2] = 0.0f; A1[3] = 0.0f; A1[4] = 0.0f; A1[5] = 0.0f;
        A2[0] = 1.0f; A2[1] = 1.0f; A2[2] = 1.0f; A2[3] = 0.0f; A2[4] = 0.0f; A2[5] = 0.0f; A2[6] = 0.0f; A2[7] = 0.0f; A2[8] = 0.0f; A2[9] = 0.0f;
        A3[0] = 1.0f; A3[1] = 1.0f; A3[2] = 1.0f; A3[3] = 1.0f; A3[4] = 0.0f; A3[5] = 0.0f; A3[6] = 0.0f; A3[7] = 0.0f; A3[8] = 0.0f; A3[9] = 0.0f; A3[10] =0.0f; A3[11] = 0.0f; A3[12] = 0.0f; A3[13] = 0.0f;

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
        //ofstream out;
        //out.open("output.txt", ios::app);

        bool threeDim = true;
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
        float par[14];
        float lambda = 0.15f;
        float info[LM_INFO_SZ];
        int b_count = voxel.bvalues.size()-1;
        int min_index = -1;
        float opts[LM_OPTS_SZ];
        opts[0] = 1E-3;   // mu
        opts[1] = 1E-8;
        opts[2] = 1E-8;  // |dp|^2
        opts[3] = 1E-8;  // |e|^2
        opts[4] = 1E-8;  //  delta, step used in difference approximation to the Jacobian
        float *x = new float[b_count];
        float *weight0 = new float[b_count];
        float *weight1 = new float[b_count];
        float *weight2 = new float[b_count];
        float *weight3 = new float[b_count];
        float *xstar = new float[b_count];
        UserData mydata;
        mydata.x = new float[b_count];
        mydata.matd_g = voxel.matg_dg;
        float e[4]={0};
        float SC[4] = {0};
        float SC_min=1E+15;
        float V_best[9];
        float F_best[3];
        float sum;
        int result;
        int maxIteration;
        int niter = 20;

        // ICA variables
        arma::mat tensor_param(6,1);
        double tensor[9];
        double V[9],d[3];
        std::vector<float> signal(b_count);
        std::vector<float> w(3);

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
        md[data.voxel_index] = (d[0]+d[1]+d[2])/3.0;
        d0[data.voxel_index] =  d[0];
        d1[data.voxel_index] = (d[1]+d[2])/2.0;


        if (data.fa[0] < 0.1)
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

                //printToFile(out, partmp0, 2, "Par 0");
                //printToFile(out, xtmp, b_count, "X");

                result = slevmar_blec_dif(&cost_function, par, x, 2, b_count, p_min0, p_max0, A0, B1, 1, weight0, maxIteration, opts, info, NULL, NULL, (void*)&mydata);
                //result = slevmar_bc_dif(&cost_function, par, x, 2, b_count, p_min0, p_max0, NULL, maxIteration, opts, info, NULL, NULL, (void*)&mydata);
                //printToFile(out, partmp0, 2, "Par-Hat");
                //printToFile(out, xtmp, b_count, "X-Hat");
                //printToFile(out, info, 10, "info");
                //printToFile(out, SC, 1, "SC");

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

            //float xtmp[64] = {0.5374,0.3333,0.2517,0.1633,0.3197,0.4082,0.5374,0.3197,0.3265,0.3333,0.2925,0.3129,0.6463,0.5646,0.3129,0.5374,0.5578,0.6599,0.6054,0.4898,0.4898,0.7687,0.8844,0.5646,0.6667,0.7891,0.6463,0.5442,0.5986,0.8776,0.7211,0.6599,0.7415,0.6531,0.6327,0.6939,0.8027,0.5442,0.3469,0.6871,0.8639,0.5986,0.6395,0.4762,0.7075,0.6463,0.6939,0.4422,0.3197,0.6327,0.4830,0.4082,0.6395,0.2993,0.2993,0.3061,0.2857,0.4830,0.4830,0.3401,0.3946,0.3673,0.1293,0.1701};
            //float partmp0[2] =  {0.8445,    0.0015};
            //float partmp1[6] =  {0.5334,    0.4666,   -0.9854,   -0.1233,    0.1170,    0.0015};
            //float partmp2[10] = {0.3424,    0.3343,    0.3234,   -0.9279,    0.2011,   -0.3140,   -0.9716,   -0.1487,    0.1840,    0.0015};
            //float partmp3[14] = {0.2505,    0.2523,    0.2543,    0.2429,   -0.6732,   -0.1339,    0.7272,   -0.9833,   -0.1618,    0.0838, -0.4323,    0.8999,    0.0572,    0.0015};

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


                //fractions[1] = 0.5f; //fractions[1] *= 0.95;
                //fractions[0] = 1-fractions[1];
                //par[0] = fractions[0];
                //par[1] = fractions[1];

                for(n=0; n<3; n++)
                {    par[n+2] = eigenvectors[n];
                     p_min1[n+2] = std::min<float>(par[n+2]*0.9f,par[n+2]*1.1f);
                     p_max1[n+2] = std::max<float>(par[n+2]*0.9f,par[n+2]*1.1f);
                }

                par[5] = lambda + (((float) rand()/(RAND_MAX))-0.5f)/10;

                mydata.n = 1; maxIteration = niter*6;

                //printToFile(out, partmp1, 6, "Par 1");
                //printToFile(out, xtmp, b_count, "X");


                result = slevmar_blec_dif(&cost_function, par, x, 6, b_count, p_min1, p_max1, A1, B1, 1, weight1, maxIteration, opts, info, NULL, NULL, (void*)&mydata);
                //result = slevmar_bc_dif(&cost_function, par, x, 6, b_count, p_min1, p_max1, NULL, maxIteration, opts, info, NULL, NULL, (void*)&mydata);
                //printToFile(out, partmp1, 6, "Par-Hat");
                //printToFile(out, xtmp, b_count, "X-Hat");
                //printToFile(out, info, 10, "info");
                //printToFile(out, SC+1, 1, "SC");

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

                //fractions[1] = 0.33f; //fractions[1] *= 0.95;
                //fractions[2] = 0.33f; //fractions[2] *= 0.95;
                //fractions[0] = 1-fractions[1]-fractions[2];

                //for(n=0; n<3; n++) // copy fractions to par.
                //    par[n] = fractions[n];

                for(n=0; n<6; n++)
                {    par[n+3] = eigenvectors[n];
                    p_min3[n+3] = std::min<float>(par[n+3]*0.9f,par[n+3]*1.1f);
                    p_max3[n+3] = std::max<float>(par[n+3]*0.9f,par[n+3]*1.1f);
                }

                par[9] = lambda  + (((float) rand()/(RAND_MAX))-0.5f)/10;

                mydata.n = 2; maxIteration = niter*10;

                //printToFile(out, partmp2, 10, "Par 2");
                //printToFile(out, xtmp, b_count, "X");

                result = slevmar_blec_dif(&cost_function, par, x, 10, b_count, p_min2, p_max2, A2, B1, 1, weight2, maxIteration, opts, info, NULL, NULL, (void*)&mydata);
                //result = slevmar_bc_dif(&cost_function, par, x, 10, b_count, p_min2, p_max2, NULL, maxIteration, opts, info, NULL, NULL, (void*)&mydata);
                //printToFile(out, partmp2, 10, "Par-Hat");
                //printToFile(out, xtmp, b_count, "X-Hat");
                //printToFile(out, info, 10, "info");
                //printToFile(out, SC+2, 1, "SC");

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
            case 3: // three sticks
                //fractions[1] = 0.25f; //fractions[1] *= 0.95;
                //fractions[2] = 0.25f; //fractions[2] *= 0.95;
                //fractions[3] = 0.25f; //fractions[3] *= 0.95;
                //fractions[0] = 1-fractions[1]-fractions[2]-fractions[3];

                //for(n=0; n<4; n++) // copy fractions to par.
                //    par[n] = fractions[n];

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

                //printToFile(out, partmp3, 14, "Par 3");
                //printToFile(out, xtmp, b_count, "X");

                result = slevmar_blec_dif(&cost_function, par, x, 14, b_count, p_min3, p_max3, A3, B1, 1, weight3, maxIteration, opts, info, NULL, NULL, (void*)&mydata);
                //result = slevmar_bc_dif(&cost_function, par, x, 14, b_count, p_min3, p_max3, NULL, maxIteration, opts, info, NULL, NULL, (void*)&mydata);
                //printToFile(out, partmp3, 14, "Par-Hat");
                //printToFile(out, xtmp, b_count, "X-Hat");
                //printToFile(out, info, 10, "info");
                //printToFile(out, SC+3, 1, "SC");

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
      } // if FA <0.1
        /* DTI
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
        d1[data.voxel_index] = 1000.0*(d[1]+d[2])/2.0; */

        //out.close();
        delete [] weight0;
        delete [] weight1;
        delete [] weight2;
        delete [] weight3;
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
        mat_writer.write("numfiber", &*num_fibers.begin(), 1, num_fibers.size());
    }
};

void flat_matrix(arma::mat &mat, std::vector<float>& v)
{
    int m, n, i, j;
    m = mat.n_rows;
    n = mat.n_cols;
    v.clear();
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
    UserData * data = (UserData *)adata;
    arma::mat matd_g = data->matd_g;
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
    unsigned int b_count = matd_g.n_rows;

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
        lamb = p[N*4+1]/100;
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
        flat_matrix(matd_g, B);
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
        lamb = p[1]/100;
        memset(Dm, 0, 9*sizeof(float));
        Dm[0] = Dm[4] = Dm[8] = 1.0f;
        memset(Vm, 0, 9*sizeof(float));
        Vm[0] = Vm[4] = Vm[8] = lamb;
        image::matrix::product(Dm, Vm, tmp1, image::dyndim(3, 3), image::dyndim(3, 3));
        image::matrix::product_transpose(tmp1, Dm, T, image::dyndim(3, 3), image::dyndim(3, 3));
        int tensor_index[6] = {0, 4, 8, 1, 2, 5};
        for(i=0; i<6; i++)
            Diso[i] = T[tensor_index[i]];
        flat_matrix(matd_g, B);
        image::matrix::product(B.data(), Diso, BDi.data(), image::dyndim(b_count, 6), image::dyndim(6, 1));
        for(i=0; i<BDi.size(); i++)
        {
            hx[i] = p[0]*expf(-1*BDi[i]);
            data->x[i] = hx[i];
        }
    }
}

#endif//_PROCESS_HPP

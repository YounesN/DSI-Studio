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

class ICA : public BaseProcess
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
    std::vector<float> num_fibers;

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
        num_fibers.clear();
        num_fibers.resize(voxel.dim.size());

        approach = FICA_APPROACH_SYMM;
        numOfIC = voxel.numberOfFibers;
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
        voxel.fr.resize(numOfIC * voxel.dim.size());
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
        int center_voxel;
        if(voxel.threeDimensionalWindow)
            center_voxel = 9;
        else
            center_voxel = 4;

        int pi, pj;
        int ica_num = 0;
        int stop_count = 0;
        int stop_flag = 0;

        double tensor[9];
        double V[9],d[3];
        int numOfIC;          // input variable
        int NumofFiber;
        float resres[4];
        float varvar[4];
        float sum0;
        float sum1;
        float sums[3];
        float p_value = voxel.p_value; // p_value from user input goes here

        float eigenvectors[9];
        float fractions[4];
        float var21 = 5.780f; // input variable
        float var32 = 5.926f; // input variable
        float *x = new float[b_count];
        unsigned int tensor_index[9] = {0,3,4,3,1,5,4,5,2};
        double sum = 0;
        double min_value;
        double max_value;
        float V_best[9];
        float F_best[3];
        float V_bests[3][9];
        float F_bests[3][3];

        arma::mat armaicasig(voxel.numberOfFibers, b_count);
        arma::mat matsignal(b_count,1);
        arma::mat reference(1,b_count);
        arma::mat DD_tmp(6,1);
        arma::mat est_ref(1,b_count);
        arma::mat tensor_param(6,1);
        arma::mat factors1(1,1);
        arma::mat factors2(1,2);
        arma::mat factors3(1,3);
        arma::mat tmp;
        arma::mat input1(1,b_count);
        arma::mat input2(2,b_count);
        arma::mat input3(3,b_count);

        std::vector<float> signal(b_count);

        if(voxel.threeDimensionalWindow)
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

        if(voxel.threeDimensionalWindow)
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


        // DTI
        if (data.space.front() != 0.0)
        {
            float logs0 = std::log(std::max<float>(1.0,data.space.front()));
            for (unsigned int i = 1; i < data.space.size(); ++i)
                signal[i-1] = std::max<float>(0.0,logs0-std::log(std::max<float>(1.0,data.space[i])));
        }


        set_arma_col(matsignal, signal, 0);
        arma::mat pos_invg_dg = -voxel.matinvg_dg;
        tensor_param = pos_invg_dg * matsignal;


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


        if (data.fa[0] < voxel.FAth)
        {
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

            for(m=0; m<b_count; m++)
                reference(0, m) = mixedSig(center_voxel, m);

            for(ica_num = 0; ica_num <= voxel.numberOfFibers; ica_num++)
            {
                if(ica_num > 0)
                {
                    arma::mat new_mix(1,ica_num);

                    stop_flag = 0;
                    stop_count = 0;
                    while (stop_flag == 0)
                    {
                        if (stop_count==0)
                        {
                            g = FICA_NONLIN_TANH;
                        }
                        else if (stop_count==1)
                        {
                            g = FICA_NONLIN_POW3;
                        }
                        else if (stop_count==2)
                        {
                            g = FICA_NONLIN_GAUSS;
                        }

                        itpp::Fast_ICA fi(mixedSig);
                        fi.set_approach(approach);
                        fi.set_nrof_independent_components(ica_num);
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

                        if ((mixing_matrix.rows() != mixedSig.rows()) || (icasig.rows() != numOfIC))
                        {
                            if (stop_count >= 3)
                            {
                                stop_flag = 1;
                            }
                            else
                            {
                                stop_count = stop_count+1;
                            }
                            continue;
                        }

                        for(m=0; m<icasig.rows(); m++)
                        {
                            for(n=0; n<icasig.cols(); n++)
                            {
                                armaicasig(m, n) = icasig(m, n);
                            }
                        }

                        new_mix =  reference * arma::pinv(armaicasig);

                        for (m = 0; m < icasig.rows(); m++)
                        {
                            for (n=0;n<icasig.cols();n++)
                            {
                                icasig.set(m,n,new_mix(m)*icasig.get(m,n));
                            }
                        }

                        stop_flag = 1;

                        if (ica_num == 1)
                        {
                            stop_flag = 1;
                        }
                        else if (ica_num == 2)
                        {
                            double sum = 0;
                            for(n=0;n<icasig.cols();n++)
                            {
                                sum+=abs(icasig.get(1, n)/icasig.get(0, n));
                            }

                            if ((sum/icasig.cols()<10) && (sum/icasig.cols()>0.1f))
                                stop_flag =1;
                        }
                        else if (ica_num == 3)
                        {
                            for(m = 0; m < icasig.rows(); m++)
                            {
                                double sum = 0;
                                for(n=0;n<icasig.cols();n++)
                                    sum+= icasig.get(m,n);
                                if (sum<0)
                                {
                                    for(n=0;n<icasig.cols();n++)
                                        icasig.set(m, n, icasig.get(m,n)*-1);
                                }
                            }


                            sums[0]=0.0f; sums[1]=0.0f; sums[2]=0.0f;

                            for(n=0;n<icasig.cols();n++)
                            {
                                sums[0] += abs(icasig.get(1, n)/icasig.get(0, n));
                                sums[1] += abs(icasig.get(1, n)/icasig.get(2, n));
                                sums[2] += abs(icasig.get(0, n)/icasig.get(2, n));
                            }
                            sums[0] = sums[0]/icasig.cols();
                            sums[1] = sums[1]/icasig.cols();
                            sums[2] = sums[2]/icasig.cols();

                            float max=sums[0], min=sums[1];
                            for(n=1; n<3; n++)
                            {
                                if(max<sums[n])
                                    max = sums[n];
                                if(min>sums[n])
                                    min = sums[n];
                            }

                            if ((max < 30) && (min > 1/30))
                            {
                                stop_flag =1;
                            }
                        }

                        if (stop_count >= 3)
                        {
                            stop_flag = 1;
                        }
                        else
                        {
                            stop_count=stop_count+1;
                        }
                    } // while

                    if (stop_count >= 3)
                    {
                        resres[ica_num] = 999;
                        varvar[ica_num] = 999;
                    }
                    else
                    {
                        for(m = 0; m < icasig.rows(); m++)
                        {
                            sum = 0;
                            min_value = icasig.get(m, 0);
                            max_value = icasig.get(m, 0);
                            for(n=0;n<icasig.cols();n++)
                            {
                                sum+=icasig.get(m,n);
                                if(icasig.get(m,n) > max_value)
                                    max_value = icasig.get(m, n);
                                if(icasig.get(m,n) < min_value)
                                    min_value = icasig.get(m,n);
                            }

                            if (min_value < 0.05f)
                            {
                                for(n=0;n<icasig.cols();n++)
                                    icasig.set(m, n, icasig.get(m,n) - min_value+0.05);
                            }
                        }

                        switch(ica_num)
                        {
                        case 1:
                            for (n=0; n<9; n++)
                            {
                                F_bests[0][n]=0.0f;
                                V_bests[0][n]=0.0f;
                            }
                            for(m = 0; m < icasig.rows(); m++)
                            {
                                for(n=0; n<numOfIC; n++)
                                    F_bests[m][n] = mixing_matrix(center_voxel, n);

                                for(n=0;n<icasig.cols();n++)
                                    signal[n]= std::max<float>(0.0, std::log(std::max<float>(1.0, icasig.get(m,n))));

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
                                V_bests[0][(m-1)*3+0] = V[0];
                                V_bests[0][(m-1)*3+1] = V[1];
                                V_bests[0][(m-1)*3+2] = V[2];
                                DD_tmp(0) = V[0]*V[0]*0.0017;
                                DD_tmp(1) = V[1]*V[1]*0.0017;
                                DD_tmp(2) = V[2]*V[2]*0.0017;
                                DD_tmp(3) = V[0]*V[1]*0.0017;
                                DD_tmp(4) = V[0]*V[2]*0.0017;
                                DD_tmp(5) = V[1]*V[2]*0.0017;

                                tmp = voxel.matinvg_dg * DD_tmp;
                                for(n=0; n<b_count; n++)
                                    input1(0, n) = tmp(n, 0);
                            }
                            tmp = reference*arma::pinv(input1);
                            factors1(0, 0) = tmp(0, 0);
                            est_ref = factors1*input1; //1xb_count
                            break;
                        case 2:
                            for (n=0; n<9; n++)
                            {
                                F_bests[1][n]=0.0f;
                                V_bests[1][n]=0.0f;
                            }
                            for(m = 0; m < icasig.rows(); m++)
                            {
                                for(n=0; n<numOfIC; n++)
                                    F_bests[m][n] = mixing_matrix(center_voxel, n);

                                for(n=0;n<icasig.cols();n++)
                                    signal[n]= std::max<float>(0.0, std::log(std::max<float>(1.0, icasig.get(m,n))));

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
                                V_bests[1][(m-1)*3+0] = V[0];
                                V_bests[1][(m-1)*3+1] = V[1];
                                V_bests[1][(m-1)*3+2] = V[2];
                                DD_tmp(0) = V[0]*V[0]*0.0017;
                                DD_tmp(1) = V[1]*V[1]*0.0017;
                                DD_tmp(2) = V[2]*V[2]*0.0017;
                                DD_tmp(3) = V[0]*V[1]*0.0017;
                                DD_tmp(4) = V[0]*V[2]*0.0017;
                                DD_tmp(5) = V[1]*V[2]*0.0017;
                                tmp = voxel.matinvg_dg * DD_tmp;
                                for(n=0; n<b_count; n++)
                                    input2(m, n) = tmp(n, 0);
                            }
                            tmp = reference*arma::pinv(input2);
                            factors2(0, 0) = tmp(0, 0);
                            factors2(0, 1) = tmp(0, 1);
                            est_ref = factors2*input2; //1xb_count
                            break;
                        case 3:
                            for (n=0; n<9; n++)
                            {
                                F_bests[2][n]=0.0f;
                                V_bests[2][n]=0.0f;
                            }
                            for(m = 0; m < icasig.rows(); m++)
                            {
                                for(n=0; n<numOfIC; n++)
                                    F_bests[m][n] = mixing_matrix(center_voxel, n);

                                for(n=0;n<icasig.cols();n++)
                                    signal[n]= std::max<float>(0.0, std::log(std::max<float>(1.0, icasig.get(m,n))));

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
                                V_bests[2][(m-1)*3+0] = V[0];
                                V_bests[2][(m-1)*3+1] = V[1];
                                V_bests[2][(m-1)*3+2] = V[2];

                                DD_tmp(0) = V[0]*V[0]*0.0017;
                                DD_tmp(1) = V[1]*V[1]*0.0017;
                                DD_tmp(2) = V[2]*V[2]*0.0017;
                                DD_tmp(3) = V[0]*V[1]*0.0017;
                                DD_tmp(4) = V[0]*V[2]*0.0017;
                                DD_tmp(5) = V[1]*V[2]*0.0017;
                                tmp = voxel.matinvg_dg * DD_tmp;
                                for(n=0; n<b_count; n++)
                                    input3(m, n) = tmp(n, 0);
                            }
                            tmp = reference*arma::pinv(input3);
                            factors3(0, 0) = tmp(0, 0);
                            factors3(0, 1) = tmp(0, 1);
                            factors3(0, 2) = tmp(0, 2);
                            est_ref = factors3*input3; //1xb_count
                            break;
                        }

                        sum=0.0f; sum1=0.0f;
                        for(n=0;n<icasig.cols();n++)
                        {
                            sum+=est_ref(n);
                            sum1+= (est_ref(n)-reference(n))*(est_ref(n)-reference(n));
                        }
                        resres[numOfIC] = sum1/icasig.cols();

                        sum1=0;
                        for(n=0;n<icasig.cols();n++)
                        {
                            sum1+= (est_ref(n)-sum/icasig.cols())*(est_ref(n)-sum/icasig.cols());
                        }

                        varvar[numOfIC] = sum1/icasig.cols();
                    }

                }
                else if (ica_num == 0)
                {
                    sum=0;
                    for(n=0;n<b_count;n++)
                    {
                        sum+=reference(n);
                    }

                    sum1=0;
                    for(n=0;n<b_count;n++)
                    {
                        sum1+= (reference(n)-sum/b_count)*(reference(n)-sum/b_count);
                    }

                    resres[4] = sum1/b_count;
                }
            }

            NumofFiber = 0;

            if (NumofFiber == 0 && resres[1] == 999)
            {
                NumofFiber = 1;
            }

            if (NumofFiber == 0 && resres[2] == 999)
            {
                if (10.5*(varvar[0]-varvar[1])/varvar[1] <var21)
                {
                    NumofFiber = 1;
                }
                else
                {
                    NumofFiber = 2;
                }
            }

            if (NumofFiber ==0 && 10.5f*(varvar[0]-varvar[1])/varvar[1] <var21)
            {
                NumofFiber = 1;
            }

            if (NumofFiber ==0 && 9.5f*(varvar[1]-varvar[2])/varvar[2] <var32)
            {
                NumofFiber = 2;
            }

            if (NumofFiber ==0)
            {
                NumofFiber = 3;
            }

            //Recored V at NumofFiber from ICA stack Copyfiles...
            switch(NumofFiber)
            {
            case 0:
                V[0] = V[1] = V[2] = 0;
                std::copy(V, V+3, voxel.fib_dir.begin() + data.voxel_index * 3);
                std::copy(V, V+3, voxel.fib_dir.begin() + voxel.dim.size() * 3 + data.voxel_index * 3);
                std::copy(V, V+3, voxel.fib_dir.begin() + 2*voxel.dim.size() * 3 + data.voxel_index * 3);
                voxel.fr[data.voxel_index] = 0;
                voxel.fr[voxel.dim.size()+data.voxel_index] = 0;
                voxel.fr[2*voxel.dim.size()+data.voxel_index] = 0;
                break;
            case 1:
                V[0] = V[1] = V[2] = 0;
                for (m=0; m<3; m++)
                {
                    V_best[m] = V_bests[0][m];
                    F_best[m] = F_bests[0][m];
                }
                std::copy(V_best, V_best+3, voxel.fib_dir.begin() + data.voxel_index * 3);
                std::copy(V, V+3, voxel.fib_dir.begin() + voxel.dim.size() * 3 + data.voxel_index * 3);
                std::copy(V, V+3, voxel.fib_dir.begin() + 2*voxel.dim.size() * 3 + data.voxel_index * 3);
                voxel.fr[data.voxel_index] = F_best[0];
                voxel.fr[voxel.dim.size()+data.voxel_index] = 0;
                voxel.fr[2*voxel.dim.size()+data.voxel_index] = 0;
                break;
            case 2:
                V[0] = V[1] = V[2] = 0;
                for (m=0; m<6; m++)
                    V_best[m] = V_bests[1][m];
                for (m=0; m<3; m++)
                    F_best[m] = F_bests[1][m];

                std::copy(V_best, V_best+3, voxel.fib_dir.begin() + data.voxel_index * 3);
                std::copy(V_best+3, V_best+6, voxel.fib_dir.begin() + voxel.dim.size() * 3 + data.voxel_index * 3);
                std::copy(V, V+3, voxel.fib_dir.begin() + 2*voxel.dim.size() * 3 + data.voxel_index * 3);
                voxel.fr[data.voxel_index] = F_best[0];
                voxel.fr[voxel.dim.size()+data.voxel_index] = F_best[1];
                voxel.fr[2*voxel.dim.size()+data.voxel_index] = 0;
                break;
            case 3:
                for (m=0; m<9; m++)
                    V_best[m] = V_bests[1][m];
                for (m=0; m<3; m++)
                    F_best[m] = F_bests[2][m];
                std::copy(V_best, V_best+3, voxel.fib_dir.begin() + data.voxel_index * 3);
                std::copy(V_best+3, V_best+6, voxel.fib_dir.begin() + voxel.dim.size() * 3 + data.voxel_index * 3);
                std::copy(V_best+6, V_best+9, voxel.fib_dir.begin() + 2*voxel.dim.size() * 3 + data.voxel_index * 3);
                voxel.fr[data.voxel_index] = F_best[0];
                voxel.fr[voxel.dim.size()+data.voxel_index] = F_best[1];
                voxel.fr[2*voxel.dim.size()+data.voxel_index] = F_best[2];
                break;
            }
        } // if FA<voxel.FAth

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

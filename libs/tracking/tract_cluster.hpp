#ifndef TRACT_CLUSTER_HPP
#define TRACT_CLUSTER_HPP
#include <vector>
#include <boost/noncopyable.hpp>
#include "image/image.hpp"
#include <map>

struct Cluster : public::boost::noncopyable
{

    std::vector<unsigned int> tracts;
    unsigned int index;

};
class BasicCluster
{
protected:
    std::vector<Cluster*> clusters;
    void sort_cluster(void);
public:
    virtual ~BasicCluster(void);

public:
    virtual void add_tract(const float* points,unsigned int count) = 0;
    virtual void run_clustering(void) = 0;
public:
    unsigned int get_cluster_count(void) const
    {
        return clusters.size();
    }
    const unsigned int* get_cluster(unsigned int cluster_index,unsigned int& cluster_size) const
    {
        cluster_size = clusters[cluster_index]->tracts.size();
        return &*clusters[cluster_index]->tracts.begin();
    }
};

template<typename method_type>
class FeatureBasedClutering : public BasicCluster
{
    std::vector<std::vector<double> > features;
    std::vector<unsigned char> classifications;
    mutable std::vector<unsigned int> result;
    method_type clustering_method;
    unsigned int cluster_number;
public:
    FeatureBasedClutering(const float* param):cluster_number(param[0]),clustering_method(param[0]) {}
    virtual ~FeatureBasedClutering(void) {}

public:
    virtual void add_tract(const float* points,unsigned int count)
    {
        std::vector<double> feature(10);
        std::copy(points,points+3,feature.begin());
        std::copy(points+count-3,points+count,feature.begin()+3);
        count >>= 1;
        count -= count%3;
        std::copy(points+count-3,points+count,feature.begin()+6);
        feature.back() = count;
        features.push_back(feature);
    }
    virtual void run_clustering(void)
    {
        classifications.resize(features.size());
        clustering_method(features.begin(),features.end(),10,classifications.begin());
        std::map<unsigned char,std::vector<unsigned int> > cluster_map;
        for (unsigned int index = 0;index < classifications.size();++index)
            cluster_map[classifications[index]].push_back(index);
		clusters.resize(cluster_map.size());
                std::map<unsigned char,std::vector<unsigned int> >::iterator iter = cluster_map.begin();
                std::map<unsigned char,std::vector<unsigned int> >::iterator end = cluster_map.end();
                for(unsigned int index = 0;iter != end;++iter,++index)
		{
			clusters[index] = new Cluster();
			clusters[index]->tracts.swap(iter->second);
			clusters[index]->index= index;
		}
		sort_cluster();
    }

};



class TractCluster : public BasicCluster
{
    image::geometry<3> dim;
    unsigned int w,wh;
    float error_distance;
private:


    void set_tract_labels(Cluster* from,Cluster* to);
    void merge_tract(unsigned int tract_index1,unsigned int tract_index2);
    int get_index(short x,short y,short z);
private:
    std::vector<std::vector<unsigned int>*> voxel_connection;
    const std::vector<unsigned int>* add_connection(unsigned short index,unsigned int track_index);

private:
    std::vector<Cluster*> tract_labels;// 0 is no cluster
    std::vector<std::vector<unsigned short> > tract_passed_voxels;
    std::vector<std::vector<unsigned short> > tract_ranged_voxels;
    std::vector<unsigned int>							 tract_length;



public:
    TractCluster(const float* param);
    ~TractCluster(void);
    void add_tract(const float* points,unsigned int count);
	void run_clustering(void){sort_cluster();}

};




#endif//TRACT_CLUSTER_HPP

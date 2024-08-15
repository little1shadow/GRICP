#ifndef ICP_OPTIMIZER_H
#define ICP_OPTIMIZER_H

#include <iostream>
#include <fstream>
#include <vector>
#include<numeric>
#include <cmath>
#include <float.h>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>
#include <Eigen/Geometry>
#include<Eigen/Dense>
#include "nanoflann.hpp"
using namespace std;
/* Each point in a point cloud is loaded as a line vector, 
but every computation is made with the mathematical convention ! (column vectors)
As a consequence, TransMatrix is a column vector */

typedef Eigen::Matrix<double,3,3> RotMatrix;               //A type for the rotation matrix
typedef Eigen::Matrix<double,3,1> TransMatrix;             //A type for the translation matrix
typedef std::pair<RotMatrix,TransMatrix> RigidTransfo;     //A type for the rigid transform
typedef Eigen::Matrix<double,Eigen::Dynamic,3> PointCloud; //A type for the point clouds
//Enumerator for setting the underlying ICP method used
enum IcpMethod {pointToPoint,pointToPlane};

class GranularBall{
  public:
  friend class icpOptimizer;
  Eigen::Vector3d geocenter;
  double radius;
  double overlapvolume;
  GranularBall(Eigen::Vector3d Geocenter, double Radius); //initialization
  GranularBall(); //initialization
  
  double OverLap(GranularBall other);
  double Distance(GranularBall& other); 
  double OverlapRegion(GranularBall other);
  vector<GranularBall>  loadGranularBall(Eigen::Matrix<double,Eigen::Dynamic,3> pointcloud, vector<double> Radius);
};

class IcpOptimizer
{
public:
  //Constructor
  IcpOptimizer(PointCloud _firstCloud, PointCloud _secondCloud, pair<string, string> point_cloud_gb_path, pair<string, string> first_point_cloud_granular_info_path, pair<string, string> second_point_cloud_granular_info_path, size_t _kNormals, int _nbIterations, int _nbIterationsIn, double _mu, int _nbIterShrink, double _p, IcpMethod _method, bool _verbose);
  //The algorithm itself
  int performSparceICP();
   
   // transfer granular ball to pointcloud
  PointCloud GBToPointCloud(vector<GranularBall>  GB);
  //First step : compute correspondances
  std::vector<int> computeCorrespondances(PointCloud refCloud, PointCloud queryCloud);

  //Computer GB corresponces index
  std::vector<int> computeCorrespondencesGB(vector<GranularBall> refGranularBalls, vector<GranularBall> queryGranularBalls);
  
  //Compute Overlap Volume of each GB
  vector<double> computeGBoverlap(vector<GranularBall> refGranularBalls);

  //Apply rigid transformation to a point cloud
  PointCloud movePointCloud(PointCloud pointCloud, RigidTransfo t);

  //Apply rigid transformation to a GBs.
  std::vector<GranularBall> moveGranularBall(std::vector<GranularBall> GB, RigidTransfo t);

  //Normal estimation
  Eigen::Matrix<double,Eigen::Dynamic,3> estimateNormals(PointCloud pointCloud, const size_t k);

  //Classical rigid transform estimation (point-to-point)
  RigidTransfo rigidTransformPointToPoint(PointCloud a, PointCloud b);//const;

  RigidTransfo rigidTransfoGB_poseadjust(vector<GranularBall> firstCloudGB, vector<GranularBall> secondCloudGB);//
  


  //Shrink operator
  TransMatrix shrink(TransMatrix h) const;

  //Computing composition of tNew by tOld (tNew o tOld)
  RigidTransfo compose(RigidTransfo tNew, RigidTransfo tOld) const;

  //Selection of a subset in a PointCloud
  PointCloud selectSubsetPC(PointCloud p, std::vector<int> indice) const;

  //Selection of a subset of GranularBalls
  
  std::vector<GranularBall> selectSubsetGB(std::vector<GranularBall> GB, std::vector<int> indices);

  //Updates the iterations measure by estimating the amplitude of rigid motion t
  void updateIter(RigidTransfo t);

  //compute RMSE of Point clouds
  double computeRMSE(PointCloud a, PointCloud b);

  //Save iterations to file
  void saveIter(std::string pathToFile);
  
  std::pair<double,vector<double>> Clustering(std::vector<double> errors, int numOfcenter);
  
  std::vector<vector<double>> MKCParameterEstimator(vector<double> errors);

  //save the distance between movedPC and matchPC
  void saveMetrics(std::string pathToFile, std::vector<double> metrics);
 
 //compute the error between eatimated transformation and the groundtruth transformation
  std::vector<double> computedTransformationError(RigidTransfo estimation,RigidTransfo truth);

//analysis input path to acquire other path
  pair<string, string> ExtractFileNameAndParentDir(const string& filePath);

  //Getters
  Eigen::Matrix<double,Eigen::Dynamic,3> getFirstNormals() const;
  Eigen::Matrix<double,Eigen::Dynamic,3> getMovedNormals() const;
  PointCloud getMovedPointCloud() const;
  RigidTransfo getComputedTransfo() const;
  double getReferenceDist() const;
private:
  const PointCloud firstCloud;
  const PointCloud secondCloud;
  PointCloud movingPC;
  Eigen::Matrix<double,Eigen::Dynamic,3> movingNormals;
  Eigen::Matrix<double,Eigen::Dynamic,3> selectedNormals; //For point-to-plane
  RigidTransfo computedTransfo;
  RigidTransfo lastItertransfo;
  std::vector<double> iterations;
  double referenceDist;
  bool hasBeenComputed;
  /*I don't use the PointCloud name for the normals in order to distinguish them 
  from the vertice*/
  Eigen::Matrix<double,Eigen::Dynamic,3> firstNormals;
  Eigen::Matrix<double,Eigen::Dynamic,3> secondNormals;

  Eigen::Matrix<double,Eigen::Dynamic,3> lambda; //Lagrange multiplier for step 2.1

  //Algorithm parameters
  const size_t kNormals;    //K-nn parameter for normal computation
  const int nbIterations;   //Number of iterations for the algorithm
  const int nbIterationsIn; //Number of iterations for the step 2 of the algorithm
  const double mu;          //Parameter for ICP step 2.1
  const int nbIterShrink;   //Number of iterations for the shrink part (2.1)
  const double p;           //We use the norm L_p
  const bool verbose;       //Verbosity trigger
  const IcpMethod method;   //The used method (point to point or point to plane)
  pair<string, string> point_cloud_gb_path; //the path of granulared point cloud
  pair<string, string> first_point_cloud_granular_info_path; //the granular ball info of first point cloud
  pair<string, string> second_point_cloud_granular_info_path; //the granular ball info of second point cloud
};


/*This section is an adapter written to enable the use of nanoflann's KD-tree.*/
template <class T, class DataSource, typename _DistanceType = T>
struct GB_Adaptor {
  typedef T ElementType;
  typedef _DistanceType DistanceType;

  const DataSource& data_source;

  GB_Adaptor(const DataSource& _data_source) : data_source(_data_source) {}

  inline DistanceType
  operator()(const T* a,
             const size_t b_idx,
             size_t size,
             DistanceType worst_dist = -1) const
  {
    DistanceType result = DistanceType();
    const T* last = a + size;
    const T* lastgroup = last - 3;
    size_t d = 0;
    Eigen::Vector3d Geocenter1;
    Geocenter1(0) = a[0];
    Geocenter1(1) = a[1];
    Geocenter1(2) = a[2];
    double Radius1 = a[3];
    Eigen::Vector3d Geocenter2;
    Geocenter2(0) = data_source.kdtree_get_pt(b_idx, 0);
    Geocenter2(1) = data_source.kdtree_get_pt(b_idx, 1);
    Geocenter2(2) = data_source.kdtree_get_pt(b_idx, 2);
    double Radius2 = data_source.kdtree_get_pt(b_idx, 3);
    result = data_source.ComputerDistanceGB(make_pair(Geocenter1, Radius1),
                                            make_pair(Geocenter2, Radius2));
    //cout << "result: " << result << endl;
    return result;
  }

  template <typename U, typename V>
  inline DistanceType
  accum_dist(const U a, const V b, int) const
  {
    return std::abs(a - b);
  }
};

struct metric_GB {
  template <class T, class DataSource>
  struct traits {
    typedef GB_Adaptor<T, DataSource> distance_t;
  };
};
struct Point {
  double x, y, z;
  double radius;

  Point(double x, double y, double z, double radius) : x(x), y(y), z(z), radius(radius)
  {}
};

// Define a dataset adapter.
class pointcloud {
public:
  std::vector<Point> points;

  inline size_t
  kdtree_get_point_count() const
  {
    return points.size();
  }

  inline double
  kdtree_get_pt(const size_t idx, const size_t dim) const
  {
    if (dim == 0)
      return points[idx].x;
    else if (dim == 1)
      return points[idx].y;
    else if (dim == 2)
      return points[idx].z;
    else if (dim == 3)
      return points[idx].radius;
    return 0.0;
  }
  inline double
  ComputerDistanceGB(pair<Eigen::Vector3d, double> GBparameter1, pair<Eigen::Vector3d, double> GBparameter2) const
  {
    double pi = 3.14159;
    double weight_overlap = 0.5; 
    double weight_distance = 1 - weight_overlap;
    Eigen::Vector3d geocenter1 = GBparameter1.first;
    Eigen::Vector3d geocenter2 = GBparameter2.first;
    double radius1 = GBparameter1.second;
    double radius2 = GBparameter2.second;

    double d = (geocenter1 - geocenter2).norm();
    if (d > radius1 + radius2) {
      return weight_distance * d;
    }
    else if (d < max(radius1, radius2) - min(radius1, radius2)) {
      return weight_overlap * (max(radius1, radius2), 3) /
                 pow(min(radius1, radius2), 3) +
             weight_distance * d;
    }
    else {
      double cosAlpha = (pow(radius1, 2) + d - pow(radius2, 2)) / (2 * radius1 * d);
      double cosBeta = (pow(radius2, 2) + d - pow(radius1, 2)) / (2 * radius2 * d);
      double h_1 = radius1 - radius1 * cosAlpha;
      double h_2 = radius2 - radius2 * cosBeta;
      double overlapsqure = pi * pow(h_1, 2) * (radius1 - h_1 / 3) +
                            pi * pow(h_2, 2) * (radius2 - h_2 / 3);
      double overlaprate = overlapsqure / (pow(max(radius1, radius2), 3) * pi * 4 / 3);
      return weight_overlap / overlaprate + weight_distance * d;
    }
  }
  template <class BBOX>
  bool
  kdtree_get_bbox(BBOX&) const
  {
    return false;
  }
};

#endif


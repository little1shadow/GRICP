#include "IcpOptimizer.h"

#include "../../lib/ObjLoader/ObjLoader.h"
#include <shark/Algorithms/KMeans.h>                     //k-means algorithm
#include <shark/Algorithms/Trainers/NormalizeComponentsUnitVariance.h> //normalize
#include <shark/Data/Csv.h>                              //load the csv file
#include <shark/Models/Clustering/HardClusteringModel.h> //model performing hard clustering of points

#include <chrono>
#include <numeric>
#include <vector>
#include<iostream>
// ###end<includes>
using namespace std;
using namespace Eigen;
using namespace nanoflann;
using namespace shark;

/*Parameters of GRICP*/
double pi = 3.14159; // pi
double weight_overlap = 0.5;
double weight_distance = 1 - weight_overlap;
double lowerbound = 0.1;     // lower bound of the soulution zone
double upperbound = 1;
double sigma_interval = 1;// sigma_traverse's interval
double tau_1 = 0.1;//the threshold of granular ball correspodences's difference in radius
double tau_2 = 0.1;//the threshold of granular ball correspodences's difference in overlap volume
double yita = 0.1;        
int maxiter = 25;           // max iteration of sigma computation
/*
Main constructor. Initializes the point clouds and the GRICP parameters.
*/
IcpOptimizer::IcpOptimizer(Matrix<double, Dynamic, 3> _firstCloud,
                           Matrix<double, Dynamic, 3> _secondCloud,
                           pair<string, string> _point_cloud_gb_path,
                           pair<string, string> _first_point_cloud_granular_info_path,
                           pair<string, string> _second_point_cloud_granular_info_path,
                           size_t _kNormals,
                           int _nbIterations,
                           int _nbIterationsIn,
                           double _mu,
                           int _nbIterShrink,
                           double _p,
                           IcpMethod _method,
                           bool _verbose)
: firstCloud(_firstCloud)
, secondCloud(_secondCloud)
, point_cloud_gb_path(_point_cloud_gb_path)
, first_point_cloud_granular_info_path(_first_point_cloud_granular_info_path)
, second_point_cloud_granular_info_path(_second_point_cloud_granular_info_path)
, kNormals(_kNormals)
, nbIterations(_nbIterations)
, nbIterationsIn(_nbIterationsIn)
, mu(_mu)
, nbIterShrink(_nbIterShrink)
, p(_p)
, method(_method)
, verbose(_verbose)
{
  // Normal estimation
  cout << "Estimating normals for first cloud" << endl;
  firstNormals = firstCloud;
  if (method == pointToPlane) {
    cout << "Estimating normals for second cloud" << endl;
    secondNormals = estimateNormals(_secondCloud, kNormals);
    cout << "Done with normal estimation" << endl;
  }

  // Initialize the computed transformation
  computedTransfo = RigidTransfo(RotMatrix::Identity(), TransMatrix::Zero(3, 1));

  // Initialize the Lagrange multipliers to 0 for step 2.1
  lambda.resize(firstCloud.rows(), 3);
  lambda.setZero();

  // Initialize the reference distance (bounding box diagonal of cloud 1)
  Matrix<double, 1, 3> minCloudOne = firstCloud.colwise().minCoeff();
  Matrix<double, 1, 3> maxCloudOne = firstCloud.colwise().maxCoeff();
  referenceDist = (maxCloudOne - minCloudOne).norm();
  cout << "The reference distance is : " << referenceDist << endl;

  // Initialize the other parameter
  hasBeenComputed = false;
}

/*
This function is the main implementation of the algorithm where every step are made
explicit.
*/
int
IcpOptimizer::performSparceICP()
{ 
  if (firstCloud.rows() == 0 || secondCloud.rows() == 0)
    return 1;
  if (method == pointToPoint)
    cout << "Beginning ICP with method Point to Point" << endl;
  else if (method == pointToPlane)
    cout << "Beginning ICP with method Point to Plane" << endl;
  // Initialize the point cloud that is going to move

  // Determine the data set, rotation direction, and size for subsequent ground truth selection.
  pair<string, string> extracted_rotation =
      ExtractFileNameAndParentDir(point_cloud_gb_path.second);
  string path = extracted_rotation.first;
  // Utilize string streams for processing.
  stringstream ss(path);
  string item;
  vector<string> parts;
  // Split the string using the '_' delimiter."
  while (getline(ss, item, '_')) {
    parts.push_back(item);
  }

  // choose groundtruth
  Matrix<double, 3, 3> TruthRotation;
  Matrix<double, 3, 1> TruthTranslation;

  if (!parts[2].compare("rotatex")) {
    if (!parts[3].compare("100")) {
      TruthRotation << 1, 0, 0, 0, -0.173, -0.985, 0, 0.985, -0.173;
    }
    if (!parts[3].compare("120")) {
      TruthRotation << 1, 0, 0, 0, -0.5, -0.866, 0, 0.866, -0.5;
    }
    if (!parts[3].compare("150")) {
      TruthRotation << 1, 0, 0, 0, -0.866, 0.5, 0, -0.5, -0.866;
    }
    if (!parts[3].compare("180")) {
      TruthRotation << 1, 0, 0, 0, -1, 0, 0, 0, -1;
    }
  }
  else {
    if (!parts[3].compare("100")) {
      TruthRotation << -0.173, 0, 0.985, 0, 1, 0, -0.985, 0, -0.173;
    }
    if (!parts[3].compare("120")) {
      TruthRotation << -0.5, 0, 0.866, 0, 1, 0, -0.866, 0, -0.5;
    }
    if (!parts[3].compare("150")) {
      TruthRotation << -0.866, 0, 0.5, 0, 1, 0, -0.5, 0, -0.866;
    }
    if (!parts[3].compare("180")) {
      if (!parts[0].compare("ArmadilloStand")) {
        TruthRotation << -1, 0, 0, 0, 1, 0, 0, 0, -1;
      }
      else {
        TruthRotation << -1, 0, 0, 0, -1, 0, 0, 0, 1;
      }
    }
  }

  // choose the groundtruth of translation
  if (!parts[0].compare("ArmadilloStand")) {
    TruthTranslation << 0, 0, 0;
  }

  else {
    if (!parts[2].compare("rotatey") && !parts[3].compare("180")) {
      TruthTranslation << 0, 0.3, 0;
    }
    else {
      TruthTranslation << 0, 0.15, 0.05;
    }
  }
  RigidTransfo TruthTransformation = RigidTransfo(TruthRotation, TruthTranslation);
  // read granular ball
  ObjectLoader myLoader;
  Matrix<double, Dynamic, 3> firstCloud_gb = myLoader(point_cloud_gb_path.first); //
  Matrix<double, Dynamic, 3> secondCloud_gb = myLoader(point_cloud_gb_path.second);
  Matrix<double, Dynamic, 3> movingPC_gb = firstCloud_gb;
  movingNormals = firstNormals;
  // Beginning of the algorithm itself  nbIterations

  vector<double> REs; // save the RMSE between movedPC and matchPC
  vector<double> TEs;
  vector<double> err;
  /*read radius and points number inside granular ball*/
  vector<int> firstCloudGranullballPointNumber; 
  vector<double> firstCloudGranullballRadius; 

  cout << "Granullball backed run!" << endl;

  /*read source granumalr ball information*/
  ifstream input_firstCloud_GranullballPointNumber(
      first_point_cloud_granular_info_path.first); 
  ifstream input_firstCloud_GranullballRadius(
      first_point_cloud_granular_info_path.second); 
  cout << "successs import pcl" << endl;
 
  if (input_firstCloud_GranullballPointNumber.is_open()) { 
    int num;
    while (input_firstCloud_GranullballPointNumber >> num) { 
      firstCloudGranullballPointNumber.push_back(num); 
    }
    input_firstCloud_GranullballPointNumber.close(); 
  }
  else {
    cout << "Unable to open source PointCloud granullball sphere number file"
         << endl; 
  }

  /*read source granumalr ball information*/
  if (input_firstCloud_GranullballRadius.is_open()) { 
    double num;
    while (input_firstCloud_GranullballRadius >> num) { 
      firstCloudGranullballRadius.push_back(num); 
    }
    input_firstCloud_GranullballRadius.close(); 
  }
  else {
    cout << "Unable to open source PointCloud granullball sphere radius file"
         << endl; 
  }

 
  vector<int> secondCloudGranullballPointNumber; 
  vector<double> secondCloudGranullballRadius; 

  ifstream input_secondCloud_GranullballPointNumber(
      second_point_cloud_granular_info_path.first); 
  ifstream input_secondCloud_GranullballRadius(
      second_point_cloud_granular_info_path.second); 

  /*read target granumalr ball information**/
  if (input_secondCloud_GranullballPointNumber.is_open()) { 
    int num;
    while (input_secondCloud_GranullballPointNumber >> num) { 
      secondCloudGranullballPointNumber.push_back(num); 
    }
    input_secondCloud_GranullballPointNumber.close(); 
  }
  else {
    cout << "Unable to open target PointCloud granullball sphere number file"
         << endl; 
  }

  
  if (input_secondCloud_GranullballRadius.is_open()) { 
    double num;
    while (input_secondCloud_GranullballRadius >> num) { 
      secondCloudGranullballRadius.push_back(num); 
    }
    input_secondCloud_GranullballRadius.close(); 
  }
  else {
    cout << "Unable to open target PointCloud granullball sphere radius file"
         << endl; 
  }
  cout << "completed load granullball input file " << endl;

  /*generate granular ball*/
  GranularBall obj;
  vector<GranularBall> GB_first =
      obj.loadGranularBall(firstCloud_gb, firstCloudGranullballRadius);
  vector<GranularBall> GB_second =
      obj.loadGranularBall(secondCloud_gb, secondCloudGranullballRadius);
  /**/

  lastItertransfo = RigidTransfo(RotMatrix::Identity(), TransMatrix::Zero(3, 1));

  PointCloud source;
  PointCloud target;
  int maxiter;
  cout << "please input the maxiter: " << endl;
  cin >> maxiter;
  for (int iter = 0; iter < maxiter; iter++) {

    cout << "Iteration " << iter << endl << endl;
    // 2nd step : Computing transformation   nbIterationsIn
    RigidTransfo iterTransfo;
    for (int iterTwo = 0; iterTwo < 1; iterTwo++) {
   
      auto start = std::chrono::steady_clock::now();
      iterTransfo = rigidTransfoGB_poseadjust(GB_first, GB_second);
      lastItertransfo = iterTransfo; 
      auto end = std::chrono::steady_clock::now();
      auto duration = end - start;
      std::cout << "Time difference: "
                << std::chrono::duration<double, std::milli>(duration).count()
                << " ms" << std::endl;
      

      // Updating the moving GB

      GB_first = moveGranularBall(GB_first, iterTransfo); // 
      movingNormals = (iterTransfo.first * movingNormals.transpose()).transpose();
      computedTransfo = compose(iterTransfo, computedTransfo);
      updateIter(iterTransfo);       // Updating the iterations measure
    }
    // Compute Metrics
    double tr = (TruthRotation * computedTransfo.first.transpose()).trace();
    double RE = acos((tr - 1) / 2);
    double TE = (TruthTranslation - computedTransfo.second).norm();
    REs.push_back(RE);
    TEs.push_back(TE);

    source = movePointCloud(firstCloud_gb, computedTransfo);
    target = secondCloud_gb;
    double errors = 0;
    for (int i = 0; i < source.rows(); i++) {
      errors = + (source.row(i).transpose() - target.row(i).transpose()).norm();
    }
    err.push_back(errors);
  }

  
  pair<string, string> extracted_metrics_path =
      ExtractFileNameAndParentDir(point_cloud_gb_path.first);
  string pathRE = "../metrics/" + extracted_metrics_path.second + "/" +
                  extracted_metrics_path.first + "_RE.txt";
  string pathTE = "../metrics/" + extracted_metrics_path.second + "/" +
                  extracted_metrics_path.first + "_TE.txt";
  string pathErrs = "../metrics/" + extracted_metrics_path.second + "/" +
                    extracted_metrics_path.first + "_errs.txt";

  // save the metrics

  saveMetrics(pathRE, REs);
  saveMetrics(pathTE, TEs);
  saveMetrics(pathErrs, err);

  // align source point cloud

  movingPC = movePointCloud(firstCloud, computedTransfo);

  hasBeenComputed = true;
  return 0;
}

/*
This function computes each closest point in refCloud for each point in queryCloud using
the nanoflann kd-tree implementation. It returns the indice of the closest points of
queryCloud.
*/
vector<int>
IcpOptimizer::computeCorrespondances(Matrix<double, Dynamic, 3> refCloud,
                                     Matrix<double, Dynamic, 3> queryCloud)
{
  // Create an adapted kd tree for the point cloud
  typedef KDTreeEigenMatrixAdaptor<Matrix<double, Dynamic, 3>> my_kd_tree_t;

  // Create an index
  my_kd_tree_t mat_index(3, refCloud, 10 /* max leaf */);
  mat_index.index->buildIndex();

  vector<int> nearestIndices;
  for (int i = 0; i < queryCloud.rows(); i++) {
    // Current query point
    Matrix<double, 1, 3> queryPoint = queryCloud.block(i, 0, 1, 3);

    // Do a knn search
    const size_t num_results = 1; // We want the nearest neighbour
    vector<size_t> ret_indexes(num_results);
    vector<double> out_dists_sqr(num_results);

    KNNResultSet<double> resultSet(num_results);
    resultSet.init(&ret_indexes[0], &out_dists_sqr[0]);

    mat_index.index->findNeighbors(resultSet, &queryPoint[0], SearchParams(10));

    // Updating the resulting index vector
    nearestIndices.push_back(ret_indexes[0]);

    if (verbose) {
      cout << queryPoint(0, 0) << " " << queryPoint(0, 1) << " " << queryPoint(0, 2)
           << " refPoint" << endl;
      cout << refCloud(ret_indexes[0], 0) << " " << refCloud(ret_indexes[0], 1) << " "
           << refCloud(ret_indexes[0], 2) << " closestPoint" << endl
           << endl
           << endl;
    }
  }
  return nearestIndices;
}

/*computer correspondences GB version*/
vector<int>
IcpOptimizer::computeCorrespondencesGB(vector<GranularBall> refGranularBalls,
                                       vector<GranularBall> queryGranularBalls)
{
  pointcloud cloud;
  for (int i = 0; i < queryGranularBalls.size(); i++) {
    double x = queryGranularBalls[i].geocenter(0);
    double y = queryGranularBalls[i].geocenter(1);
    double z = queryGranularBalls[i].geocenter(2);
    double radius = queryGranularBalls[i].radius;
    cloud.points.emplace_back(x, y, z, radius);
  }
  vector<int> result;
  // build kd tree
  typedef nanoflann::
      KDTreeSingleIndexAdaptor<GB_Adaptor<double, pointcloud>, pointcloud, 3 /* dim */>
          my_kd_tree_t;
  my_kd_tree_t index(3, cloud);

  index.buildIndex();

  // run nearest point search
  std::vector<size_t> indices(1);
  std::vector<double> distances(1);
  for (int i = 0; i < refGranularBalls.size(); i++) {
    double x = refGranularBalls[i].geocenter(0);
    double y = refGranularBalls[i].geocenter(1);
    double z = refGranularBalls[i].geocenter(2);
    double radius = refGranularBalls[i].radius;
    Point query(x, y, z, radius);
    index.knnSearch(&query.x, 1, &indices[0], &distances[0]);
    result.push_back(indices[0]);
  }
  return result;
}



/*computer each overlap volume of gb*/
vector<double>
IcpOptimizer::computeGBoverlap(vector<GranularBall> GranularBalls){
  pointcloud cloud;
  for (int i = 0; i < GranularBalls.size(); i++) {
    double x = GranularBalls[i].geocenter(0);
    double y = GranularBalls[i].geocenter(1);
    double z = GranularBalls[i].geocenter(2);
    double radius = GranularBalls[i].radius;
    cloud.points.emplace_back(x, y, z, radius);
  }
  vector<double> result;
  // build kd tree
  typedef nanoflann::
      KDTreeSingleIndexAdaptor<GB_Adaptor<double, pointcloud>, pointcloud, 3 /* dim */>
          my_kd_tree_t;
  my_kd_tree_t index(3, cloud);

  index.buildIndex();

  // compute the overlap volume with nearest gb
  std::vector<size_t> indices(1);
  std::vector<double> distances(1);
  for (int i = 0; i < GranularBalls.size(); i++) {
    double x = GranularBalls[i].geocenter(0);
    double y = GranularBalls[i].geocenter(1);
    double z = GranularBalls[i].geocenter(2);
    double radius = GranularBalls[i].radius;
    Point query(x, y, z, radius);
    index.knnSearch(&query.x, 1, &indices[0], &distances[0]);
    double OverlapVolume = GranularBalls[i].OverlapRegion(GranularBalls[indices[0]]);
    result.push_back(OverlapVolume);
  }
  
  return result;
}
/*
Move the pointCloud according to the rigid transformation in t
*/
PointCloud
IcpOptimizer::movePointCloud(PointCloud pointCloud, RigidTransfo t)
{
  return (t.first * pointCloud.transpose() + t.second.replicate(1, pointCloud.rows()))
      .transpose();
}

/*
Move the GB according to the rigid transformation in t
*/
std::vector<GranularBall>
IcpOptimizer::moveGranularBall(std::vector<GranularBall> GB, RigidTransfo t)
{
  vector<GranularBall> GB_moved(GB.size());
  for (int i = 0; i < GB.size(); i++) {
    GB_moved[i].geocenter = t.first * GB[i].geocenter + t.second;
    GB_moved[i].radius = GB[i].radius;
  }
  return GB_moved;
}
/*
This function estimates the normals for the point cloud pointCloud. It makes use of the
k nearest neighbour algorithm implemented in FLANN
*/
Matrix<double, Dynamic, 3>
IcpOptimizer::estimateNormals(Matrix<double, Dynamic, 3> pointCloud, const size_t k)
{
  // Create an adapted kd tree for the point cloud
  typedef KDTreeEigenMatrixAdaptor<Matrix<double, Dynamic, 3>> my_kd_tree_t;

  // Create an index
  my_kd_tree_t mat_index(3, pointCloud, 10 /* max leaf */);
  mat_index.index->buildIndex();

  Matrix<double, Dynamic, 3> normals;
  normals.resize(pointCloud.rows(), 3);
  for (int i = 0; i < pointCloud.rows(); i++) {
    // Current point for which the normal is being computed
    Matrix<double, 1, 3> currentPoint = pointCloud.block(i, 0, 1, 3);

    // Do a knn search
    vector<size_t> ret_indexes(k);
    vector<double> out_dists_sqr(k);

    KNNResultSet<double> resultSet(k);
    resultSet.init(&ret_indexes[0], &out_dists_sqr[0]);

    mat_index.index->findNeighbors(resultSet, &currentPoint[0], SearchParams(10));

    // Compute the covariance matrix
    // Compute the barycentre of the nearest neighbours
    Matrix<double, 1, 3> barycentre;
    for (int j = 0; j < 3; j++) {
      double curVal = 0.;
      for (int neighbour = 0; neighbour < k; neighbour++)
        curVal += pointCloud(ret_indexes[neighbour], j);
      barycentre(0, j) = curVal / double(k);
    }

    // Compute the centered nearest neighbour matrix
    Matrix<double, Dynamic, 3> centeredNN;
    centeredNN.resize(k, 3);
    for (int j = 0; j < k; j++)
      centeredNN.row(j) = pointCloud.row(ret_indexes[j]) - barycentre;

    // Compute the covariance matrix
    Matrix<double, 3, 3> covariance = centeredNN.transpose() * centeredNN;

    // Computing its eigen values
    EigenSolver<Matrix<double, Dynamic, Dynamic>> eigenSolver(covariance);

    // Find the indice of the lowest eigen value
    int bestIndice = -1;
    double bestVal = DBL_MAX;
    for (int j = 0; j < 3; j++)
      if (eigenSolver.eigenvalues()(j, 0).real() < bestVal) {
        bestVal = eigenSolver.eigenvalues()(j, 0).real();
        bestIndice = j;
      }

    // Filling the normal
    normals.row(i) = eigenSolver.eigenvectors()
                         .block(0, bestIndice, 3, 1)
                         .normalized()
                         .transpose()
                         .real();
  }
  return normals;
}

/*
This function is to find the centers of errors
*/
std::pair<double, vector<double>>
IcpOptimizer::Clustering(std::vector<double> errors, int numOfcenter)
{
  Data<shark::RealVector> data;
  std::vector<std::vector<double>> rows; 
  for (int i = 0; i < errors.size(); i++) {
    std::vector<double> temp;
    temp.push_back(errors[i]);
    rows.push_back(temp);
    temp.clear();
  }
  std::size_t maximumBatchSize = 100;
  // copy rows of the file into the dataset
  std::size_t dimensions = rows[0].size();
  std::vector<std::size_t> batchSizes =
      shark::detail::optimalBatchSizes(rows.size(), maximumBatchSize);
  data = Data<shark::RealVector>(batchSizes.size());
  std::size_t currentRow = 0;
  for (std::size_t b = 0; b != batchSizes.size(); ++b) {
    shark::RealMatrix& batch = data.batch(b);
    batch.resize(batchSizes[b], dimensions);
    // copy the rows into the batch
    for (std::size_t i = 0; i != batchSizes[b]; ++i, ++currentRow) {
      SHARK_RUNTIME_CHECK(rows[currentRow].size() == dimensions,
                          "Vectors are required to have same size");

      for (std::size_t j = 0; j != dimensions; ++j) {
        batch(i, j) = rows[currentRow][j];
      }
    }
  }
  data.shape() = {dimensions};
  SIZE_CHECK(currentRow == rows.size());

  std::size_t elements = data.numberOfElements();

  // compute centroids using k-means clustering
  Centroids centroids;
  size_t iterations = kMeans(data, numOfcenter, centroids);
  //  report number of iterations by the clustering algorithm
  // write cluster centers/centroids

  Data<RealVector> c = centroids.centroids();

  std::vector<double> centers;
  for (int i = 0; i < numOfcenter; i++) { 
    double center = c.batch(0)(i, 0);
    centers.push_back(center);
  }
  HardClusteringModel<RealVector> model(&centroids);
  Data<unsigned> clusters = model(data);

  /*计算本次聚类的Ilertia*/
  double ilertia = 0;
  for (int i = 0; i < elements; i++) {
    double a_i = 0; // Intra-class distance"
    double b_i = 0; // Inter-class distance"
    for (int j = 0; j < elements; j++) {
      if (i == j) {
        continue;
      }
      if (clusters.element(i) == clusters.element(j)) { 
        a_i = a_i + abs(data.element(i)(0) - data.element(j)(0));
      }
      else { // 异类
        b_i = b_i + abs(data.element(i)(0) - data.element(j)(0));
      }
    }
    double s_i = (b_i - a_i) / max(a_i, b_i);

    ilertia = ilertia + s_i;
  }
  ilertia = ilertia / elements;
  return make_pair(ilertia, centers);
}

/*
this function aims to sovle the 3m parameter of MKC
*/
std::vector<vector<double>>
IcpOptimizer::MKCParameterEstimator(vector<double> errors)
{
  int maxcenters = 5; // Predefined maximum number of clusters
  double ilertia = -1;
  int optimal_centers_num = 1; 
  std::vector<double> optimal_centers;
  for (int numOfcenter = 1; numOfcenter < maxcenters; numOfcenter++) {

    pair<double, vector<double>> clusterResult = Clustering(errors, numOfcenter);
    if (clusterResult.first > ilertia) {
      ilertia = clusterResult.first;
      optimal_centers = clusterResult.second;
      optimal_centers_num = numOfcenter;
    }
  }
  cout << "the center of errors: ";
  for (int i = 0; i < optimal_centers.size(); i++) {
    cout << optimal_centers[i] << " ";
  }
  vector<double> C = optimal_centers;
  int M = optimal_centers_num;
  cout << endl;
  VectorXd Lambde(M);
  VectorXd Sigma(M);
  VectorXd h_hat(M);
  VectorXd g_hat(M);
  MatrixXd K(M, M);
  Sigma.setConstant(lowerbound);
  double objectvalue = 0;
  VectorXd optimal_Sigma(M);
  optimal_Sigma.setConstant(lowerbound);
  for (int iter = 0; iter < maxiter; iter++) {
    for (int index_sigma = 0; index_sigma < M; index_sigma = index_sigma + 1) {
      // cout<<"index_sigma: "<<index_sigma<<endl;
      for (double sigma_traverse = lowerbound; sigma_traverse < upperbound; sigma_traverse = sigma_traverse + sigma_interval) {
        // cout<<"sigma_traverse: "<<sigma_traverse<<endl;
        Sigma(index_sigma) = sigma_traverse;
        g_hat.setConstant(0);
        h_hat.setConstant(0);
        for (int i = 0; i < errors.size(); i++) {
          VectorXd temp(M);
          for (int j = 0; j < M; j++) {
            temp(j) = exp(-pow((errors[i] - C[j]), 2) / (2 * (pow(Sigma(j), 2)))) /
                      sqrt(2 * pi);
          }
          g_hat = g_hat + temp;
        }
        h_hat = g_hat / errors.size();

        for (int i = 0; i < M; i++) {
          for (int j = 0; j < M; j++) {
            K(i, j) = exp(-pow((C[i] - C[j]), 2) /
                          (2 * (pow(Sigma(i), 2) + pow(Sigma(j), 2)))) /
                      (sqrt(2 * pi) * sqrt(pow(Sigma(i), 2) + pow(Sigma(j), 2)));
          }
        }
        MatrixXd I = MatrixXd::Identity(M, M);
        MatrixXd obj = -((K + yita * I).inverse() * h_hat).transpose() * K *
                           (K + yita * I).inverse() * h_hat / 2 +
                       ((K + yita * I).inverse() * h_hat).transpose() * h_hat;
        double objv = obj(0, 0); 
        if (objv > objectvalue) {
          optimal_Sigma(index_sigma) = sigma_traverse;
          objectvalue = objv;
        }
      }
      Sigma(index_sigma) = optimal_Sigma(index_sigma);
    }
  }

  g_hat.setConstant(0);
  h_hat.setConstant(0);
  for (int i = 0; i < errors.size(); i++) {
    VectorXd temp(M);
    for (int j = 0; j < M; j++) {
      temp(j) =
          exp(-pow((errors[i] - C[j]), 2) / (2 * (pow(Sigma(j), 2)))) / sqrt(2 * pi);
    }
    g_hat = g_hat + temp;
  }
  h_hat = g_hat / errors.size();

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < M; j++) {
      K(i, j) =
          exp(-pow((C[i] - C[j]), 2) / (2 * (pow(Sigma(i), 2) + pow(Sigma(j), 2)))) /
          (sqrt(2 * pi) * sqrt(pow(Sigma(i), 2) + pow(Sigma(j), 2)));
    }
  }

  MatrixXd I = MatrixXd::Identity(M, M);
  Lambde = (K + yita * I).inverse() * h_hat; // compute the optimal combination weights
  std::vector<double> lambda_optimal(
      &Lambde[0],
      Lambde.data() + Lambde.cols() * Lambde.rows()); 
  std::vector<double> sigma_optimal(&Sigma[0],
                                    Sigma.data() + Sigma.cols() * Lambde.rows());
  std::vector<std::vector<double>> ans{optimal_centers, lambda_optimal, sigma_optimal};
  return ans;
}

/*Perfomer the GRICP*/
RigidTransfo
IcpOptimizer::rigidTransfoGB_poseadjust(vector<GranularBall> firstCloudGB,
                                        vector<GranularBall> secondCloudGB)
{ 
  cout<<"poseadjust"<<endl;
  vector<int> correspondingIndex;
  if (weight_overlap == 0) {
    int size = firstCloudGB.size();
    Matrix<double,Dynamic,3> source;
    Matrix<double,Dynamic,3> target;
    source.conservativeResize(source.rows() + size, Eigen::NoChange);
    target.conservativeResize(target.rows() + size, Eigen::NoChange);
    for (int i = 0; i < firstCloudGB.size(); i++) {
      source.row(i) = firstCloudGB[i].geocenter;
      target.row(i) = secondCloudGB[i].geocenter;
    }
    correspondingIndex = computeCorrespondances(target, source);
  }

  else{
  correspondingIndex = computeCorrespondencesGB(firstCloudGB, secondCloudGB);
  }

  //compute granular balls overlap volume with neighbors
  vector<double> firstCloudGB_overlapvolume = computeGBoverlap(firstCloudGB);
  vector<double> secondCloudGB_overlapvolume = computeGBoverlap(secondCloudGB);
  for(int i=0; i<firstCloudGB.size();i++){
    firstCloudGB[i].overlapvolume = firstCloudGB_overlapvolume[i];
  }
  for(int i=0; i<secondCloudGB.size();i++){
    secondCloudGB[i].overlapvolume = secondCloudGB_overlapvolume[i];
  }

  /*Outlier rejection with granular ball*/ 
  vector<pair<int, int>> PointPairWise;
  // build correspondence

  for (int i = 0; i < firstCloudGB.size(); i++) {
    pair<int, int> PointPair = {i, correspondingIndex[i]};
    /*radius difference*/
    double sourceGranullballRadius = firstCloudGB[i].radius;
    double targetGranullballRadius = secondCloudGB[correspondingIndex[i]].radius;
    double radiusdiff = (sourceGranullballRadius - targetGranullballRadius) /
                             (min(sourceGranullballRadius, targetGranullballRadius)+0.0000001);
   /*overlap difference*/
    double sourceGranullballOverlapVolume = firstCloudGB[i].overlapvolume;
    double targetGranullballOverlapVolume = secondCloudGB[correspondingIndex[i]].overlapvolume;
    double overlapdiff = (sourceGranullballOverlapVolume - targetGranullballOverlapVolume) /
                             (min(sourceGranullballOverlapVolume ,targetGranullballOverlapVolume)+0.00000001);


    if (radiusdiff < tau_1 && overlapdiff< tau_2) {
      PointPairWise.push_back(PointPair);
    }
  }

  cout << "the inlier size is:" << PointPairWise.size() << endl;

  cout << "the computed rotatetion is:" << endl
       << computedTransfo.first << endl
       << endl;
  cout << "the computed translation is:" << endl
       << computedTransfo.second << endl
       << endl;

  vector<int> firstIndices(PointPairWise.size());
  vector<int> secondIndices(PointPairWise.size());
  std::transform(PointPairWise.begin(),
                 PointPairWise.end(),
                 firstIndices.begin(),
                 [](const std::pair<int, int>& pair) { return pair.first; });
  std::transform(PointPairWise.begin(),
                 PointPairWise.end(),
                 secondIndices.begin(),
                 [](const std::pair<int, int>& pair) { return pair.second; });

  vector<GranularBall> firstCloudGB_selected =
      selectSubsetGB(firstCloudGB, firstIndices);

  vector<GranularBall> secondCloudGB_selected =
      selectSubsetGB(secondCloudGB, secondIndices);

  vector<double> errors;
  for (int i = 0; i < firstCloudGB_selected.size(); i++) {
    errors.push_back(firstCloudGB_selected[i].Distance(secondCloudGB_selected[i]));
  }
  vector<vector<double>> mkcparameter =
      MKCParameterEstimator(errors);        // Estimate the error distribution
  vector<double> centers = mkcparameter[0]; // kernel center
  vector<double> Lambde = mkcparameter[1];  // combination weights
  vector<double> Sigma = mkcparameter[2];   // kernel width
  cout << "labmda = ";
  for (int i = 0; i < Lambde.size(); i++) {
    cout << Lambde[i] << " ";
  }
  cout << endl << endl << "Sigma = ";
  for (int i = 0; i < Sigma.size(); i++) {
    cout << Sigma[i] << " ";
  }
  cout << endl << endl;

  /*get solution use SVD*/
  vector<vector<double>> w;
  double W = 0;
  for (int j = 0; j < errors.size(); j++) {
    vector<double> wj;
    for (int i = 0; i < centers.size(); i++) {
      double w_ji = Lambde[i] / (sqrt(2 * pi) * Sigma[i]) *
                    exp(-pow((errors[j] - centers[i]), 2) / (2 * pow(Sigma[i], 2))) *
                    (centers[i] - errors[j]) / pow(Sigma[i], 2);
      double d =
          (firstCloudGB_selected[j].geocenter - secondCloudGB_selected[j].geocenter)
              .norm();
      double r1 = firstCloudGB_selected[j].radius;
      double r2 = secondCloudGB_selected[j].radius;
      if (d <= max(r1, r2) - min(r1, r2) || d >= r1 + r2) {
        w_ji = w_ji * weight_distance / d;
      }
      else {
        double cosalpha = (pow(r1, 2) + pow(d, 2) - pow(r2, 2)) / (2 * r1 * d);
        double cosbeta = (pow(r2, 2) + pow(d, 2) - pow(r1, 2)) / (2 * r2 * d);
        double h1 = r1 - r1 * cosalpha;
        double h2 = r2 - r2 * cosbeta;
        double temp1 = (pow(d, 2) - pow(r1, 2) + pow(r2, 2)) / (2 * r1 * d * d);
        double temp2 = (pow(d, 2) - pow(r2, 2) + pow(r1, 2)) / (2 * r2 * d * d);
        w_ji = w_ji *
               (weight_distance / d +
                weight_overlap * 4 / 3 * pow(max(r1, r2), 3) /
                    pow((pow(h1, 2) * (r1 - h1 / 3) + pow(h2, 2) * (r2 - h2 / 3)), 2) *
                    ((2 * r1 * h1 - pow(h1, 2)) * r1 * temp1 +
                     (2 * r2 * h2 - pow(h2, 2)) * r2 * temp2) /
                    d);
      }
      wj.push_back(w_ji);
      W = W + w_ji;
    }
    w.push_back(wj);
    wj.clear();
  }
  Vector3d sub_xj;
  sub_xj.setConstant(0);
  Vector3d sub_yj;
  sub_yj.setConstant(0);
  for (int j = 0; j < firstCloudGB_selected.size(); j++) {
    for (int i = 0; i < centers.size(); i++) {
      sub_xj += w[j][i] * firstCloudGB_selected[j].geocenter;
      sub_yj += w[j][i] * secondCloudGB_selected[j].geocenter;
    }
  }
  vector<Vector3d> p, q;
  for (int j = 0; j < firstCloudGB_selected.size(); j++) {
    p.push_back((firstCloudGB_selected[j].geocenter - sub_xj) / W);
    q.push_back((secondCloudGB_selected[j].geocenter - sub_yj) / W);
  }
  /*Define a coefficient to merge the two terms in the exponent*/
  vector<vector<double>> k, b;
  for (int j = 0; j < firstCloudGB_selected.size(); j++) {
    double d =
        (firstCloudGB_selected[j].geocenter - secondCloudGB_selected[j].geocenter)
            .norm();
    double r1 = firstCloudGB_selected[j].radius;
    double r2 = secondCloudGB_selected[j].radius;
    vector<double> b_j;
    vector<double> k_j;
    for (int i = 0; i < centers.size(); i++) {
      if (d > r1 + r2) { // not overlap
        b_j.push_back(-centers[i]);
        k_j.push_back(weight_distance);
      }
      else if (d < max(r1, r2) - min(r1, r2)) { // full overlap
        k_j.push_back(weight_distance);
        b_j.push_back(weight_overlap * pow(min(r1, r2), 3) / pow(max(r1, r2), 3) -
                      centers[i]);
      }
      else { // partial overlap
        b_j.push_back(-centers[i]);
        k_j.push_back(weight_distance +
                      weight_overlap /
                          firstCloudGB_selected[j].OverLap(secondCloudGB_selected[j]) /
                          (lastItertransfo.first * p[j] - q[j]).norm());
      }
    }
    b.push_back(b_j);
    k.push_back(k_j);
    k_j.clear();
    b_j.clear();
  }

 
  vector<vector<double>> u;
  vector<double> v;
  for (int j = 0; j < firstCloudGB_selected.size(); j++) {
    vector<double> u_j;
    for (int i = 0; i < centers.size(); i++) {
      u_j.push_back(
          -exp(-pow(k[j][i] * (lastItertransfo.first * p[j] - q[j]).norm() + b[j][i],
                    2) /
               (2 * Sigma[i] * Sigma[i])) *
          Lambde[i] * pow(k[j][i], 2) / (sqrt(2 * pi) * pow(Sigma[i], 3)));
    }
    u.push_back(u_j);
    u_j.clear();
  }
  for (int j = 0; j < firstCloudGB_selected.size(); j++) {
    double vj = 0;
    for (int i = 0; i < centers.size(); i++) {
      vj += u[j][i];
    }
    
    v.push_back(vj);
  }

  MatrixXd Q = MatrixXd::Zero(firstCloudGB_selected.size(), 3);
  MatrixXd P = MatrixXd::Zero(3, firstCloudGB_selected.size());
  MatrixXd V =
      MatrixXd::Zero(firstCloudGB_selected.size(), firstCloudGB_selected.size());
  for (int j = 0; j < firstCloudGB_selected.size(); j++) {
    Q.row(j) = q[j];
    P.col(j) = p[j].transpose();
  }
  for (int j = 0; j < firstCloudGB_selected.size(); j++) {
    V(j, j) = v[j];
  }
  Matrix3d H = P * V * Q;
  /*Decompose H use SVD*/
  // compute the svd of H
  JacobiSVD<Matrix<double, 3, 3>> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
  // compyrte D
  Matrix<double, 3, 3> D = Matrix<double, 3, 3>::Identity(3, 3);
  if (H.determinant() > 0) {
    D = -1 * D;
    D.row(2) = -D.row(2);
  }
  else {
    D = -D;
  }

  // compute rotation
  RotMatrix rotation = svd.matrixV() * D * svd.matrixU().transpose();

  Matrix<double, 3, 1> translation = Matrix<double, 3, 1>::Zero(3, 1);
  for (int j = 0; j < firstCloudGB_selected.size(); j++) {
    for (int i = 0; i < centers.size(); i++) {
      translation =
          translation + w[j][i] * (secondCloudGB_selected[j].geocenter -
                                   rotation * firstCloudGB_selected[j].geocenter);
    }
  }
  translation /= W;
  return RigidTransfo(rotation, translation);
}


PointCloud IcpOptimizer::GBToPointCloud(vector<GranularBall>  GB){
  PointCloud pointcloud;
  int size = GB.size();
  pointcloud.conservativeResize(pointcloud.rows() + size, Eigen::NoChange);
  for (int i = 0; i < GB.size(); i++) {
    pointcloud.row(i) = GB[i].geocenter;
  }
  return pointcloud;
}

/*
Computing composition of tNew by tOld (tNew o tOld)
*/
RigidTransfo
IcpOptimizer::compose(RigidTransfo tNew, RigidTransfo tOld) const
{
  return RigidTransfo(tNew.first * tOld.first, tNew.first * tOld.second + tNew.second);
}

/*
Selects the subset of rows whose index is in indice in the GranularBall set GB.
*/
std::vector<GranularBall>
IcpOptimizer::selectSubsetGB(std::vector<GranularBall> GB, std::vector<int> indices)
{
  std::vector<GranularBall> selection;
  for (int i = 0; i < indices.size(); i++) {
    selection.push_back(GB[i]);
  }
  return selection;
}

/*
Selects the subset of rows whose index is in indice in the Point Cloud p
*/
PointCloud
IcpOptimizer::selectSubsetPC(PointCloud p, vector<int> indice) const
{
  PointCloud selection = PointCloud::Zero(indice.size(), 3);
  for (int i = 0; i < indice.size(); i++)
    selection.row(i) = p.row(indice[i]);
  return selection;
}

/*
Updates the iterations measure by estimating the amplitude of rigid motion t
*/
void
IcpOptimizer::updateIter(RigidTransfo t)
{
  Matrix<double, 4, 4> id = Matrix<double, 4, 4>::Identity();
  Matrix<double, 4, 4> curT = Matrix<double, 4, 4>::Identity();
  curT.block(0, 0, 3, 3) = t.first;
  curT.block(0, 3, 3, 1) = t.second / referenceDist;
  Matrix<double, 4, 4> diff =
      curT - id; // Difference between t and identity
  iterations.push_back((diff * diff.transpose()).trace()); // Returning matrix norm
}

/*
Save iterations to file
*/
void
IcpOptimizer::saveIter(string pathToFile)
{
  ofstream txtStream(pathToFile.c_str());
  for (int i = 0; i < iterations.size(); i++)
    txtStream << iterations[i] << endl;
  txtStream.close();
}

/*
Just a getter to the normals of the first cloud (moving cloud)
*/
Matrix<double, Dynamic, 3>
IcpOptimizer::getFirstNormals() const
{
  return firstNormals;
}

/*
Return a copy of the first point cloud which has been moved by the computed
rigid motion.
If the rigid motion has not been computed it returns just the original first point
cloud.
*/
PointCloud
IcpOptimizer::getMovedNormals() const
{
  if (hasBeenComputed)
    return firstNormals;
  else {
    cout
        << "Warning ! The transformation has not been computed ! Please use the method \
    performSparceICP() before retrieving the moved normals."
        << endl;
    return movingNormals;
  }
}

/*
Return a copy of the first point cloud which has been moved by the computed
rigid motion.
If the rigid motion has not been computed it returns just the original first point
cloud.
*/
PointCloud
IcpOptimizer::getMovedPointCloud() const
{
  if (hasBeenComputed)
    return movingPC;
  else {
    cout
        << "Warning ! The transformation has not been computed ! Please use the method \
    performSparceICP() before retrieving the moved point cloud."
        << endl;
    return firstCloud;
  }
}

/*
Return the computed transformation.
If it has not been computed, just returns the identity.
*/
RigidTransfo
IcpOptimizer::getComputedTransfo() const
{
  if (!hasBeenComputed)
    cout
        << "Warning ! The transformation has not been computed ! Please use the method \
      performSparceICP() before retrieving the rigid motion."
        << endl;
  return computedTransfo;
}

/*
Returns the reference distance which is the length of the great diagonal of the first
point cloud's bounding box.
*/
double
IcpOptimizer::getReferenceDist() const
{
  return referenceDist;
}

double
IcpOptimizer::computeRMSE(PointCloud a, PointCloud b)
{
  double rmse = 0;
  for (int i = 0; i < a.rows(); i++) {
    rmse = rmse + pow((a.row(i) - b.row(i)).norm(), 2);
  }
  return rmse / a.rows();
}


vector<double>
IcpOptimizer::computedTransformationError(RigidTransfo estimation, RigidTransfo truth)
{
  vector<double> errors;
  errors.push_back(
      acos(((truth.first * estimation.first.transpose()).trace() - 1) / 2));
  errors.push_back((estimation.second - truth.second).lpNorm<1>());
  return errors;
}
void
IcpOptimizer::saveMetrics(std::string pathToFile, vector<double> metrics)
{
  ofstream txtStream(pathToFile.c_str());
  for (int i = 0; i < metrics.size(); i++)
    txtStream << metrics[i] << endl;
  txtStream.close();
}

// Parse the input path to obtain additional paths
pair<string, string>
IcpOptimizer::ExtractFileNameAndParentDir(const string& filePath)
{
  
  stringstream ss(filePath);
  string item;
  string fileName, parentDir;

 
  while (getline(ss, item, '/')) {
    parentDir = fileName;
    fileName = item;
  }
 
  size_t dotIndex = fileName.find_last_of('.');
  if (dotIndex != string::npos) {
    fileName = fileName.substr(0, dotIndex);
  }
  return make_pair(fileName, parentDir);
}

GranularBall::GranularBall(Eigen::Vector3d Geocenter, double Radius) // Initial Granular ball
{
  this->geocenter = Geocenter;
  this->radius = Radius;
}

GranularBall::GranularBall() 
{}



double 
GranularBall::OverlapRegion(GranularBall other){
  double d = (this->geocenter - other.geocenter).norm();
  if (d > this->radius + other.radius) {
    return 0;
  }
  else if (d < max(this->radius, other.radius) - min(this->radius, other.radius)) {
    return pow(min(this->radius, other.radius), 3);
  }
  else {
    double cosAlpha =
        (pow(this->radius, 2) + d - pow(other.radius, 2)) / (2 * this->radius * d);
    double cosBeta =
        (pow(other.radius, 2) + d - pow(this->radius, 2)) / (2 * other.radius * d);
    double h_1 = this->radius - this->radius * cosAlpha;
    double h_2 = other.radius - other.radius * cosBeta;
    double overlapsqure = pi * pow(h_1, 2) * (this->radius - h_1 / 3) +
                          pi * pow(h_2, 2) * (other.radius - h_2 / 3);
    return overlapsqure;
  }
}
double
GranularBall::OverLap(GranularBall other) // Compute Overlap rate
{
  double d = (this->geocenter - other.geocenter).norm();
  if (d > this->radius + other.radius) {
    return 0;
  }
  else if (d < max(this->radius, other.radius) - min(this->radius, other.radius)) {
    return pow(min(this->radius, other.radius), 3) /
           pow(max(this->radius, other.radius), 3);
  }
  else {
    double cosAlpha =
        (pow(this->radius, 2) + d - pow(other.radius, 2)) / (2 * this->radius * d);
    double cosBeta =
        (pow(other.radius, 2) + d - pow(this->radius, 2)) / (2 * other.radius * d);
    double h_1 = this->radius - this->radius * cosAlpha;
    double h_2 = other.radius - other.radius * cosBeta;
    double overlapsqure = pi * pow(h_1, 2) * (this->radius - h_1 / 3) +
                          pi * pow(h_2, 2) * (other.radius - h_2 / 3);
    double overlaprate =
        overlapsqure / (pow(max(this->radius, other.radius), 3) * pi * 4 / 3);
    return overlaprate;
  }
}
vector<GranularBall>
GranularBall::loadGranularBall(Eigen::Matrix<double, Eigen::Dynamic, 3> pointcloud,
                               vector<double> Radius)
{ // TRansfer Point Cloud to Granular Ball
  vector<GranularBall> GranularBalls;
  for (int i = 0; i < pointcloud.rows(); i++) {
    Eigen::Vector3d geocenter = pointcloud.row(i);
    double radius = Radius[i];
    GranularBalls.push_back(GranularBall(geocenter, radius));
  }
  return GranularBalls;
}

double
GranularBall::Distance(GranularBall& other)
{
  double dist_geocenter = (geocenter - other.geocenter).norm();
  double overlapsqure = OverLap(other);
  double overlaprate;
  if (overlapsqure == 0) {
    overlaprate = 0;
  }
  else {
    overlaprate = 1 / overlapsqure;
  }
  return weight_distance * dist_geocenter + weight_overlap * overlaprate;
}
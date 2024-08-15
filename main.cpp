#include <Eigen/Core>

#include "IcpOptimizer.h"
#include "ObjLoader.h"
#include "option_parser.h"
#include <nanoflann.hpp>

#include <iostream>

using namespace nanoflann;
using namespace std;
using namespace Eigen;

#include <Eigen/Core>

#include <nanoflann.hpp>

#include <iostream>
#include <vector>

using namespace std;
using namespace nanoflann;


pair<string, string> extractFileNameAndParentDir(const string& filePath) {
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
int main(int argc, char* argv[])
{
  //Create parsing options
  op::OptionParser opt;
  opt.add_option("-h", "--help", "show option help");
  opt.add_option("-d","--demo", "demo mode (uses integrated media)");
  opt.add_option("-i1","--input_1","Path to first input obj file (REQUIRED)","");
  opt.add_option("-i2","--input_2","Path to second input obj file (REQUIRED)","");
  opt.add_option("-o","--output","Path to the output directory (REQUIRED)","");
  opt.add_option("-n","--name","Name of the output file","output");
  opt.add_option("-k", "--k_normals", "knn parameter for normals computation", "10"
  ); opt.add_option("-n1", "--n_iterations_1","Nb of iterations for the algorithm","25"); 
  opt.add_option("-n2", "--n_iterations_2","Nb of iterations for the algorithm's step 2","2"); 
  opt.add_option("-mu","--mu","Parameter for step 2.1","10"); 
  opt.add_option("-ns","--n_iterations_shrink","Number of iterations for shrink step (2.1)","3"); 
  opt.add_option("-p","--p_norm","Use of norm L_p","0.5"); 
  opt.add_option("-po","--point_to_point","Use point to point variant");
  opt.add_option("-pl","--point_to_plane","Use point to plane variant");
  opt.add_option("-v","--verbose","Verbosity trigger");

  //Parsing options
  bool correctParsing = opt.parse_options(argc, argv);
  if(!correctParsing)
    return EXIT_FAILURE;

  //Parameters
  const string first_path = opt["-i1"];
  const string second_path = opt["-i2"];
  string output_path = opt["-o"];
  size_t kNormals = op::str2int(opt["-k"]);
  const int nbIterations = op::str2int(opt["-n1"]);
  const int nbIterationsIn = op::str2int(opt["-n2"]);
  const double mu = op::str2double(opt["-mu"]);
  const int nbIterShrink = op::str2int(opt["-ns"]);
  const double p = op::str2double(opt["-p"]);
  const bool verbose = op::str2bool(opt["-v"]);
  const bool demoMode = op::str2bool(opt["-d"]);
  const bool hasHelp  =  op::str2bool(opt["-h"]);

  const bool isPointToPoint = op::str2bool(opt["-po"]);
  const bool isPointToPlane = op::str2bool(opt["-pl"]);

  //Making checks


  if(hasHelp)
  {
    opt.show_help();
    return 0;
  }

  if(first_path == "")
  {
    cerr << "Please specify the path of the first object file." << endl;
    opt.show_help();
    return EXIT_FAILURE;
  }

  if(second_path == "")
  {
    cerr << "Please specify the path of the second object file." << endl;
    opt.show_help();
    return EXIT_FAILURE;
  }

  if(output_path == "")
  {
    cerr << "Please specify the path of the output directory." << endl;
    opt.show_help();
    return EXIT_FAILURE;
  }

 //parsed the result save path
  pair<string, string> extracted_out_path_moved_first_pc = extractFileNameAndParentDir(first_path); 
  pair<string, string> extracted_out_path_second_pc = extractFileNameAndParentDir(second_path);
  if(output_path[output_path.size()-1] != '/')
    output_path.append("/");
  string output_moved_first_pc_path = output_path + "/" + extracted_out_path_moved_first_pc.second + "/" + extracted_out_path_second_pc.first+ "_moved" +".ply";
  string output_second_pc_path = output_path + "/" + extracted_out_path_second_pc.second + "/" + extracted_out_path_second_pc.first+".ply";

  cout << "output_moved_first_pc_path: " << output_moved_first_pc_path << endl;
  cout << "output_second_pc_path: " << output_second_pc_path << endl;

  if(isPointToPlane && isPointToPoint)
  {
    cerr << "Please choose only one ICP method !" << endl;
    opt.show_help();
    return EXIT_FAILURE;
  }

  IcpMethod method = pointToPoint;

  if(isPointToPlane)
    method = pointToPlane;
  else if (isPointToPoint)
    method = pointToPoint;
  else
  {
    cerr << "Please choose at least one ICP method (point to point or point to plane)." << endl; opt.show_help(); return EXIT_FAILURE;
  }

  //Loading the point clouds
  ObjectLoader myLoader;
  Matrix<double,Dynamic,3> pointCloudOne = myLoader(first_path);
  Matrix<double,Dynamic,3> pointCloudTwo = myLoader(second_path);
  string path_gb = "../data_gb/";
  string path_granuleball = "../granuleball_info/";

  pair<string, string> extracted_first_path = extractFileNameAndParentDir(first_path); 
  string first_fileName = extracted_first_path.first; 
  string first_parentDir = extracted_first_path.second;

  pair<string, string> extracted_second_path = extractFileNameAndParentDir(second_path); 
  string second_fileName = extracted_second_path.first; 
  string second_parentDir = extracted_second_path.second;
  
  string first_path_gb = path_gb + first_parentDir + "/" + first_fileName + ".obj";
  string first_path_granuleball_num = path_granuleball + first_parentDir + "/" + first_fileName + "_pointNum.txt"; string first_path_granuleball_radius = path_granuleball + first_parentDir + "/" + first_fileName + "_radius.txt"; 
  cout << "first_path_gb: " << first_path_gb << endl; 
  cout << "first_path_granuleball_num: "<< first_path_granuleball_num << endl; 
  cout << "first_path_granuleball_radius: " << first_path_granuleball_radius << endl;


  string second_path_gb = path_gb + second_parentDir + "/" + second_fileName + ".obj"; 
  string second_path_granuleball_num = path_granuleball + second_parentDir + "/" + second_fileName + "_pointNum.txt"; 
  string second_path_granuleball_radius = path_granuleball + second_parentDir + "/" + second_fileName + "_radius.txt"; 
  cout<< "second_path_gb: " << second_path_gb << endl; cout << "second_path_granuleball_num: " << second_path_granuleball_num << endl; 
  cout <<"second_path_granuleball_radius: " << second_path_granuleball_radius << endl;

   //Loading the point clouds
  pair<string, string> first_point_cloud_granular_info_path = make_pair(first_path_granuleball_num, first_path_granuleball_radius);//第一个点云粒球信息 
  pair<string, string> second_point_cloud_granular_info_path = make_pair(second_path_granuleball_num, second_path_granuleball_radius);//第二个点云粒球信息 
  pair<string, string> point_cloud_gb_path = make_pair(first_path_gb, second_path_gb);//粒球化后的点云路径

  //Creating an IcpOptimizer in order to perform the sparse icp
  IcpOptimizer myIcpOptimizer(pointCloudOne,pointCloudTwo,point_cloud_gb_path,first_point_cloud_granular_info_path,second_point_cloud_granular_info_path,kNormals,nbIterations,nbIterationsIn,mu,nbIterShrink,p,method,verbose);
  

  //Perform ICP
  int hasIcpFailed = myIcpOptimizer.performSparceICP();
  if(hasIcpFailed)
  {
    cerr << "Failed to load the point clouds. Check the paths." << endl;
    return EXIT_FAILURE;
  }
  PointCloud resultingCloud = myIcpOptimizer.getMovedPointCloud();

  //save the point cloud after registration
  myLoader.dumpToFile(resultingCloud,myIcpOptimizer.getMovedNormals(),output_moved_first_pc_path);
  myLoader.dumpToFile(pointCloudTwo,myIcpOptimizer.getMovedNormals(),output_second_pc_path);

  //Show resulting transformation
  RigidTransfo resultingTransfo = myIcpOptimizer.getComputedTransfo();
  cout <<endl<< "The final Computed Rotation : " << endl << resultingTransfo.first << endl <<
  "The final Computed Translation : " << endl << resultingTransfo.second << endl;

  return 0;
}

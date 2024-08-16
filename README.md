# GRICP: Granular-Ball Iterative Closest Point with Multikernel Correntropy for Point Cloud Fine Registration #

This repository contains an implementation of the [GRICP: Granular-Ball Iterative Closest Point with MultiKernel Correntropy for Point Cloud Fine Registration](). 

## Dependencies ##

Most dependencies are header-only and are included in the ext directory. However, you need to ensure that Boost and Shark are installed in your environment. You can do this by running the following commands:


* [nanoflann](https://github.com/jlblancoc/nanoflann)
* [Eigen](http://eigen.tuxfamily.org/index.php?title=Main\_Page)
* [Boost](sudo apt-get install libboost-all-dev)
* [Shark](git clone https://github.com/Shark-ML/Shark.git)
## Usage ##

Steps to Run Code:
Create a new directory build.
Navigate to the build directory and run `cmake ..` .. followed by `make`.
Execute `/GRICP -i1 /path/to/first/objectFile.obj -i2 ../path/to/second/objectFile.obj -o /path/to/Registration_Result/ -po`


## Granular Ball ##
You can convert the point cloud into a granular ball cloud by running `HB_Test_v3.py`in `Granular_Ball` folder, which will also save radius and other related information to the `granuleball_info/` directory.


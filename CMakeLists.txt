cmake_minimum_required(VERSION 2.8)
project(GRICP)


#####################################################################
#           Boost configuration
#####################################################################
set(Boost_USE_STATIC_LIBS OFF CACHE BOOL "use static libraries from Boost")
set(Boost_USE_MULTITHREADED ON)
add_definitions(-DBOOST_PARAMETER_MAX_ARITY=15)
add_definitions(-DBOOST_FILESYSTEM_VERSION=3)
add_definitions(-DBOOST_RESULT_OF_USE_DECLTYPE)

# Should we link the boost test dynamically
if(NOT Boost_USE_STATIC_LIBS)
	add_definitions(-DBOOST_TEST_DYN_LINK)
	add_definitions(-DBOOST_ALL_DYN_LINK)
endif()

find_package( 
	Boost 1.48.0 REQUIRED COMPONENTS
	serialization
	filesystem system
)

if(NOT Boost_FOUND)
	message(FATAL_ERROR "Please make sure Boost 1.48.0 is installed on your system")
endif()

if (WIN32)
	# disable autolinking in boost
	add_definitions( -DBOOST_ALL_NO_LIB )
endif()

include_directories(SYSTEM ${Boost_INCLUDE_DIR} )
link_directories( ${Boost_LIBRARY_DIR} )

# Set the libraries needed by Shark
list(APPEND LINK_LIBRARIES ${Boost_LIBRARIES})
find_package(Shark REQUIRED)
include(${SHARK_USE_FILE})

# Set the Cmake version
set( CMAKE_EXPORT_COMPILE_COMMANDS ON )
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2")

#Adding reference to the media directory
add_definitions(-DICPSPARSE_MEDIA_DIR="\\"${CMAKE_CURRENT_SOURCE_DIR}/media/\\"")

#Adding current directory for include
include_directories(${CMAKE_CURRENT_LIST_DIR})

#Adding OptionParser
include_directories(${CMAKE_CURRENT_LIST_DIR}/ext/OptionParser)

#Adding Eigen
include_directories(${CMAKE_CURRENT_LIST_DIR}/ext/Eigen)

#Adding Nanoflann
include_directories(${CMAKE_CURRENT_LIST_DIR}/ext/NanoFlann)

#Adding projects headers
include_directories(${CMAKE_CURRENT_LIST_DIR}/lib/ObjLoader)
include_directories(${CMAKE_CURRENT_LIST_DIR}/lib/IcpOptimizer)

#Adding projects libraries
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/lib/ObjLoader)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/lib/IcpOptimizer)



add_executable(GRICP main.cpp)
target_link_libraries(GRICP ObjectLoader)
target_link_libraries(GRICP IcpOptimizer)
target_link_libraries(GRICP ${SHARK_LIBRARIES})




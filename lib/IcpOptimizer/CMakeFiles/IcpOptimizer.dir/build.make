# CMAKE generated file: DO NOT EDIT!
# Generated by "MinGW Makefiles" Generator, CMake Version 3.24

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = E:\Cmake\bin\cmake.exe

# The command to remove a file.
RM = E:\Cmake\bin\cmake.exe -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = E:\SignalProccessing\ExperimentCode\icpSparse_MEE_MCC

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = E:\SignalProccessing\ExperimentCode\icpSparse_MEE_MCC

# Include any dependencies generated for this target.
include lib/IcpOptimizer/CMakeFiles/IcpOptimizer.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include lib/IcpOptimizer/CMakeFiles/IcpOptimizer.dir/compiler_depend.make

# Include the progress variables for this target.
include lib/IcpOptimizer/CMakeFiles/IcpOptimizer.dir/progress.make

# Include the compile flags for this target's objects.
include lib/IcpOptimizer/CMakeFiles/IcpOptimizer.dir/flags.make

lib/IcpOptimizer/CMakeFiles/IcpOptimizer.dir/IcpOptimizer.cpp.obj: lib/IcpOptimizer/CMakeFiles/IcpOptimizer.dir/flags.make
lib/IcpOptimizer/CMakeFiles/IcpOptimizer.dir/IcpOptimizer.cpp.obj: lib/IcpOptimizer/CMakeFiles/IcpOptimizer.dir/includes_CXX.rsp
lib/IcpOptimizer/CMakeFiles/IcpOptimizer.dir/IcpOptimizer.cpp.obj: lib/IcpOptimizer/IcpOptimizer.cpp
lib/IcpOptimizer/CMakeFiles/IcpOptimizer.dir/IcpOptimizer.cpp.obj: lib/IcpOptimizer/CMakeFiles/IcpOptimizer.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=E:\SignalProccessing\ExperimentCode\icpSparse_MEE_MCC\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object lib/IcpOptimizer/CMakeFiles/IcpOptimizer.dir/IcpOptimizer.cpp.obj"
	cd /d E:\SignalProccessing\ExperimentCode\icpSparse_MEE_MCC\lib\IcpOptimizer && E:\mingw64\bin\c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT lib/IcpOptimizer/CMakeFiles/IcpOptimizer.dir/IcpOptimizer.cpp.obj -MF CMakeFiles\IcpOptimizer.dir\IcpOptimizer.cpp.obj.d -o CMakeFiles\IcpOptimizer.dir\IcpOptimizer.cpp.obj -c E:\SignalProccessing\ExperimentCode\icpSparse_MEE_MCC\lib\IcpOptimizer\IcpOptimizer.cpp

lib/IcpOptimizer/CMakeFiles/IcpOptimizer.dir/IcpOptimizer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/IcpOptimizer.dir/IcpOptimizer.cpp.i"
	cd /d E:\SignalProccessing\ExperimentCode\icpSparse_MEE_MCC\lib\IcpOptimizer && E:\mingw64\bin\c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E E:\SignalProccessing\ExperimentCode\icpSparse_MEE_MCC\lib\IcpOptimizer\IcpOptimizer.cpp > CMakeFiles\IcpOptimizer.dir\IcpOptimizer.cpp.i

lib/IcpOptimizer/CMakeFiles/IcpOptimizer.dir/IcpOptimizer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/IcpOptimizer.dir/IcpOptimizer.cpp.s"
	cd /d E:\SignalProccessing\ExperimentCode\icpSparse_MEE_MCC\lib\IcpOptimizer && E:\mingw64\bin\c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S E:\SignalProccessing\ExperimentCode\icpSparse_MEE_MCC\lib\IcpOptimizer\IcpOptimizer.cpp -o CMakeFiles\IcpOptimizer.dir\IcpOptimizer.cpp.s

# Object files for target IcpOptimizer
IcpOptimizer_OBJECTS = \
"CMakeFiles/IcpOptimizer.dir/IcpOptimizer.cpp.obj"

# External object files for target IcpOptimizer
IcpOptimizer_EXTERNAL_OBJECTS =

lib/IcpOptimizer/libIcpOptimizer.a: lib/IcpOptimizer/CMakeFiles/IcpOptimizer.dir/IcpOptimizer.cpp.obj
lib/IcpOptimizer/libIcpOptimizer.a: lib/IcpOptimizer/CMakeFiles/IcpOptimizer.dir/build.make
lib/IcpOptimizer/libIcpOptimizer.a: lib/IcpOptimizer/CMakeFiles/IcpOptimizer.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=E:\SignalProccessing\ExperimentCode\icpSparse_MEE_MCC\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libIcpOptimizer.a"
	cd /d E:\SignalProccessing\ExperimentCode\icpSparse_MEE_MCC\lib\IcpOptimizer && $(CMAKE_COMMAND) -P CMakeFiles\IcpOptimizer.dir\cmake_clean_target.cmake
	cd /d E:\SignalProccessing\ExperimentCode\icpSparse_MEE_MCC\lib\IcpOptimizer && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles\IcpOptimizer.dir\link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
lib/IcpOptimizer/CMakeFiles/IcpOptimizer.dir/build: lib/IcpOptimizer/libIcpOptimizer.a
.PHONY : lib/IcpOptimizer/CMakeFiles/IcpOptimizer.dir/build

lib/IcpOptimizer/CMakeFiles/IcpOptimizer.dir/clean:
	cd /d E:\SignalProccessing\ExperimentCode\icpSparse_MEE_MCC\lib\IcpOptimizer && $(CMAKE_COMMAND) -P CMakeFiles\IcpOptimizer.dir\cmake_clean.cmake
.PHONY : lib/IcpOptimizer/CMakeFiles/IcpOptimizer.dir/clean

lib/IcpOptimizer/CMakeFiles/IcpOptimizer.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" E:\SignalProccessing\ExperimentCode\icpSparse_MEE_MCC E:\SignalProccessing\ExperimentCode\icpSparse_MEE_MCC\lib\IcpOptimizer E:\SignalProccessing\ExperimentCode\icpSparse_MEE_MCC E:\SignalProccessing\ExperimentCode\icpSparse_MEE_MCC\lib\IcpOptimizer E:\SignalProccessing\ExperimentCode\icpSparse_MEE_MCC\lib\IcpOptimizer\CMakeFiles\IcpOptimizer.dir\DependInfo.cmake --color=$(COLOR)
.PHONY : lib/IcpOptimizer/CMakeFiles/IcpOptimizer.dir/depend


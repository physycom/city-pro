#! /usr/bin/env bash

mkdir -p build_linux
cd build_linux
cmake .. -DCMAKE_BUILD_TYPE=Release 
#cmake -G Ninja -DCMAKE_BUILD_TYPE=Debug -DVALGRIND=ON ..
cmake --build . --parallel 8
cd ..
# I have added include (CTest) in CmakeLists.txt https://stackoverflow.com/questions/40325957/how-do-i-add-valgrind-tests-to-my-cmake-test-target
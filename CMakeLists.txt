cmake_minimum_required (VERSION 3.1)
#do not lower cmake due to Threads requirements!
project(venice)

if (EXISTS "$ENV{WORKSPACE}/sysconfig/cmake/physycom_config.cmake")
   message(STATUS "Enabling physycom settings")
   include("$ENV{WORKSPACE}/sysconfig/cmake/physycom_config.cmake")
else()
   message(STATUS "Unable to find physycom settings file")
endif()
# VALGRIND SETUP
#include (CTest)
#enable_testing()
#
### Set output directories on a per-configuration base
# Single configuration
# SETTING CLANG AS COMPILER
# SET (CLANG_DIR "$ENV{WORKSPACE}/city-pro/source/llvm-project")
# SET (CMAKE_C_COMPILER             "${CLANG_DIR}/build/bin/clang")
# SET (CMAKE_C_FLAGS                "-Wall -std=c99")
# SET (CMAKE_C_FLAGS_DEBUG          "-g")
# SET (CMAKE_C_FLAGS_MINSIZEREL     "-Os -DNDEBUG")
# SET (CMAKE_C_FLAGS_RELEASE        "-O4 -DNDEBUG")
# SET (CMAKE_C_FLAGS_RELWITHDEBINFO "-O2 -g")

# SET (CMAKE_CXX_COMPILER             "${CLANG_DIR}/build/bin/clang++")
# SET (CMAKE_CXX_FLAGS                "-Wall")
# SET (CMAKE_CXX_FLAGS_DEBUG          "-g")
# SET (CMAKE_CXX_FLAGS_MINSIZEREL     "-Os -DNDEBUG")
# SET (CMAKE_CXX_FLAGS_RELEASE        "-O4 -DNDEBUG")
# SET (CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")

# SET (CMAKE_AR      "${CLANG_DIR}/build/bin/llvm-ar")
# SET (CMAKE_LINKER  "${CLANG_DIR}/build/bin/llvm-ld")
# SET (CMAKE_NM      "${CLANG_DIR}/build/bin/llvm-nm")
# SET (CMAKE_OBJDUMP "${CLANG_DIR}/build/bin/llvm-objdump")
# SET (CMAKE_RANLIB  "${CLANG_DIR}/build/bin/llvm-ranlib")
#

set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin )
# Multi configuration
foreach( OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES} )
    string( TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG )
    set( CMAKE_RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_SOURCE_DIR}/bin )
endforeach( OUTPUTCONFIG CMAKE_CONFIGURATION_TYPES )

if(UNIX AND NOT APPLE)
  set(OpenGL_GL_PREFERENCE LEGACY) # for compatibility, see https://cmake.org/cmake/help/v3.10/module/FindOpenGL.html
  find_package(LibRt REQUIRED)
endif()
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
find_package(FLTK REQUIRED)
find_package(Boost COMPONENTS filesystem REQUIRED)
find_package(OpenGL REQUIRED)
find_package(Eigen3 REQUIRED)

set(UTILS_DIR "${CMAKE_SOURCE_DIR}/source/utils")
set(SOURCE_DIR "${CMAKE_SOURCE_DIR}/source")
set(RAPIDXML_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/source/rapidxml)
set(JSONCONS_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/source/jsoncons/include)

include_directories(${EIGEN3_INCLUDE_DIR}
                    ${UTILS_DIR}
                    ${SOURCE_DIR}
                    ${OPENGL_INCLUDE_DIRS}
                    ${FLTK_INCLUDE_DIR}
                    ${RAPIDXML_INCLUDE_DIR}
                    ${JSONCONS_INCLUDE_DIR}
                    ${Boost_INCLUDE_DIR}
                    ${CMAKE_SOURCE_DIR}/source/ale
                    )

# creating folder structure
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/input/cartography)
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/input/data)
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/output)
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/output/weights)

# List of projects
add_library(city-pro_lib            ${CMAKE_SOURCE_DIR}/source/carto.cpp
                                    ${CMAKE_SOURCE_DIR}/source/config.cpp
                                    ${CMAKE_SOURCE_DIR}/source/data_analysis.cpp
                                    ${CMAKE_SOURCE_DIR}/source/data_reading.cpp
                                    ${CMAKE_SOURCE_DIR}/source/draw.cpp
                                    ${CMAKE_SOURCE_DIR}/source/form.cpp
                                    ${CMAKE_SOURCE_DIR}/source/frame.cpp
                                    ${CMAKE_SOURCE_DIR}/source/gltrans.cpp
                                    ${CMAKE_SOURCE_DIR}/source/record.cpp
#                                    ${CMAKE_SOURCE_DIR}/source/analyzer_object.cpp

									)
target_link_libraries(city-pro_lib
  ${OPENGL_glu_LIBRARY}
  ${OPENGL_gl_LIBRARY}
  ${OPENGL_LIBRARIES}
  ${FLTK_LIBRARIES}
  Threads::Threads
  ${Boost_LIBRARIES}
)

add_executable(city-pro             ${CMAKE_SOURCE_DIR}/source/main_city-pro.cpp)

target_link_libraries(city-pro      city-pro_lib)



add_executable(city-pro_subnet           ${CMAKE_SOURCE_DIR}/source/main_city-pro_subnet.cpp)
target_link_libraries(city-pro_subnet    Threads::Threads)
# VALGRIND TEST
#add_test(NAME my_app_valgrind_test COMMAND ${CMAKE_CTEST_COMMAND}
#         --force-new-ctest-process --test-action memcheck
#         --label-exclude no_valgrind
#         --build-config $<CONFIGURATION>
#         --start-with-test city-pro)
#set_tests_properties(my_app_valgrind_test PROPERTIES LABELS valgrind)# Add linking dependencies
#
if(UNIX AND NOT APPLE)
  target_link_libraries(city-pro          LIBRT::LIBRT )
  target_link_libraries(city-pro_subnet   LIBRT::LIBRT )
endif()

if(UNIX)
  target_link_libraries(city-pro          ${CMAKE_DL_LIBS} )
  target_link_libraries(city-pro_subnet   ${CMAKE_DL_LIBS} )
endif()



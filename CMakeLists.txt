cmake_minimum_required (VERSION 3.16)

if(DEFINED ENV{VCPKG_DEFAULT_TRIPLET})
  message(STATUS "Setting default vcpkg target triplet to $ENV{VCPKG_DEFAULT_TRIPLET}")
  set(VCPKG_TARGET_TRIPLET $ENV{VCPKG_DEFAULT_TRIPLET})
endif()

# If you are in windows environment (initial configuration for the project)
# If it is not installed, install vcpkg and set the important paths about it. (vpkg.cmake)

if(DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
  set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
  set(X_VCPKG_APPLOCAL_DEPS_INSTALL ON)
  message(STATUS "VCPKG found: $ENV{VCPKG_ROOT}")
  message(STATUS "Using VCPKG integration")
  set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
  # Adds the vcpkg installed packages to the search path
  include("$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")
  if(VCPKG_MANIFEST_FEATURES)
    message(STATUS "VCPKG_MANIFEST_FEATURES: ${VCPKG_MANIFEST_FEATURES}")
  endif()
# Use the vcpkg toolchain file if it is defined
elseif(DEFINED CMAKE_TOOLCHAIN_FILE)
  message(STATUS "Using toolchain: ${CMAKE_TOOLCHAIN_FILE}")
  if(CMAKE_TOOLCHAIN_FILE MATCHES "vcpkg.cmake")
    message(STATUS "Toolchain uses VCPKG integration")
    if(VCPKG_MANIFEST_FEATURES)
      message(STATUS "VCPKG_MANIFEST_FEATURES: ${VCPKG_MANIFEST_FEATURES}")
    endif()
  endif()
else()
  message(WARNING "vcpkg not found, toolchain not defined, build might fail")
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/vcpkg.json)
  file(READ ${CMAKE_CURRENT_SOURCE_DIR}/vcpkg.json VCPKG_JSON_STRING)
  string(JSON VERSION_STRING GET ${VCPKG_JSON_STRING} version)
else()
  set(VERSION_STRING "0.0.0")
endif()

project(venice VERSION ${VERSION_STRING})

enable_language(C)
enable_language(CXX)

set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
      STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}" CACHE PATH "Install prefix" FORCE)
endif()

set(INSTALL_BIN_DIR      "bin"                   CACHE PATH "Path where exe and dll will be installed")
set(INSTALL_LIB_DIR      "lib"                   CACHE PATH "Path where lib will be installed")
set(INSTALL_INCLUDE_DIR  "include"               CACHE PATH "Path where headers will be installed ")
set(INSTALL_CMAKE_DIR    "share/${PROJECT_NAME}" CACHE PATH "Path where cmake configs will be installed")

# VALGRIND SETUP
#include (CTest)
#enable_testing()

if(WIN32)
  add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
endif()

if(APPLE)
  add_compile_definitions(GL_SILENCE_DEPRECATION)
endif()

if(UNIX AND NOT APPLE)
  set(OpenGL_GL_PREFERENCE LEGACY) # for compatibility, see https://cmake.org/cmake/help/v3.10/module/FindOpenGL.html
endif()
find_library(LIBRT rt)
set(THREADS_PREFER_PTHREAD_FLAG ON)
# JSONCONS with ./vcpkg install jsoncons
if(DEFINED ENV{VCPKG_DEFAULT_TRIPLET})
  set(JSONCONS_PATH "$ENV{VCPKG_ROOT}/packages/jsoncons_$ENV{VCPKG_DEFAULT_TRIPLET}/include")
  set(PHYSYCOM_UTILS_PATH "$ENV{VCPKG_ROOT}/packages/physycom-utils_$ENV{VCPKG_DEFAULT_TRIPLET}/include")
  include_directories(${JSONCONS_PATH})
  include_directories(${PHYSYCOM_UTILS_PATH})
  message(STATUS "Using jsoncons from ${JSONCONS_PATH}")
  message(STATUS "Using physycom-utils from ${PHYSYCOM_UTILS_PATH}")
endif()
find_package(Threads REQUIRED)
find_package(FLTK CONFIG REQUIRED)
find_package(Boost COMPONENTS filesystem REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(physycom-utils REQUIRED)
# JSONCONS with ./vcpkg install jsoncons
find_package(jsoncons CONFIG REQUIRED)
# PYBIND INSTALLATION VIA VCPKG
find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)
#add_subdirectory($ENV{WORKSPACE}/pybind11 pybind11)


include_directories(${CMAKE_SOURCE_DIR}/source
                    ${CMAKE_SOURCE_DIR}/source/ale)

# creating folder structure
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/input/cartography)
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/input/data)
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/output)
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/output/weights)

# List of projects
add_library(city-pro_lib ${CMAKE_SOURCE_DIR}/source/carto.cpp
                         ${CMAKE_SOURCE_DIR}/source/config.cpp
                         ${CMAKE_SOURCE_DIR}/source/data_analysis.cpp
                         ${CMAKE_SOURCE_DIR}/source/data_reading.cpp
                         ${CMAKE_SOURCE_DIR}/source/draw.cpp
                         ${CMAKE_SOURCE_DIR}/source/form.cpp
                         ${CMAKE_SOURCE_DIR}/source/frame.cpp
                         ${CMAKE_SOURCE_DIR}/source/gltrans.cpp
                         ${CMAKE_SOURCE_DIR}/source/record.cpp
                         #${CMAKE_SOURCE_DIR}/source/analyzer_object.cpp
)

target_link_libraries(city-pro_lib
  OpenGL::GL
  OpenGL::GLU
  fltk fltk_gl fltk_forms fltk_images
  Threads::Threads
  Boost::headers
  Boost::filesystem
  Eigen3::Eigen
  Physycom::utils
)
add_executable(city-pro             ${CMAKE_SOURCE_DIR}/source/main_city-pro.cpp)
target_include_directories(city-pro PRIVATE "${JSONCONS_PATH}")
target_include_directories(city-pro PRIVATE "${PHYSYCOM_UTILS_PATH}")
target_link_libraries(city-pro      city-pro_lib)

add_executable(city-pro_subnet           ${CMAKE_SOURCE_DIR}/source/main_city-pro_subnet.cpp)
target_include_directories(city-pro_subnet PRIVATE "${JSONCONS_PATH}")
target_include_directories(city-pro_subnet PRIVATE "${PHYSYCOM_UTILS_PATH}")

target_link_libraries(city-pro_subnet    
                      city-pro_lib
                      )

set_property(TARGET city-pro_lib PROPERTY CXX_STANDARD 14)
set_property(TARGET city-pro PROPERTY CXX_STANDARD 14)
set_property(TARGET city-pro_subnet PROPERTY CXX_STANDARD 14)

# VALGRIND TEST
#add_test(NAME my_app_valgrind_test COMMAND ${CMAKE_CTEST_COMMAND}
#         --force-new-ctest-process --test-action memcheck
#         --label-exclude no_valgrind
#         --build-config $<CONFIGURATION>
#         --start-with-test city-pro)
#set_tests_properties(my_app_valgrind_test PROPERTIES LABELS valgrind)# Add linking dependencies


#pybind11_add_module(pycitypro ${CMAKE_CURRENT_LIST_DIR}/source/python_interface.cpp)

if(LIBRT)
  target_link_libraries(city-pro ${LIBRT})
  target_link_libraries(city-pro_subnet ${LIBRT})
endif()

if(UNIX)
  target_link_libraries(city-pro          ${CMAKE_DL_LIBS} )
  target_link_libraries(city-pro_subnet   ${CMAKE_DL_LIBS} )
endif()

install(TARGETS city-pro city-pro_subnet
  RUNTIME DESTINATION "${INSTALL_BIN_DIR}"
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}"
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
)

set(CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS_SKIP TRUE)
include(InstallRequiredSystemLibraries)

install(
    PROGRAMS ${CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS}
    DESTINATION ${INSTALL_BIN_DIR}
)

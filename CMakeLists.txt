cmake_minimum_required(VERSION 3.15)
project(Abstraction)

#set (Python_TARGET_VERSION "" CACHE STRING "Specify a target Python version. \
#Any Python library found with CMake modules FindPython2/FindPython3 that \
#agrees  with the major version and is at least as high for minor version and \
#patch number is accepted. If the variable is not set, we use the FindPython \
#module which favours Python 3 over Python 2 if both are available.")
set (CMAKE_CXX_STANDARD 17)
#if(APPLE)
#    # On MacOS:
#    #   -Werror causes problems with abseil on MacOS
#    #   -undefined dynamic_lookup is necessary for pybind11 linking
#    set (CMAKE_CXX_FLAGS "-w -undefined dynamic_lookup")
#else()
#     set (CMAKE_CXX_FLAGS "-Werror")
#endif()
#
## Position-independent code is needed for Python extension modules.
#set (CMAKE_POSITION_INDEPENDENT_CODE ON)


add_subdirectory (pybind11)
add_subdirectory (handIndex)
add_subdirectory (handRank)
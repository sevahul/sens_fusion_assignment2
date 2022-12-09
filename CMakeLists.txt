cmake_minimum_required(VERSION 2.8)

project( Assignment2_filtering )

find_package ( OpenCV REQUIRED )
FIND_PACKAGE(Boost COMPONENTS program_options filesystem REQUIRED)
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
find_package(OpenMP)

add_executable ( filters src/main.cpp )
target_link_libraries ( filters ${OpenCV_LIBS} )
TARGET_LINK_LIBRARIES( filters ${Boost_LIBRARIES})

if(OpenMP_CXX_FOUND)
    target_link_libraries( filters OpenMP::OpenMP_CXX)
endif()


cmake_minimum_required(VERSION 3.22)
project(cpp_study)

set(CMAKE_CXX_STANDARD 14)

include_directories(machine_learning/gradientDescent)

add_executable(cpp_study
        machine_learning/gradientDescent/gradient_descent.cpp
        machine_learning/gradientDescent/gradient_descent.h
        machine_learning/linearRegression/linear_regression.h
        machine_learning/linearRegression/linar_regression.cpp
        machine_learning/logisticRegression/logistc_regression.h
        machine_learning/logisticRegression/logistic_regression.cpp
)

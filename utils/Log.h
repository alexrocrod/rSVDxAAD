/// Alexandre Rodrigues , Jan 2021 - May 2023

#pragma once

#include <iostream>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <chrono>

typedef Eigen::SparseMatrix<double> SparseMatrixType;

namespace Log{
    /// log function to print the result of x and fx 
    void logX(const Eigen::VectorXd& x, const double& fx, bool log_x){
        if (!log_x) return;
        std::cout << std::endl;
        std::cout << "x = \n" << x.transpose() << std::endl;
        std::cout << "f(x) = " << fx << std::endl;
    }
    /// log function to print matrix related data 
    void logA(const Eigen::MatrixXd& A, const std::string& str){
        std::cout << "Matrix " << str << ": (" << A.rows() << ", " << A.cols() << ")";
        std::cout << ", norm: " << A.norm() << std::endl;
    }

    template<class TMatrix>
    void logiA(const TMatrix& A, const std::string& str){
        std::cout << "Matrix " << str << ": (" << A.rows() << ", " << A.cols() << ")";
        std::cout << ", norm: " << A.norm() << std::endl;
    }

    /// log function to print sparse matrix related data 
    void logASparse(const SparseMatrixType& A, const std::string& str){
        std::cout << "Matrix " << str << ": (" << A.rows() << ", " << A.cols() << ")";
        std::cout << ", norm: " << A.norm() << std::endl;
    }
    // void check_orth_matrix(const Eigen::MatrixXd& A, const std::string matrixName, bool full_print = false)
    void check_orth_matrix(Eigen::MatrixXd A, const std::string matrixName, bool full_print = false)
    {   
        // to remove final columns of 0s 
        // while (A(0,A.cols()-1) == 0)
        while (A.col(A.cols()-1).norm() == 0)
        {
            A = A.topLeftCorner(A.rows(),A.cols()-1);
            if (full_print) std::cout << "Removed collumn of index " << A.cols() << std::endl;
        }

        Eigen::MatrixXd AAT;
        if (A.rows()==A.cols())
            AAT = A*A.transpose();
        else
            AAT = A.transpose()*A;

        Eigen::MatrixXd A0s = AAT - Eigen::MatrixXd::Identity(AAT.rows(),AAT.cols());
        if (full_print)
        {
            std::cout << "[CHECK_ORTH] A\n" << A << std::endl;
            std::cout << "[CHECK_ORTH] [" << matrixName << "] A*A^T\n" << AAT << std::endl;
            std::cout << "[CHECK_ORTH] [" << matrixName << "] A*A^T-Iden\n" << A0s << std::endl;
        }
        // double errorA0s = A0s.norm();
        double errorA0s = A0s.cwiseAbs2().mean();
        std::cout << "[CHECK_ORTH] [" << matrixName << "] Norm(A*A^T - Iden) " << errorA0s << std::endl;
        std::cout << "[CHECK_ORTH] [" << matrixName << "] max(abs(A*A^T - Iden)) " << A0s.cwiseAbs().maxCoeff() << std::endl;

    }
        

    /// Timer class for simple timing of any code, returning in milli or microseconds
    class Timer{
    private:
        std::chrono::time_point<std::chrono::high_resolution_clock> startpoint;
    public:
        Timer() {
            startpoint = std::chrono::high_resolution_clock::now();
        }
        double stopMicro(){
            auto endpoint =std::chrono::high_resolution_clock::now();
            auto start = std::chrono::time_point_cast<std::chrono::microseconds>(startpoint).time_since_epoch().count();
            auto end = std::chrono::time_point_cast<std::chrono::microseconds>(endpoint).time_since_epoch().count();
            auto duration = end - start;
            return duration ;
        }
        double stop(){
            auto endpoint =std::chrono::high_resolution_clock::now();
            auto start = std::chrono::time_point_cast<std::chrono::milliseconds>(startpoint).time_since_epoch().count();
            auto end = std::chrono::time_point_cast<std::chrono::milliseconds>(endpoint).time_since_epoch().count();
            auto duration = end - start;
            return duration ;
        }
    };
}
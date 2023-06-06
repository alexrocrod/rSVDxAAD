// Alexandre Rodrigues alexrocrod

// #define EIGEN_DONT_VECTORIZE
#define EIGEN_VECTORIZE

#include <iostream>
#include <vector>
#include "../utils/Log.h"
#include "../utils/readHelper.h"

#include <Eigen/Core>
#include <Eigen/QR>
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>
#include <Eigen/Sparse>

#include <string>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using Eigen::HouseholderQR;
using Eigen::BDCSVD;
using Eigen::JacobiSVD;
using Eigen::SparseQR;

#define SVDxSIMD_OUR_S 5


// QR Factorization Using Modified Gram-Schmidt (by Schwarz-Rutishauser)
void qr_gs_modsr_onlyQ_vFinal(const MatrixXd& A, MatrixXd& Q)
{
    int n = A.cols();

    Q = A;
    double R;

    for( int k = 0; k < n; k++ )
    {
        for( int i = 0; i < k; i++ )
        {
            R = Q.col(i).dot(Q.col(k));

            Q.col(k) -= R * Q.col(i);
        }
        Q.col(k) /= Q.col(k).norm();
    }
    Q = -Q;
}

void basicrSVD_square(const MatrixXd& A, const int k, const int p, MatrixXd& Ures, MatrixXd& Sres, MatrixXd& Vres)
{
    // int s = 5;
    int s = SVDxSIMD_OUR_S;
    std::cout << "s " << s << std::endl;

    int m = A.rows();
    int n = A.cols();
    if (m!=n)
    {
        std::cout << "ERROR: basicrSVD_square is only for square matrices!\n";
        std::abort();
    }

    MatrixXd B = MatrixXd::Random(n, k+s);

    MatrixXd Q;
    qr_gs_modsr_onlyQ_vFinal(A * B, Q);

    for (size_t i = 0; i < p; i++)
    {
        qr_gs_modsr_onlyQ_vFinal(A.transpose()*Q, Q);
        qr_gs_modsr_onlyQ_vFinal(A * Q, Q);
    }

    B = Q.transpose() * A;

    BDCSVD<Eigen::MatrixXd> svdB(B, Eigen::ComputeThinU | Eigen::ComputeThinV);

    Ures = Q * svdB.matrixU();
    Sres = svdB.singularValues().asDiagonal();
    Vres = svdB.matrixV(); 
}

void basicrSVD(const MatrixXd& A, const int k, const int p, MatrixXd& Ures, MatrixXd& Sres, MatrixXd& Vres)
{
    // int s = 5;
    int s = SVDxSIMD_OUR_S;
    std::cout << "s " << s << std::endl;

    // [m, n] = size(A);
    int m = A.rows();
    int n = A.cols();

    // B = randn(n, k+s);
    MatrixXd B = MatrixXd::Random(n, k+s);

    // [Q, ~] = qr(A*B, 0);
    MatrixXd Q0;
    qr_gs_modsr_onlyQ_vFinal(A * B, Q0);
    // MatrixXd Q = Q0;
    MatrixXd Q = Q0.topLeftCorner(m,k+s);
    // Log::logA(Q0,"Q0");
    // Log::logA(Q,"Q");


    // for j = 1:p
    //     [Q, ~] = qr((A'*Q), 0);
    //     [Q, ~] = qr((A*Q), 0);
    // end
    for (size_t i = 0; i < p; i++)
    {
        // Q0 = HouseholderQR<MatrixXd>(A.transpose() * Q).householderQ();
        qr_gs_modsr_onlyQ_vFinal(A.transpose()*Q, Q0);
        // Q = Q0;
        Q = Q0.topLeftCorner(Q0.rows(),Q.cols());
        // Log::logA(Q0,"Q0");
        // Log::logA(Q,"Q");
        
        // Q0 = HouseholderQR<MatrixXd>(A * Q).householderQ();
        qr_gs_modsr_onlyQ_vFinal(A * Q, Q0);
        // Q = Q0;
        Q = Q0.topLeftCorner(Q0.rows(),Q.cols());
        // Log::logA(Q0,"Q0");
        // Log::logA(Q,"Q");
    }

    // B = Q'*A;
    B = Q.transpose() * A;

    //[U, S, V] = svd(B, 'econ');
    BDCSVD<Eigen::MatrixXd> svdB(B, Eigen::ComputeThinU | Eigen::ComputeThinV);
    MatrixXd U = Q * svdB.matrixU();
    MatrixXd S = svdB.singularValues().asDiagonal();
    MatrixXd V = svdB.matrixV();

    // Ures = Q * svdB.matrixU();
    // Sres = svdB.singularValues().asDiagonal();
    // Vres = svdB.matrixV();


    // U = Q*U;
    // U = U(:, 1:k);
    // S = S(1:k, 1:k);
    // V = V(:, 1:k);
    Ures = U.topLeftCorner(U.rows(), k);
    // Log::logA(U,"U");
    // Log::logA(Ures,"Ures");
    Sres = S.topLeftCorner(k, k);
    // Log::logA(S,"S");
    // Log::logA(Sres,"Sres");
    Vres = V.topLeftCorner(V.rows(), k);
    // Log::logA(V,"V");
    // Log::logA(Vres,"Vres");
 
}

void basicrSVD_analyse(const MatrixXd& A, const int k, const int p, MatrixXd& Ures, MatrixXd& Sres, MatrixXd& Vres)
{
    // int s = 5;
    int s = SVDxSIMD_OUR_S;

    // [m, n] = size(A);
    int m = A.rows();
    int n = A.cols();

    // B = randn(n, k+s);
    Log::Timer timer1;
    MatrixXd B = MatrixXd::Random(n, k+s);
    std::cout << "timer1: " << timer1.stop() << " ms" << std::endl; // 75 74 71 73 64 62 61 62

    // [Q, ~] = qr(A*B, 0);
    Log::Timer timer2;
    MatrixXd temp = A * B;
    std::cout << "timer2: " << timer2.stop() << " ms" << std::endl; // 22112  22132 22142 22993 20608 18235 18257 18250
    
    // MatrixXd Q0 = HouseholderQR<MatrixXd>(A * B).householderQ();
    // MatrixXd Q = Q0.topLeftCorner(m,k+s);
    Log::Timer timer3;
    MatrixXd Q0;
    qr_gs_modsr_onlyQ_vFinal(temp, Q0);
    // Q0 = HouseholderQR<MatrixXd>(temp).householderQ();
    std::cout << "timer3: " << timer3.stop() << " ms" << std::endl;

    Log::Timer timer6;
    MatrixXd Q = Q0.topLeftCorner(m,k+s);
    std::cout << "timer6: " << timer6.stop() << " ms" << std::endl; // 22 31 21 35 21 20 20 20 ms


    // for j = 1:p
    //     [Q, ~] = qr((A'*Q), 0);
    //     [Q, ~] = qr((A*Q), 0);
    // end

    MatrixXd At = A.transpose();

    Log::Timer timer7;
    for (size_t i = 0; i < p; i++)
    {
        qr_gs_modsr_onlyQ_vFinal(At*Q, Q0);
        // Q0 = HouseholderQR<MatrixXd>(At*Q).householderQ();
        Q = Q0.topLeftCorner(Q0.rows(),Q.cols());
        qr_gs_modsr_onlyQ_vFinal(A*Q, Q0);
        // Q0 = HouseholderQR<MatrixXd>(A*Q).householderQ();
        Q = Q0.topLeftCorner(Q0.rows(),Q.cols());
    }
    std::cout << "timer7: " << timer7.stop() << " ms" << std::endl; // 0 (p=0)

    // B = Q'*A;
    Log::Timer timer8;
    // B = Q.transpose() * A;
    B.noalias() = Q.transpose() * A;
    std::cout << "timer8: " << timer8.stop() << " ms" << std::endl; // 30904 31812 31615 16074 20980 13588 13706 13538 ms

    //[U, S, V] = svd(B, 'econ');
    Log::Timer timer9;
    BDCSVD<MatrixXd> svdB;
    std::cout << "timer9: " << timer9.stop() << " ms" << std::endl; // 0  ms

    Log::Timer timer10;
    svdB.compute(B, Eigen::ComputeThinU | Eigen::ComputeThinV);
    std::cout << "timer10: " << timer10.stop() << " ms" << std::endl; // 882  878 896 881 908 877 892 876 ms

    // Log::Timer timer11;
    MatrixXd U = svdB.matrixU();
    // std::cout << "timer11: " << timer11.stop() << " ms" << std::endl; // 0  ms

    // Log::Timer timer12;
    MatrixXd S = svdB.singularValues().asDiagonal();
    // std::cout << "timer12: " << timer12.stop() << " ms" << std::endl; // 0  ms

    // Log::Timer timer13;
    MatrixXd V = svdB.matrixV();
    // std::cout << "timer13: " << timer13.stop() << " ms" << std::endl; // 27 26 26 28 26 42 26 27 ms

    // U = Q*U;
    Log::Timer timer13_5;
    // U = Q * U;
    U = (Q * U).eval();
    std::cout << "timer13.5: " << timer13_5.stop() << " ms" << std::endl; // 81 80 81 92 82 81 83 82 ms

    // U = U(:, 1:k);
    Log::Timer timer14;
    Ures = U.topLeftCorner(U.rows(), k);
    std::cout << "timer14: " << timer14.stop() << " ms" << std::endl; // 19 21 20 19 20 20 19 19 ms

    // S = S(1:k, 1:k);
    // Log::Timer timer15;
    Sres = S.topLeftCorner(k, k);
    // std::cout << "timer15: " << timer15.stop() << " ms" << std::endl; // 0 ms

    // V = V(:, 1:k);
    Log::Timer timer16;
    Vres = V.topLeftCorner(V.rows(), k);
    std::cout << "timer16: " << timer16.stop() << " ms" << std::endl; // 20 22 19 20 19 19 20 19 ms
}



int main() 
{
    std::string sep(20,'-');

    // Eigen::setNbThreads(1); // disable multi-threading
    Eigen::setNbThreads(10); // Eigen docs say the most efficient is Nthreads = Ncores != Nths_cpu
    // Eigen::setNbThreads(20); // Eigen docs say the most efficient is Nthreads = Ncores != Nths_cpu
    // Eigen::setNbThreads(0); // set to max number of threads (40)

    std::cout << "Num. Threads: " << Eigen::nbThreads() << "\n" ;

    // A matrix sizes
    const int m = 1000; // 45115; //1000;
    const int n = 1000; // 45115; //1000;

    const int k = 500; // 100 as default
    const int p = 1; // 1 found to be the best tradeoff

    SparseMatrixType A;
    readSparseMatrix(A, "../../utils/ml.txt", m, n);
    MatrixXd ADense = MatrixXd(A);


    // MatrixXd ADense = MatrixXd::Random(m,n);
    
    Log::logA(ADense,"A");
    double Anorm = ADense.norm();
    std::cout << "k = " << k << ", p = " << p << std::endl;

    // //-------------------------------------      
    {
        // FULL MATRIX, RELEASE ->>>>>> more than 136GB of RAM XXXXXXXXXXXXXXX

        std::cout << sep << "BCDSVD" << sep << std::endl;
        Log::Timer timer;

        BDCSVD<MatrixXd> svdB(ADense, Eigen::ComputeThinU | Eigen::ComputeThinV);
        MatrixXd U = svdB.matrixU();
        MatrixXd S = svdB.singularValues().asDiagonal();
        MatrixXd V = svdB.matrixV().transpose();
        
        std::cout << "BCD SVD: " << timer.stop() << " ms" << std::endl;


        Log::Timer timer2;
        Log::logA(U,"U");
        Log::logA(S,"S");
        Log::logA(V,"V");
        MatrixXd A2 = U * S * V;
        Log::logA(A2,"A2");
        MatrixXd E = A2 - ADense;
        Log::logA(E,"E");
        std::cout << "Norm err: " << E.norm()/Anorm << std::endl;
        std::cout << "MSE err: " << E.cwiseAbs2().mean() << std::endl;
        std::cout << "Timer prints: " << timer2.stop() << " ms" << std::endl;
    }

    //-------------------------------------      
    {
        // FULL MATRIX, RELEASE ->>>>>> more than 136GB of RAM XXXXXXXXXXXXXXX

        // std::cout << sep << "JacobiSVD" << sep << std::endl;
        // Log::Timer timer;

        // JacobiSVD<MatrixXd> svdB(ADense, Eigen::ComputeThinU | Eigen::ComputeThinV);
        // MatrixXd U = svdB.matrixU();
        // MatrixXd S = svdB.singularValues().asDiagonal();
        // MatrixXd V = svdB.matrixV().transpose();
        
        // std::cout << "JacobiSVD: " << timer.stop() << " ms" << std::endl;


        // Log::Timer timer2;
        // Log::logA(U,"U");
        // Log::logA(S,"S");
        // Log::logA(V,"V");
        // MatrixXd A2 = U * S * V;
        // Log::logA(A2,"A2");
        // MatrixXd E = A2 - ADense;
        // Log::logA(E,"E");
        // std::cout << "err: " << E.norm()/Anorm << std::endl;
        // std::cout << "MSE err: " << E.cwiseAbs2().mean() << std::endl;
        // std::cout << "Timer prints: " << timer2.stop() << " ms" << std::endl;
    }
    //-------------------------------------   

    //-------------------------------------   
    {
        MatrixXd U, S, V;
        std::cout << sep << "Basic rSVD Dense" << sep << std::endl;

        Log::Timer timer;

        basicrSVD(ADense, k, p, U, S, V);

        std::cout << "Basic rSVD: " << timer.stop() << " ms" << std::endl;
  

        Log::Timer timer2;
        Log::logA(U,"U");
        Log::logA(S,"S");
        Log::logA(V,"V");
        MatrixXd A2 = U * S * V.transpose();
        Log::logA(A2,"A2");
        MatrixXd E = A2 - ADense;
        Log::logA(E,"E");
        std::cout << "err: " << E.norm()/Anorm << std::endl;
        std::cout << "MSE err: " << E.cwiseAbs2().mean() << std::endl;
        std::cout << "Timer prints: " << timer2.stop() << " ms" << std::endl;
        
    }
    //------------------------------------- 
    //-------------------------------------   
    {
        MatrixXd U, S, V;
        std::cout << sep << "Basic rSVD Square" << sep << std::endl;

        Log::Timer timer;

        basicrSVD_square(ADense, k, p, U, S, V);

        std::cout << "Basic rSVD Square: " << timer.stop() << " ms" << std::endl;

        Log::Timer timer2;
        Log::logA(U,"U");
        Log::logA(S,"S");
        Log::logA(V,"V");
        MatrixXd A2 = U * S * V.transpose();
        Log::logA(A2,"A2");
        MatrixXd E = A2 - ADense;
        Log::logA(E,"E");
        std::cout << "err: " << E.norm()/Anorm << std::endl;
        std::cout << "MSE err: " << E.cwiseAbs2().mean() << std::endl;
        std::cout << "Timer prints: " << timer2.stop() << " ms" << std::endl;

    }
    //-------------------------------------  
    //-------------------------------------   
    {
        MatrixXd U, S, V;
        std::cout << sep << "Basic rSVD Analyse" << sep << std::endl;

        Log::Timer timer;

        basicrSVD_analyse(ADense, k, p, U, S, V);

        std::cout << "Basic rSVD: " << timer.stop() << " ms" << std::endl;

        Log::Timer timer2;
        Log::logA(U,"U");
        Log::logA(S,"S");
        Log::logA(V,"V");
        MatrixXd A2 = U * S * V.transpose();
        Log::logA(A2,"A2");
        MatrixXd E = A2 - ADense;
        Log::logA(E,"E");
        std::cout << "err: " << E.norm()/Anorm << std::endl;
        std::cout << "MSE err: " << E.cwiseAbs2().mean() << std::endl;
        std::cout << "Timer prints: " << timer2.stop() << " ms" << std::endl;

    }
    //------------------------------------- 

    return 0;
}
// Alexandre Rodrigues
// alexrocrod
// May 2023

// rSVDxAAD to use our method and print the sum of all adjoint values to compare indirectly

#pragma once

#include <iostream>
#include <vector>
#include <functional>
#include <cmath>

#include "../utils/Log.h"
#include "../utils/readHelper.h"
using Log::logA;
using Log::logiA;


#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/QR>

using Eigen::HouseholderQR;
using Eigen::CompleteOrthogonalDecomposition;
using Eigen::VectorXd;
using Eigen::MatrixXd;
using Eigen::BDCSVD;

class SVDxAdjHelper
{
private:
    MatrixXd m_A; // original matrix 
    int m_N; // N = A.rows()= A.cols()
    int m_k; // Used number of eigenvectors/values
    int m_s; // oversampling for rSVD (default: 5)
    int m_p; // number of power iterations in p (default: 1)

    MatrixXd m_Ursvd, m_Srsvd, m_Vrsvd; // matrix to store rSVD results
    
    double m_cutoffVal; // cutoff (default is smaller lambda/2)

    VectorXd m_SingVals; // reduced singular values used for adjoint
    MatrixXd m_U_extended; // extended U matrix
    MatrixXd m_V_extended; // transpose of U_ext

    double m_smallSingValsApprox; // value to give to unknown eigenvalues

public:
    SVDxAdjHelper(const MatrixXd& A, const int k)
    : m_A(A), m_N(A.rows()), m_k(k), 
        m_p(1), m_s(5), m_smallSingValsApprox(0), m_cutoffVal(-1.0)
    {
        if (A.rows() != A.cols())
        {
            std::cout << "SVDxAdjHelper ERROR: Rectangular Matrices are not supported\n";
            std::abort();
        }
        if (k>A.rows())
        {
            std::cout << "SVDxAdjHelper ERROR: Value of k is larger than N\n";
            std::abort();
        }
    }

    void run()
    {
        Log::Timer timerRSVD;
        basicrSVD();
        std::cout << "Basic rSVD: " << timerRSVD.stopMicro() << " microsecs\n";

        Log::Timer timerPrints;
        MatrixXd A2 = m_Ursvd * m_Srsvd * m_Vrsvd.transpose();
        MatrixXd E = A2 - m_A;
        std::cout << "Norm err: " << E.norm()/m_A.norm() << std::endl;
        std::cout << "MSE err: " << E.cwiseAbs2().mean() << std::endl;

        //------------------------------------- 
        if(m_cutoffVal < 0) // not given by the user
            m_cutoffVal = m_Srsvd.diagonal().minCoeff()/2;

        // our SingVals will be the known from matrix S and then m_smallSingValsApprox for the unknowns
        m_SingVals = VectorXd::Ones(m_N) * m_smallSingValsApprox;
        m_SingVals.topLeftCorner(m_k,1) = m_Srsvd.diagonal();

        m_U_extended = MatrixXd::Identity(m_N,m_N);
        m_U_extended.topLeftCorner(m_N,m_k) = m_Ursvd;
        qr_gs_mod(m_U_extended);

        m_V_extended = m_U_extended.transpose();

        //---------------
        Log::Timer tCompDerivs;
        compareDerivatives();
        std::cout << "Timer compDerivs: " << tCompDerivs.stopMicro() << " microsecs\n";
        
    }
    

    // Setters and Getters for rSVD
    const int getK()
    {
        return m_k;
    }
    void setK(const int k)
    {
        if (k>m_N)
        {
            std::cout << "SVDxAdjHelper ERROR: Value of k (changed) is larger than N\n";
            return;
        }
        m_k = k;
    }
    const int get_Oversampling()
    {
        return m_s;
    }
    void setOversampling(const int s)
    {
        m_s = s;
    }
    const int getPowerItNum()
    {
        return m_p;
    }
    void setPowerItNum(const int p)
    {
        m_p = p;
    }

    const MatrixXd& rSVD_matrixU()
    {
        return m_Ursvd;
    }
    const MatrixXd& rSVD_matrixS()
    {
        return m_Srsvd;
    }
    const MatrixXd& rSVD_matrixV()
    {
        return m_Vrsvd;
    }

    // Setters and Getters for matrix Adjoint
    const double getCutoffVal()
    {
        return m_cutoffVal;
    }
    void setCutoffVal(const double cutoffVal)
    {
        m_cutoffVal = cutoffVal;
    }
    const double getSmallSingValsApprox()
    {
        return m_smallSingValsApprox;
    }
    void setSmallSingValsApprox(const double x)
    {
        m_smallSingValsApprox = x;
    }

    const MatrixXd& matrixU_extended()
    {
        return m_U_extended;
    }
    const MatrixXd& matrixV_extended()
    {
        return m_V_extended;
    }
    const VectorXd& reducedSingVals()
    {
        return m_SingVals;
    }

private:
    // QR Factorization Using Modified Gram-Schmidt (by Schwarz-Rutishauser)
    void qr_gs_mod(MatrixXd& Q)
    {
        double R;

        for( int k = m_k; k < m_N; k++ ) // (N-K)  //N 
        {
            for( int i = 0; i < k; i++ ) // (1+2+3+...+N) = 1/2 N (N+1)
            {
                R = Q.col(i).dot(Q.col(k));
                Q.col(k) -= R * Q.col(i);
            }
            Q.col(k) /= Q.col(k).norm();
        }
        // Q = -Q;
    } 

    double myStep(const double x)
    {
        if (x < m_cutoffVal)
            return 0;
        return x;
    }

    double myStepDeriv(const double x)
    {
        if (x < m_cutoffVal)
            return 0;
        return 1;
    }

    // QR Factorization Using Modified Gram-Schmidt (by Schwarz-Rutishauser)
    void qr_gs_modsr_onlyQ(const MatrixXd& A, MatrixXd& Q)
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

    void basicrSVD()
    {
        MatrixXd B = MatrixXd::Random(m_N, m_k+m_s);

        MatrixXd Q0;
        qr_gs_modsr_onlyQ(m_A * B, Q0);
        MatrixXd Q = Q0.topLeftCorner(m_N,m_k+m_s);

        for (size_t i = 0; i < m_p; i++)
        {
            qr_gs_modsr_onlyQ(m_A.transpose()*Q, Q0);
            Q = Q0.topLeftCorner(Q0.rows(),Q.cols());

            qr_gs_modsr_onlyQ(m_A * Q, Q0);
            Q = Q0.topLeftCorner(Q0.rows(),Q.cols());
        }
        B = Q.transpose() * m_A;

        BDCSVD<Eigen::MatrixXd> svdB(B, Eigen::ComputeThinU | Eigen::ComputeThinV);
        MatrixXd U = Q * svdB.matrixU();
        MatrixXd S = svdB.singularValues().asDiagonal();
        MatrixXd V = svdB.matrixV();

        m_Ursvd = U.topLeftCorner(U.rows(), m_k);
        m_Srsvd = S.topLeftCorner(m_k, m_k);
        m_Vrsvd = V.topLeftCorner(V.rows(), m_k);
    }

    void fAdjoint(const MatrixXd& A_bar, MatrixXd& A_bar_res)
    {
        MatrixXd F = MatrixXd::Zero(m_N,m_N);

        for (size_t i = 0; i < m_N; i++)
        {
            for (size_t j = 0; j < m_N; j++)
            {
                // # ARTICLE: formula (5) F
                if (fabs(m_SingVals[i]-m_SingVals[j]) > 1e-6)
                    F(i,j)=(myStep(m_SingVals[i])-myStep(m_SingVals[j])) / (m_SingVals[i]-m_SingVals[j]);
                else
                    F(i,j)=myStepDeriv(m_SingVals[i]);
            }
        } 

        A_bar_res = m_U_extended * ((m_V_extended * A_bar * m_U_extended).cwiseProduct(F)) * m_V_extended;
    }

    void compareDerivatives()
    {
        double sumAbsA_bar = 0.0;
        MatrixXd A_bar;

        for (size_t i = 0; i < m_N; i++)
            for (size_t j = 0; j < i+1; j++)
            {
                MatrixXd E = MatrixXd::Zero(m_N,m_N);
                E(i,j) += 1;
                E(j,i) += 1;
                fAdjoint(E, A_bar);

                for (size_t k = 0; k < m_N; k++)
                    for (size_t l = 0; l < k+1; l++)
                        sumAbsA_bar += fabs(A_bar(k,l));
            }  
        std::cout << "(Adjoint) sumAbsA_bar: " <<  sumAbsA_bar << std::endl; 
    }
};

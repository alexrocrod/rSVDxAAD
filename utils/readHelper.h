// March 2023 - May 2023
// Alexandre Rodrigues alexrocrod

#pragma once

#include <iostream>

#include <fstream>
#include <Eigen/Sparse>

#include <random>

using Eigen::SparseMatrix;

typedef SparseMatrix<double> SparseMatrixType;

void readSparseMatrix( SparseMatrixType& A, const std::string& filepath = "ml.txt", int m = 45115, int n = 45115)
{
    typedef Eigen::Triplet<double> T;
    std::vector<T> tripletList;
    tripletList.reserve(std::max(m,n));

    std::ifstream infile(filepath);

    if (!infile.good())
    {
        std::cout << "ERROR: FILE NOT PRESENT\n";
        std::abort();
    }

    double a, b, c;

    while (infile >> a >> b >> c)
    {
        if (a > m || b > n) continue;
        tripletList.push_back(T((int)a-1,(int)b-1,c));
    }

    A.resize(m,n);
    A.setFromTriplets(tripletList.begin(), tripletList.end());
}

double sample(double dummy)
{
    std::random_device rd;
    std::mt19937_64 rng(rd());
    std::normal_distribution<double> nd;
    return nd(rng);
}

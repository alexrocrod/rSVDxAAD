// Alexrocrod May 2023

// #define EIGEN_DONT_VECTORIZE
#define EIGEN_VECTORIZE
 
// #include "SVDxAdjHelper.h" // only uses our method and prints the sum of all adjoint values to compare indirectly
// #include "SVDxAdjHelperCompare.h" // compare with Finite Differences and print error indicators
#include "SVDxAdjHelperCompareTrad.h" // compare with Traditional Adjoint Formula and print error indicators

int main()
{
    std::string sep(20,'-');

    Eigen::setNbThreads(10); // Eigen docs say the most efficient is Nthreads = Ncores != Nths_cpu
    // Eigen::setNbThreads(1); // Eigen docs say the most efficient is Nthreads = Ncores != Nths_cpu
    std::cout << "Num. Threads: " << Eigen::nbThreads() << "\n" ;

    Log::Timer timerData;

    const int N = 20;   // A matrix sizes
    const int k = 15;   //
    const int p = 1;    // 1 found to be the best tradeoff

    MatrixXd Ubase = MatrixXd::Random(N,N); // Eigen's random
    // MatrixXd Ubase = MatrixXd::NullaryExpr(N,N,std::ptr_fun(sample)); // using c++ random
    logA(Ubase,"Ubase");
    Ubase = HouseholderQR<MatrixXd>(Ubase).householderQ(); // ortogonalization of the random matrix
    logA(Ubase,"Ubase");

    MatrixXd Vbase = Ubase.transpose();

    VectorXd Svec(N);
    for (size_t i = 0; i < k; i++)
    {
        Svec(i) = double(N-i)+5;
        // Svec(i) = double(k-i)+10; 
    }
    for (size_t i = k; i < N; i++)
    {
        Svec(i) = double(N-i)/100;
    }
    
    MatrixXd ADense = Ubase * Svec.asDiagonal() * Vbase;

    Log::logA(ADense,"A");
    double Anorm = ADense.norm();
    std::cout << "k = " << k << ", p = " << p << std::endl; 

    std::cout << "Data Gen : " << timerData.stopMicro() << " microsecs\n";

    //-------------------------------------   
    SVDxAdjHelper All_helper = SVDxAdjHelper(ADense,k);

    std::cout << sep << "run"<< sep << std::endl;
    All_helper.setPowerItNum(p);
    All_helper.run();

    // worse results due to discarding 3 of the large eigenvalues
    std::cout << sep << "k=12"<< sep << std::endl;
    All_helper.setK(12);
    All_helper.run();

    // should do nothing due to k>N (same results as previous run)
    std::cout << sep << "k=1000"<< sep << std::endl;
    All_helper.setK(1000);
    All_helper.run();
    
    // should improve but can be almost insignificant
    std::cout << sep << "p=5"<< sep << std::endl;
    All_helper.setK(k); // default
    All_helper.setPowerItNum(5);
    All_helper.run();

    // should be just slighlty worse 
    std::cout << sep << "s=0"<< sep << std::endl;
    All_helper.setPowerItNum(p); // default
    All_helper.setOversampling(0); 
    All_helper.run();

    // should be bad because it is larger than some of the eigenvalues
    std::cout << sep << "cutooff=12.5" << sep << std::endl;
    All_helper.setOversampling(5); // default
    // All_helper.setCutoffVal(2.0);  // should do nothing due to 2<11=minEigenVal
    All_helper.setCutoffVal(12.5);  // larger than some eigenvalues
    All_helper.run();

    // should be bad because our small eigenvalues are not close to 1
    std::cout << sep << "unkwonEigenVals=1"<< sep << std::endl;
    All_helper.setCutoffVal(5.5); // default
    All_helper.setSmallSingValsApprox(1.0);
    All_helper.run();
    
    return 0;
}
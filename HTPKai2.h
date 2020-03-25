#include <stdio.h>
#include <algorithm>
#include <utility>
#include <functional>
#include <cstring>
#include <queue>
#include <stack>
#include <cmath>
#include <iterator>
#include <vector>
#include <string>
#include <set>
#include <iostream>
#include <random>
#include <map>
#include <iomanip>
#include <stdlib.h>
#include <list>
#include <typeinfo>
#include <list>
#include <set>
#include <cassert>
#include <fstream>
#include <unordered_map>
#include <unordered_set>
#include <cstdlib>
#include <complex>
#include <cctype>
#include <bitset>
#include <chrono>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;
using Eigen::MatrixXd;
using Supp = bitset<1024>;
using ll   = long long;
using vi   = vector<int>;
using vpii = vector<pair<int, int>>;
using pi   = pair<int, int>;
using pdi  = pair<double, int>;
using mat  = Eigen::MatrixXd;

#define REP(i,n) for(int i(0);(i)<(n);(i)++)
#define ALL(a) a.begin(), a.end()

//* declarations
struct QrData;

struct GivensRotation{
    //* member variables
    JacobiRotation<double> G;
    int i1, i2;
    //* member functions
    GivensRotation(){}

    GivensRotation(JacobiRotation<double> const &_G, int _i1, int _i2){
        G = _G; i1 = _i1; i2 = _i2;
    }

    GivensRotation(double p, double q, int _i1, int _i2){
        G.makeGivens(p, q); i1 = _i1; i2 = _i2;
    }

    void makeGivens(double p, double q, int _i1, int _i2){
        G.makeGivens(p, q); 
        i1 = _i1; 
        i2 = _i2;
    }
};

struct GivensRotationSequence{
    //* member variables
    vector<GivensRotation > data;
    int len, end_pt;
    //* menber functions
    GivensRotationSequence(){
        len = 0; end_pt = 0;
    }
    GivensRotationSequence(int _len){
        len = _len;
        data.resize(len);
        end_pt = 0;
    }

    void push( GivensRotation  const &G){
        if(end_pt < len){
            data[end_pt] = G;
            end_pt++;
        }
        else{
            cout << "GivensRotationSequenceLimitExceed, len= "<< len <<endl;
        }
    }
};

struct TestRunAggregater{
    double test1;
    double test2;
    double test3;
    void Reset()  { test1 = test2 = test3 = 0;}
};

template<std::size_t N>
bool operator<(const std::bitset<N>& x, const std::bitset<N>& y){   
    for (int i = N-1; i >= 0; i--) {
        if (x[i] ^ y[i]) return y[i];
    }
    return false;
}

template<size_t N> struct BitsetComparer {
    bool operator() (const bitset<N> &b1, const bitset<N> &b2) const {
        return b1 < b2;
    }
};

struct QrData{
    Supp S;
    vector<int> Seq;
    HouseholderQR<MatrixXd> qr;
    MatrixXd QTransA;
    vector<int> ref_seq;
};

//* Global variables
int gM; //* number of rows
int gN; //* number of columns
int gK; //* number of non-zeros allowed
double gResTolRatio    = 1e-6; //* residual tolerence ratio
double gSuppDiffRatio  = 0.1; //* reference difference ratio
double gSuppGuardRatio = 0.25; //* ratio of index should be reserved in makeSeed
double gSuppDropRatio = 0.1;
int gMaxHTPCnt = 1000;
mat gA; //* original sensing matrix
mat gRhs; //* original right-hand-side 
Supp gZeroS("0"); //* empty support
Supp gOne("1"); //* == 1LL
double gRhsNorm; //! NEET SETTING in initialization
double gResTol; //! NEET SETTING in initialization
unordered_set<Supp> gVisitedSupp; //* record the support info for all the visited vectex
                                 //! "visited" means this residual is calculated for Supp
unordered_set<Supp> gFixedPoint; //* record the support info for all fixed point
vector<QrData> gQrInfo; //* record the QR decomposition data calculated so far

int gHTPCnt = 0;
Supp gResultS;


//* function declaration
GivensRotationSequence allocateGivensRotationSequence(int m, int n, vector<int> cols);
void getAndApplyGivensRotation(MatrixXd &A, vector<int> cols, int j, double p, double q, int i1, int i2);
void QrRecoverOneCol(MatrixXd &A, vector<int> &cols, int j, GivensRotationSequence &Gseq);
void QrRecover(MatrixXd &A, vector<int> &cols, GivensRotationSequence &Gseq);
MatrixXd LsReference(MatrixXd const &A, MatrixXd const &b);
MatrixXd LsRecover(MatrixXd const &A, MatrixXd const &b, vector<int> const &cols, HouseholderQR<MatrixXd> &qr_base);
MatrixXd LsSequenceInner(HouseholderQR<MatrixXd> &qr, MatrixXd const &b);
MatrixXd LsSequence(MatrixXd const &A, MatrixXd const &b);
MatrixXd LsSequenceRef(MatrixXd const &A, MatrixXd const &b);

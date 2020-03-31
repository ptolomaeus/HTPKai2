# HTPKai2

C++ implementation for a variable selection algorithm for compressive problem. The key ingredient of this algorithm is based on the Hard Threshold Pursuit algorithm and thus it is named as HTPKai2 meaning a remodeling of HTP. Eigen3, a C++ numerical linear algebra package is used.

##  How to compile

1. Download the Eigen3 source code from http://eigen.tuxfamily.org/index.php?title=Main_Page
No installization is needed: you only need to link the head files.

2. Use the following command to compile. (Tested on my own macpro only with clang compiler)

```Bash
g++ -I Eigen3_Dir  -O3 -mavx HTPKai2.cpp -o HTPKai2
```

## How to run
Follow the instruction of the program.

LFW_API
=======
>**Designer:** Junbo Zhao, Wuhan University, Working in Tsinghua National lab of intelligent images and documents processing.      
**Contact:** zhaojunbo1992chasing@gmail.com +86-18672365683       

Introduction
-----------------------------------
For better using Labeled Faces in the Wild Benchmark, I provide the code of three versions, namely C++, python and Octave. Both **Restricted** and **Unrestricted** configurations are implemented. Files named with "unrest" are written for unrestricted configuration, and those with "rest" aim at restricted configuration.   

C++ version
--------------------------------------
### Eigen
Eigen is a C++ template library for linear algebra: matrices, vectors, numerical solvers, and related algorithms. You should install Eigen following the tutorial:   http://eigen.tuxfamily.org/dox/GettingStarted.html          
Our input as well as output feature matrix are constructed with Eigen::MatrixXd or Eigen::VectorXd. After installing Eigen on your project, you can write matrix as simply as in Matlab!

### Get started
Note that you should firstly extract features on the images in LFW, your features should be constructed as a stack of "dict" class, which includes considered each person's name, number of his or her images and extracted feature matrix of the person. You can see how this class constructed in both .cpp files. Furthermore, you can make your feature matrix as row-ordered or column-ordered. The row-ordered matrix means each row represents a feature vector of a specific image, and column-ordered means each column represents the vector. You can change this using the parameter "axis", whose default setting is 0, which means row-ordered setting.

### pairs.txt and people.txt
Prepare the .txt files for different configurations of LFW! You can find instructions here:         
http://vis-www.cs.umass.edu/lfw/README.txt

python version
------------------------------------------------------------
### Get started
You can simply make the parameters like what I suggested above with C++ version, but since python provides dictionary and tuple structs, it turns much more gorgeous. The input feature matrix is a dictionary, whose keys are human names and values are feature matrix as numpy.ndarray. Output is a dictionary as well whose keys are "train" and "test", pointing to two tuples which involve feature matrix and their labels.

Platform
--------------------------------------------------------
My platform is Ubuntu 12.04 LTS with g++ 4.6.3, Eigen 3.1.4 and python 2.7.5. I propose the code is compatible to other platforms, and if you have some problems compiling or running it, feel free to contact me.

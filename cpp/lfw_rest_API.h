/***********************************************************************
 *  * Designed by Junbo Zhao, working in Center for Intelligent Images and 
 *  * Document Processing Lab, Tsinghua University.
 *  * Better using Labeled Faces in the Wild (LFW).
 *  * ***********************************************************************/

#ifndef _LFW_REST_API_H_
#define _LFW_REST_API_H_

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <vector>
#include <string>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

class dict{
public:
  string name;
  int num;
  MatrixXd fea;
  dict(string na, int nu){
    name = na;
    num = nu;
  }
  dict(string na, int nu, MatrixXd fe){
    name = na;
    num = nu;
    fea = fe;
  }
};

class samplepair{
public:
  string name1;
  string name2;
  int num1;
  int num2;
  samplepair(string na1, string na2){
    name1 = na1;
    name2 = na2;
  }
  samplepair(string na1, string na2,
      int nu1, int nu2){
    name1 = na1;
    name2 = na2;
    num1 = nu1;
    num2 = nu2;
  }
};

string alphabet("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_ ");
string alphabet_pure("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz");

bool RestLFW(string filename, int fold, vector<dict> features,
              MatrixXd& feature_train, MatrixXd& feature_test, 
              VectorXd& label_train, VectorXd& label_test, int axis=0);

#endif

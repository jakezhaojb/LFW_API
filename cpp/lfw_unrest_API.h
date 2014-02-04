 /***********************************************************************                                                                                                           
*  * Designed by Junbo Zhao, working in Center for Intelligent Images and
*  * Document Processing Lab, Tsinghua University.
*  * Better using Labeled Faces in the Wild (LFW).
*  * ***********************************************************************/

#ifndef _LFW_UNREST_API_H_
#define _LFW_UNREST_API_H_

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

int sum_sample_num(vector<dict> fea){
  int i, sum=0;
  for(i = 0; i < fea.size(); i++) {
    sum += fea[i].num;
  }
}

bool UnRestLFW(string filename, int fold, vector<dict> features,
              MatrixXd& feature_train, MatrixXd& feature_test, 
              VectorXd& label_train, VectorXd& label_test, int axis=0);

#endif

#include "lfw_rest_API.h"

bool RestLFW(string filename, int fold, vector<dict> features,
              MatrixXd& feature_train, MatrixXd& feature_test, 
              VectorXd& label_train, VectorXd& label_test, int axis){
// axis: INPUT and OUTPUT features are stored column-ordred or row-ordred.
// And axis = 0 meansrow orded, axis = 1 means column-ordred.
// The default setting is axis = 0.
  int i, j, k;
  // QA
  if (axis == 0) {
    for (i = 0; i < features.size(); i++) {
      if(features[i].num != features[i].fea.rows()){
        cout<< "Input feature are bad. Examine it carefully" <<endl;
        return false;
      }
    }
  }
  else if(axis == 1){
    for (i = 0; i < features.size(); i++) {
      if(features[i].num != features[i].fea.cols()){
        cout<< "Input feature are bad. Examine it carefully" <<endl;
        return false;
      }
    }
  }
  else{
    cout<<"The input parameter 'axis' is limited to 0 and 1"<<endl;
    return false;
  }

  // Read the whole dataset
  ifstream fin(filename.c_str());
  if(!fin){
    cout<< "File not found"<< endl;
    exit(-1);
  }

  // Get the Nfold
  string str_Nfold_Npos_neg, str_Nfold, str_Npos_neg;
  int Nfold, Npos_neg;
  getline(fin, str_Nfold_Npos_neg);
  str_Nfold.assign(str_Nfold_Npos_neg.begin(), str_Nfold_Npos_neg.begin() + 
                   str_Nfold_Npos_neg.find_first_of(" "));
  str_Npos_neg.assign(str_Nfold_Npos_neg.begin() + 
                      str_Nfold_Npos_neg.find_last_of(" ") + 1, 
                      str_Nfold_Npos_neg.end());
  Nfold = atoi(str_Nfold.c_str());
  Npos_neg = atoi(str_Npos_neg.c_str());

  // Get the name of the file
  vector<samplepair> per_fea_num_train, per_fea_num_test;
  for (i = 0; i < Nfold; i++) {
    vector<samplepair> per_fea_num_unit;
    for (j = 0; j < Npos_neg; j++) {
      string pos_line, pos_per, str_num;
      int position, num1, num2;
      getline(fin, pos_line);
      pos_per.assign(pos_line.begin(), pos_line.begin() + pos_line.find_first_of(" "));
      position = pos_line.find_first_not_of(alphabet);
      str_num.assign(str_num.begin() + position, str_num.begin() + 
                     str_num.find_first_of(" ", position));
      num1 = atoi(str_num.c_str());
      str_num.assign(str_num.begin() + str_num.find_last_of(" ") + 1, 
                     str_num.end());
      num2 = atoi(str_num.c_str());
      per_fea_num_unit.push_back(samplepair(pos_per, pos_per, num1, num2)); 
    }
    for (j = 0; j < Npos_neg; j++) {
      string neg_line, neg_per1, neg_per2, str_num;
      int position1, position2, num1, num2;
      getline(fin, neg_line);
      position1 = neg_line.find_first_of(alphabet_pure, 
                                         neg_line.find_first_not_of(alphabet));
      position2 = neg_line.find_first_not_of(alphabet);
      neg_per1.assign(neg_line.begin(), neg_line.begin() + 
                      neg_line.find_first_of(" "));
      neg_per2.assign(neg_line.begin() + position1, neg_line.begin() + 
                      neg_line.find_first_of(" ", position1));
      str_num.assign(neg_line.begin() + position2, neg_line.begin() + 
                     neg_line.find_first_of(" ", position2));
      num1 = atoi(str_num.c_str());
      str_num.assign(neg_line.begin() + neg_line.find_first_not_of(alphabet), 
                     neg_line.end());
      num2 = atoi(str_num.c_str());
      per_fea_num_unit.push_back(samplepair(neg_per1, neg_per2, num1, num2));
    }
    if (i==fold) {
      per_fea_num_test.assign(per_fea_num_unit.begin(), per_fea_num_unit.end());
    }
    else{
      per_fea_num_train.insert(per_fea_num_train.end(),
                              per_fea_num_unit.begin(), 
                              per_fea_num_unit.end());
    }
  }
  // QA
  assert(fin.eof());
  fin.close();

  vector<string> totalname;
  for (i = 0; i < features.size(); i++) {
    totalname.push_back(features[i].name);
  }
  vector<string>::iterator it1, it2;
  int it_num1, it_num2; //获取迭代器的位置有没有更好的方法？
  if (axis == 0) {
    feature_train = MatrixXd(per_fea_num_train.size() * 2, 
                            features[0].fea.cols());
    feature_test = MatrixXd(per_fea_num_test.size() * 2, 
                            features[0].fea.cols());
    label_train = VectorXd(per_fea_num_train.size());
    label_test = VectorXd(per_fea_num_test.size());
    for ( i = 0; i < per_fea_num_train.size(); i++) {
      it1 = find(totalname.begin(), totalname.end(), per_fea_num_train[i].name1);
      it_num1 = it1 - totalname.begin();
      it2 = find(totalname.begin(), totalname.end(), per_fea_num_train[i].name2);
      it_num2 = it2 - totalname.begin();
      feature_train.row(2*i) = features[it_num1].fea.row(per_fea_num_train[i].num1);
      feature_train.row(2*i+1) = features[it_num2].fea.row(per_fea_num_train[i].num2);
      if(per_fea_num_train[i].name1 == per_fea_num_train[i].name2)
        label_train[i] = 1;
      else
        label_train[i] = 0;
    }
    for ( i = 0; i < per_fea_num_test.size(); i++) {
      it1 = find(totalname.begin(), totalname.end(), per_fea_num_test[i].name1);
      it_num1 = it1 - totalname.begin();
      it2 = find(totalname.begin(), totalname.end(), per_fea_num_test[i].name2);
      it_num2 = it2 - totalname.begin();
      feature_test.row(2*i) = features[it_num1].fea.row(per_fea_num_test[i].num1);
      feature_test.row(2*i+1) = features[it_num2].fea.row(per_fea_num_test[i].num2);
      if(per_fea_num_test[i].name1 == per_fea_num_test[i].name2)
        label_test[i] = 1;
      else
        label_test[i] = 0;
    }
    return true;
  }
  else {
    feature_train = MatrixXd(features[0].fea.rows(),
                            per_fea_num_train.size() * 2) ;
    feature_test = MatrixXd(features[0].fea.rows(),
                            per_fea_num_test.size() * 2) ;
    label_train = VectorXd(per_fea_num_train.size());
    label_test = VectorXd(per_fea_num_test.size());
    for ( i = 0; i < per_fea_num_train.size(); i++) {
      it1 = find(totalname.begin(), totalname.end(), per_fea_num_train[i].name1);
      it_num1 = it1 - totalname.begin();
      it2 = find(totalname.begin(), totalname.end(), per_fea_num_train[i].name2);
      it_num2 = it2 - totalname.begin();
      feature_train.col(2*i) = features[it_num1].fea.col(per_fea_num_train[i].num1);
      feature_train.col(2*i+1) = features[it_num2].fea.col(per_fea_num_train[i].num2);
      if(per_fea_num_train[i].name1 == per_fea_num_train[i].name2)
        label_train[i] = 1;
      else
        label_train[i] = 0;
    }
    for ( i = 0; i < per_fea_num_test.size(); i++) {
      it1 = find(totalname.begin(), totalname.end(), per_fea_num_test[i].name1);
      it_num1 = it1 - totalname.begin();
      it2 = find(totalname.begin(), totalname.end(), per_fea_num_test[i].name2);
      it_num2 = it2 - totalname.begin();
      feature_test.col(2*i) = features[it_num1].fea.col(per_fea_num_test[i].num1);
      feature_test.col(2*i+1) = features[it_num2].fea.col(per_fea_num_test[i].num2);
      if(per_fea_num_test[i].name1 == per_fea_num_test[i].name2)
        label_test[i] = 1;
      else
        label_test[i] = 0;
    }
    return true;
  }

}

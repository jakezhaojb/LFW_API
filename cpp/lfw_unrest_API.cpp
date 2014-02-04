#include "lfw_unrest_API.h"

bool UnRestLFW(string filename, int fold, vector<dict> features,
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
  string str_Nfold;
  int Nfold;
  getline(fin, str_Nfold);
  Nfold = atoi(str_Nfold.c_str());

  // Get the name of the file
  vector<dict> per_fea_num_train;
  vector<dict> per_fea_num_test;
  for( i = 0; i < Nfold; i++){
    vector<dict> per_fea_num_unit;
    string str_person_num;
    int person_num;
    vector<int> sample_num;
    getline(fin, str_person_num);
    person_num = atoi(str_person_num.c_str());
    vector<string> person;
    for ( j = 0; j < person_num; j++) {
      string line;
      string per, str_num;
      int num;
      getline(fin, line);
      per.assign(line.begin(), line.begin() + line.find_first_of(" "));
      str_num.assign(line.begin() + line.find_last_of(" ") + 1, line.end());
      num = atoi(str_num.c_str());
      person.push_back(per);
      sample_num.push_back(num);
    }
    assert(sample_num.size() == person.size());   // QA
    for ( k = 0; k < person.size(); k++) {
      per_fea_num_unit.push_back(dict(person[k], sample_num[k]));
    }
    if (i == fold) {
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
  vector<string>::iterator it;
  int it_num;
  if (axis == 0) {
    feature_train = MatrixXd(sum_sample_num(per_fea_num_train), 
                            features[0].fea.cols());
    feature_test = MatrixXd(sum_sample_num(per_fea_num_test), 
                            features[0].fea.cols());
    label_train = VectorXd(sum_sample_num(per_fea_num_train));
    label_test = VectorXd(sum_sample_num(per_fea_num_test));
    k = 0;
    for ( i = 0; i < per_fea_num_train.size(); i++) {
      it = find(totalname.begin(), totalname.end(), per_fea_num_train[i].name);
      it_num = it - totalname.begin();
      for ( j = 0; j < per_fea_num_train[i].num; j++) {
        feature_train.row(k) = features[it_num].fea.row(j);
        label_train(k) = i;
        k++;
      }
    }
    k = 0;
    for ( i = 0; i < per_fea_num_test.size(); i++) {
      it = find(totalname.begin(), totalname.end(), per_fea_num_test[i].name);
      it_num = it - totalname.begin();
      for ( j = 0; j < per_fea_num_test[i].num; j++) {
        feature_test.row(k) = features[it_num].fea.row(j);
        label_test(k) = i;
        k++;
      }
    }
    return true;
  }

  else{
    feature_train = MatrixXd(features[0].fea.rows(),
                            sum_sample_num(per_fea_num_train));
    feature_test = MatrixXd(features[0].fea.rows(),
                            sum_sample_num(per_fea_num_test));
    label_train = VectorXd(sum_sample_num(per_fea_num_train));
    label_test = VectorXd(sum_sample_num(per_fea_num_test));
    k = 0;
    for ( i = 0; i < per_fea_num_train.size(); i++) {
      it = find(totalname.begin(), totalname.end(), per_fea_num_train[i].name);
      it_num = it - totalname.begin();
      for ( j = 0; j < per_fea_num_train[i].num; j++) {
        feature_train.col(k) = features[it_num].fea.col(j);
        label_train(k) = i;
        k++;
      }
    }
    k = 0;
    for ( i = 0; i < per_fea_num_test.size(); i++) {
      it = find(totalname.begin(), totalname.end(), per_fea_num_test[i].name);
      it_num = it - totalname.begin();
      for ( j = 0; j < per_fea_num_test[i].num; j++) {
        feature_test.col(k) = features[it_num].fea.col(j);
        label_test(k) = i;
        k++;
      }
    }
    return true;
  }
}

# Designed by Junbo Zhao, *  * Designed by Junbo Zhao,
# working in Center for Intelligent Images and Document
# Processing Lab, Tsinghua University.

import os
import numpy as np
# Note that this script can be used as an API for processing
# the Labeled Faces in Wild benchmark.
# Regularly building cross-validation training and testing dataset.
# 10 Fold


def unrestricted_lfw_API(filename, fold, features, axis=0):
# axis: INPUT and OUTPUT features are stored column-ordred or row-ordred.
# And axis = 0 meansrow orded, axis = 1 means column-ordred.
# The default setting is axis = 0.
    # Read the whole dataset
    if not isinstance(features, dict):
        print "The features (para NO.3) must be a dictionry data, whose keys\
                    indicate the people's name and values are extracted\
                    feature vectors"
        return False
    if not os.path.isfile(filename):
        return False
    fid = file(filename)
    Nfold = int(fid.readline())
    per_fea_num = [dict()] * Nfold
    per_fea_num_train = dict()
    per_fea_num_test = dict()
    for i in range(Nfold):
        person_num = int(fid.readline())
        person = []
        sample_num = []
        for j in range(person_num):
            line = fid.readline()
            line = line.split()
            person.append(line[0])
            sample_num.append(int(line[1]))
        per_fea_num[i] = dict(zip(person, sample_num))
        if i is fold:
            per_fea_num_test = per_fea_num[i]
        else:
            per_fea_num_train.update(per_fea_num[i])

    # QA, to make sure file is completed.
    assert(len(fid.readline()) == 0)
    fid.close()

    if axis is 0:
        feature_train = np.zeros((1, features.shape[1]))
        feature_test = np.zeros((1, features.shape[1]))
        label_train = []
        label_test = []
        for per in per_fea_num_train.keys():
            feature_train = np.hstack(feature_train, features[per])
            label_train = [per_fea_num_train.key().index(per)] * \
                per_fea_num_train.get(per)
            # QA
            assert(per_fea_num_train.get(per) == features[per].shape[0])
        feature_train = np.delete(feature_train, 0)
        label_train = np.array(label_train)

        for per in per_fea_num_test.get(per):
            feature_test = np.hstack(feature_test, features[per])
            label_test = [per_fea_num_test.key().index(per)] * \
                per_fea_num_test.get(per)
            # QA
            assert(per_fea_num_test.get(per) == features[per].shape[0])
        feature_test = np.delete(feature_test, 0)
        label_test = np.array(label_test)
        return dict(zip(["train", "test"], [(feature_train, label_train),
                    [feature_test, label_test]]))

    elif axis is 1:
        feature_train = np.zeros((features.shape[1], 1))
        feature_test = np.zeros((features.shape[1], 1))
        label_train = []
        label_test = []
        for per in per_fea_num_train.keys():
            feature_train = np.vstack(feature_train, features[per])
            label_train = [per_fea_num_train.key().index(per)] * \
                per_fea_num_train.get(per)
            # QA
            assert(per_fea_num_train.get(per) == features[per].shape[1])
        feature_train = np.delete(feature_train, 0, 1)
        label_train = np.array(label_train)

        for per in per_fea_num_test.get(per):
            feature_test = np.vstack(feature_test, features[per])
            label_test = [per_fea_num_test.key().index(per)] * \
                per_fea_num_test.get(per)
            # QA
            assert(per_fea_num_test.get(per) == features[per].shape[1])
        feature_test = np.delete(feature_test, 0, 1)
        label_test = np.array(label_test)
        return dict(zip(["train", "test"], [(feature_train, label_train),
                    [feature_test, label_test]]))

    else:
        print "The input parameter 'axis' is limited to 0 and 1"
        return False

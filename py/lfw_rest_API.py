# Designed by Junbo Zhao, *  * Designed by Junbo Zhao,
# working in Center for Intelligent Images and Document
# Processing Lab, Tsinghua University.

import os
import sys
import numpy as np
# Note that this script can be used as an API for processing
# the Labeled Faces in Wild benchmark.
# Regularly building cross-validation training and testing dataset.
# 10 Fold


def restricted_lfw_API(filename, fold, features, axis=0):
# axis: INPUT and OUTPUT features are stored column-ordred or row-ordred.
# And axis = 0 meansrow orded, axis = 1 means column-ordred.
# The default setting is axis = 0.
    # Read the whole dataset
    if not isinstance(features, dict):
        print "The featuers (para NO.3) must be a dictionry data, whose keys\
                    indicate the people's name and values are extracted\
                    feature vectors"
        return False
    if not os.path.isfile(filename):
        return False
    fid = file(filename)
    Nfold = int(fid.readline().split()[0])
    Npos_neg = int(fid.readline().split()[1])
    per_fea_num = [dict()] * Nfold
    per_fea_num_train = dict()
    per_fea_num_test = dict()
    for i in range(Nfold):
        pos_pair = []
        neg_pair = []
        pos_sample_num = []
        neg_sample_num = []
        for j in range(Npos_neg):
            pos_line = fid.readline()
            pos_line = pos_line.split()
            pos_pair.append(pos_line[0])
            pos_sample_num.append([int(pos_line[1], int(pos_line[2]))])
        for j in range(Npos_neg):
            neg_line = fid.readline()
            neg_line = neg_line.split()
            neg_pair.append([neg_line[0], neg_line[2]])
            neg_sample_num.append([int(neg_line[1], int(neg_line[3]))])
        per_fea_num[i] = dict(zip(pos_pair, pos_sample_num)).items() + \
            dict(zip(neg_pair, neg_sample_num)).items()
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
            if len(per) is 1:
                feature_train = np.hstack(
                    feature_train,
                    features[per][per_fea_num_train.get(per), :])
            if len(per) is 2:
                feature_train = np.hstack(
                    feature_train,
                    features[per[0]][per_fea_num_train.get(per)[0], :])
                feature_train = np.hstack(
                    feature_train,
                    features[per[1]][per_fea_num_train.get(per)[1], :])
            else:
                sys.exit(-1)
        feature_train = np.delete(feature_train, 0)
        label_train = [1] * Npos_neg + [-1] * Npos_neg

        for per in per_fea_num_test.keys():
            if len(per) is 1:
                feature_test = np.hstack(
                    feature_test,
                    features[per][per_fea_num_test.get(per), :])
            if len(per) is 2:
                feature_test = np.hstack(
                    feature_test,
                    features[per[0]][per_fea_num_test.get(per)[0], :])
                feature_test = np.hstack(
                    feature_test,
                    features[per[1]][per_fea_num_test.get(per)[1], :])
            else:
                sys.exit(-1)
        feature_test = np.delete(feature_test, 0)
        label_test = [1] * Npos_neg + [-1] * Npos_neg
        return dict(zip(["train", "test"], [(feature_train, label_train),
                    [feature_test, label_test]]))

    elif axis is 1:
        feature_train = np.zeros((features.shape[1], 1))
        feature_test = np.zeros((features.shape[1], 1))
        label_train = []
        label_test = []
        for per in per_fea_num_train.keys():
            if len(per) is 1:
                feature_train = np.vstack(
                    feature_train,
                    features[per][:, per_fea_num_train.get(per)])
            if len(per) is 2:
                feature_train = np.vstack(
                    feature_train,
                    features[per[0]][:, per_fea_num_train.get(per)[0]])
                feature_train = np.hstack(
                    feature_train,
                    features[per[1]][:, per_fea_num_train.get(per)[1]])
            else:
                sys.exit(-1)
        feature_train = np.delete(feature_train, 0, 1)
        label_train = [1] * Npos_neg + [-1] * Npos_neg

        for per in per_fea_num_test.keys():
            if len(per) is 1:
                feature_test = np.vstack(
                    feature_test,
                    features[per][:, per_fea_num_test.get(per)])
            if len(per) is 2:
                feature_test = np.vstack(
                    feature_test,
                    features[per[0]][:, per_fea_num_test.get(per)[0]])
                feature_test = np.hstack(
                    feature_test,
                    features[per[1]][:, per_fea_num_test.get(per)[1]])
            else:
                sys.exit(-1)
        feature_test = np.delete(feature_test, 0, 1)
        label_test = [1] * Npos_neg + [-1] * Npos_neg
        return dict(zip(["train", "test"], [(feature_train, label_train),
                    [feature_test, label_test]]))
    else:
        print "The input parameter 'axis' is limited to 0 and 1"
        return False

#!/usr/bin/env python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import xgboost
from sklearn.feature_selection import f_classif, f_regression, SelectKBest, chi2
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import export_graphviz

from sklearn.pipeline import Pipeline

from scipy.spatial.distance import mahalanobis
from sqlalchemy import create_engine
from tqdm import tqdm
import math

import pandas            as pd
import numpy             as np
import matplotlib.pyplot as plt
import _pickle           as cPickle
from sklearn.neighbors import KDTree
import itertools
import argparse
import pydot

import torch
import torch.optim

from commonModel import FLOAT_COLUMNS, INT_COLUMNS, STR_COLUMNS, TARGET_COLUMN, QuantileRegressionLoss, \
    HuberRegressionLoss

import matplotlib

# matplotlib.use('Agg')
pd.options.display.max_columns = 25
pd.options.display.max_rows = 400
pd.options.display.max_colwidth = 256


def XGBoost(dataFrame, targetColumn, featureNames):
    dataFrame_short = dataFrame[featureNames]
    FEATURE_NAMES = list(dataFrame_short.columns);
    FEATURE_NAMES.remove(targetColumn)
    COLUMNS = list(dataFrame_short.columns);
    LABEL = targetColumn;

    Y_dataFrame = dataFrame_short[[targetColumn]];
    Y_values = Y_dataFrame.values;
    X_dataFrame = dataFrame_short.drop(targetColumn, axis=1);
    X_values = X_dataFrame.values;
    Y_values = Y_values

    print(X_dataFrame.describe())

    FEATURE_DEFAULTS = ((X_dataFrame.max() + X_dataFrame.min()) * 0.5).to_dict()

    # preprocessorY = MinMaxScaler()
    # preprocessorY = StandardScaler()
    preprocessorY = MaxAbsScaler()
    preprocessorY.fit(Y_values)
    preprocessorX = MinMaxScaler()
    # preprocessorX = StandardScaler()
    preprocessorX.fit(X_values)

    Y_values = preprocessorY.transform(Y_values)
    X_values = preprocessorX.transform(X_values)
    Y_values_indexes = np.arange(0, len(Y_values), 1)
    X_numpyTrainVal, X_numpyTest, Y_numpyTrainVal_indexes, Y_numpyTest_indexes = train_test_split(X_values,
                                                                                                  Y_values_indexes,
                                                                                                  test_size=0.1)
    Y_numpyTrainVal = Y_values[Y_numpyTrainVal_indexes]
    Y_numpyTest = Y_values[Y_numpyTest_indexes]

    FEATURE_DEFAULTS = ((X_dataFrame.max() + X_dataFrame.min()) * 0.5).to_dict()
    model = xgboost.XGBRegressor(max_depth=20, objective='reg:squarederror', n_estimators=120, learning_rate=0.1,
                                 n_jobs=-1)
    model.fit(X_numpyTrainVal, Y_numpyTrainVal)  # обучение
    Y_numpyPredict = model.predict(X_numpyTest)

    X_numpyTotal = X_values
    Y_numpyTotal = Y_values
    eps = 0.001
    Y_relErr = np.abs(Y_numpyPredict - Y_numpyTest.flatten()) / (Y_numpyTest.flatten() + eps)

    for threshold in [0.025, 0.05, 0.10, 0.15]:
        bad_s = np.sum((Y_relErr > threshold))
        good_s = np.sum((Y_relErr <= threshold))
        total_s = Y_relErr.size
        print("threshold = {:5}, good = {:10}, bad = {:10}, err = {:4}".format(threshold, good_s,
                                                                               bad_s,
                                                                               bad_s / (good_s + bad_s)))


    Y_numpyPredict = preprocessorY.inverse_transform(Y_numpyPredict.reshape(-1, 1))
    Y_numpyTest = preprocessorY.inverse_transform(Y_numpyTest.reshape(-1, 1))
    modelPacket = dict()
    modelPacket['model'] = model
    modelPacket['preprocessorX'] = preprocessorX
    modelPacket['preprocessorY'] = preprocessorY

    modelPacket['feature_names'] = FEATURE_NAMES
    modelPacket['feature_defaults'] = FEATURE_DEFAULTS
    threshold = 10
    print()
    Y_relativeError = np.abs(Y_numpyPredict - Y_numpyTest) * 100 / Y_numpyTest

    allValues = dataFrame.loc[Y_numpyTest_indexes]
    mask = Y_relativeError > threshold
    badValues = allValues[mask]
    mask = Y_relativeError <= threshold
    goodValues = allValues[mask]
    #print(badValues)
    f_bad_values = open("bad_values.txt",'w')
    f_bad_values.write(str(badValues[featureNames+['source_url']]))
    f_bad_values.close()
    allValues = Y_numpyTest
    mask = Y_relativeError > threshold
    badValues = Y_numpyTest[mask]
    mask = Y_relativeError <= threshold
    goodValues = Y_numpyTest[mask]
    bins = range(1, 20)
    bins = [i * 0.5e6 for i in bins]

    figure, axes = plt.subplots(3, 1)
    axes[1].axis('tight')
    axes[1].axis('off')

    resultValues = axes[0].hist([allValues, goodValues, badValues], bins=bins, histtype='bar',
                                color=['green', 'yellow', 'red'])
    allValues = resultValues[0][0];
    goodValues = resultValues[0][1];
    badValues = resultValues[0][2];

    accuracy = goodValues * 100 / (allValues + 0.01)
    col_label = ['{:5d}'.format(int((bins[i + 0] + bins[i + 1]) / 2)) for i in range(len(bins) - 1)]
    cell_text = [['{:2.1f}'.format(acc_) for acc_ in accuracy], ]

    table_ = axes[1].table(cellText=cell_text, colLabels=col_label, loc='center')
    table_.auto_set_font_size(False)
    table_.set_fontsize(8)

    Y_numpyTest_max = np.max(Y_numpyTest)
    Y_numpyTest_min = np.min(Y_numpyTest)

    # axes[2].set_position([Y_numpyTotal_min-Y_numpyTotal_width*0.1,Y_numpyTotal_min-Y_numpyTotal_width*0.1,Y_numpyTotal_width*0.2,Y_numpyTotal_width*0.2])
    axes[2].plot(Y_numpyTest, Y_numpyTest, c='blue')
    axes[2].plot(Y_numpyTest, Y_numpyTest * (1.0 + 0.1), c='red')
    axes[2].plot(Y_numpyTest, Y_numpyTest * (1.0 - 0.1), c='red')
    axes[2].scatter(Y_numpyPredict, Y_numpyTest)
    plt.show()

    # figure, axes =plt.subplots(3,1)
    # clust_data = np.random.random((10,3))
    # collabel=("col 1", "col 2", "col 3")
    # axs[0].axis('tight')
    # axs[0].axis('off')
    # the_table = axs[0].table(cellText=clust_data,colLabels=collabel,loc='center')

    # axs[1].plot(clust_data[:,0],clust_data[:,1])
    # plt.show()

    return modelPacket, (Y_numpyPredict, Y_numpyTotal)




parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default="")

parser.add_argument("--database", type=str,
                    default="mysql://root:password@188.120.245.195:3306/domprice_dev1_v2")
parser.add_argument("--table", type=str, default="src_ads_raw_16_processed")

parser.add_argument("--model", type=str, default="XGBoost_Kazan.pkl")

parser.add_argument("--output", type=str, default="")
parser.add_argument("--limits", type=str, default="input/NizhnyNovgorodLimits.json")

parser.add_argument("--features", type=str,
                    default="price,latitude,longitude,floor_number,total_square,number_of_rooms,number_of_floors,exploitation_start_year")
# parser.add_argument("--verbose" , action="store_true" )
parser.add_argument("--verbose", type=bool, default="True")
args = parser.parse_args()
inputFileName = args.input  # NizhniyNovgorod.csv
inputDatabase = args.database
inputTable = args.table  # nn

modelFileName = args.model
outputFileName = args.output
limitsFileName = args.limits

featureNames = (args.features).split(',')
verboseFlag = args.verbose

trainDataFrame = None
if inputDatabase != "" and inputTable != "":
    engine = create_engine(inputDatabase)
    trainDataFrame = pd.read_sql_table(inputTable, engine)
if verboseFlag:
    print(trainDataFrame.describe())

# trainDataFrame = trainDataFrame.select_dtypes(include=['number'])

trainedModelPacket, (Y_predict, Y_test) = XGBoost(trainDataFrame, TARGET_COLUMN, featureNames)
with open(modelFileName, 'wb') as fid:
    cPickle.dump(trainedModelPacket, fid)

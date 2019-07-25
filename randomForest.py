#!/usr/bin/env python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn.feature_selection import f_classif, f_regression, SelectKBest, chi2
from sklearn.ensemble          import IsolationForest
from sklearn.neighbors         import LocalOutlierFactor

from sklearn.model_selection   import train_test_split
from sklearn.model_selection   import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble          import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics           import mean_squared_error, mean_absolute_error, median_absolute_error

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model  import LinearRegression

from sklearn.preprocessing     import QuantileTransformer
from sklearn.preprocessing     import LabelEncoder
from sklearn.preprocessing     import MinMaxScaler, StandardScaler, MaxAbsScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score
from sklearn.neighbors         import KNeighborsRegressor
from sklearn.tree              import export_graphviz

from sklearn.pipeline          import Pipeline

from scipy.spatial.distance    import mahalanobis
from sqlalchemy import create_engine

import math

import pandas            as pd
import numpy             as np
import matplotlib.pyplot as plt
import _pickle           as cPickle

import itertools
import argparse
import pydot

import torch
import torch.optim

from commonModel import FLOAT_COLUMNS, INT_COLUMNS, STR_COLUMNS, TARGET_COLUMN, QuantileRegressionLoss, HuberRegressionLoss
from commonModel import limitDataUsingLimitsFromFilename
from commonModel import limitDataUsingProcentiles
from commonModel import ImprovedLinearNet, LinearNet
#from preProcessDatabase_scratch import loadDataFrame, clearDataFromAnomalies

import matplotlib
#matplotlib.use('Agg')
def randomForest( dataFrame, targetColumn, featureNames ):
    dataFrame = dataFrame[featureNames]

    FEATURE_NAMES = list(dataFrame.columns);
    FEATURE_NAMES.remove(targetColumn)
    COLUMNS = list(dataFrame.columns);
    LABEL = targetColumn;

    Y_dataFrame = dataFrame[[targetColumn]];
    Y_values = Y_dataFrame.values;
    X_dataFrame = dataFrame.drop(targetColumn, axis=1);
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
    X_numpyTrainVal, X_numpyTest, Y_numpyTrainVal, Y_numpyTest = train_test_split(X_values, Y_values, test_size=0.1)
    model = RandomForestRegressor(n_estimators=10, n_jobs = -1)
    model.fit(X_numpyTrainVal, Y_numpyTrainVal)  # обучение
    Y_numpyPrediction = model.predict(X_numpyTest)  # предсказание
    Accuracy = np.abs((Y_numpyPrediction - Y_numpyTest.flatten()))/Y_numpyTest

    print(Accuracy.mean())

parser = argparse.ArgumentParser()
parser.add_argument("--input"   , type=str, default="" )

parser.add_argument("--database", type=str, default="mysql://root:password" )
parser.add_argument("--table"   , type=str, default="src_ads_raw_52_processed" )

parser.add_argument("--model"  , type=str, default="random_forest" )

parser.add_argument("--output" , type=str, default="" )
parser.add_argument("--limits" , type=str, default="input/NizhnyNovgorodLimits.json" )

parser.add_argument("--features", type=str, default="price,latitude,longitude,floor_number,total_square,number_of_rooms,number_of_floors,exploitation_start_year")
#parser.add_argument("--verbose" , action="store_true" )
parser.add_argument("--verbose", type=bool, default="True")
args = parser.parse_args()
inputFileName = args.input    #NizhniyNovgorod.csv
inputDatabase = args.database 
inputTable    = args.table    #nn

modelFileName  = args.model
outputFileName = args.output
limitsFileName = args.limits


featureNames   = (args.features).split(',')
verboseFlag    = args.verbose

trainDataFrame = None
if inputDatabase != "" and inputTable != "" :
    engine = create_engine(inputDatabase)
    trainDataFrame = pd.read_sql_table(inputTable, engine)
if verboseFlag :
    print( trainDataFrame.describe() )


trainDataFrame = trainDataFrame.select_dtypes(include=['number'])

if modelFileName == "random_forest" :
	randomForest(trainDataFrame, TARGET_COLUMN, featureNames)

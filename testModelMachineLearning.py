#!/usr/bin/env python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error

import pandas            as pd
import numpy             as np
import matplotlib.pyplot as plt
import _pickle           as cPickle

import itertools
import datetime
import argparse
import types
import torch
import timeit
import math
import sys

from sqlalchemy import create_engine

from commonModel import FLOAT_COLUMNS, INT_COLUMNS, STR_COLUMNS, TARGET_COLUMN
from commonModel import limitDataUsingLimitsFromFilename
from commonModel import limitDataUsingProcentiles

from commonModel import LinearNet

pd.set_option('display.width', 500)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="XGBoost/XGBoost_Kazan.pkl")
parser.add_argument("--query", type=str, default="longitude=49.085699, latitude=55.834972, total_square=33.6, living_square=22.6, kitchen_square=9.6, number_of_rooms=2, floor_number=2, number_of_floors=3, exploitation_start_year=1953")
parser.add_argument("--limits", type=str, default="input/KazanLimits.json")
parser.add_argument("--price", type=int, default=0)

parser.add_argument("--database", type=str, default="mysql://root:UWmnjxPdN5ywjEcN@188.120.245.195:3306/domprice_dev1_v2")
parser.add_argument("--table", type=str, default="src_ads_raw_52_processed")
parser.add_argument("--verbose", action="store_true")


def getClosestItemsInDatabase(inputSeries, inputDataBase, inputTable, inputTolerances):
    engine = create_engine(inputDataBase)

    inputTolerancesFields = set(inputTolerances.keys())
    inputDataFrameFields = set(inputSeries.index)

    processedTolerances = {field: inputTolerances[field] for field in
                           inputTolerancesFields.intersection(inputDataFrameFields)}

    processedLimits = dict()
    for (field, tolerance) in processedTolerances.items():
        processedLimits[field] = (inputSeries[field] - abs(tolerance), inputSeries[field] + abs(tolerance))

    sql_query = """SELECT * FROM {} WHERE """.format(inputTable)
    sql_query += """ AND """.join(
        "{1} <= {0} AND {0} <= {2}".format(field, min_value, max_value) for (field, (min_value, max_value)) in
        processedLimits.items())

    resultValues = pd.read_sql_query(sql_query, engine)
    subset = ['price', 'total_square', 'number_of_rooms']
    resultValues.drop_duplicates(subset=subset, keep='first', inplace=True)

    return resultValues


def getTopKClosestItems(inputItem, closestItem_s, PREPROCESSOR_X, MODEL_FEATURE_NAMES, topk=5):
    if not closestItem_s.empty:
        processedInputItem = inputItem[MODEL_FEATURE_NAMES]
        processedClosestItem_s = closestItem_s[MODEL_FEATURE_NAMES]

        processedInputItem_numpy = processedInputItem.values
        processedClosestItem_s_numpy = processedClosestItem_s.values
        processedInputItem_numpy = processedInputItem_numpy.reshape(1, -1)

        processedInputItem_numpy = PREPROCESSOR_X.transform(processedInputItem_numpy);
        processedClosestItem_s_numpy = PREPROCESSOR_X.transform(processedClosestItem_s_numpy);

        processedResult_s_numpy = processedClosestItem_s_numpy - processedInputItem_numpy
        processedResult_s_numpy = np.linalg.norm(processedResult_s_numpy, axis=1)

        index_s = processedResult_s_numpy.argsort()[:topk]
        return closestItem_s.iloc[index_s]
    else:
        return closestItem_s


def processClosestItems(inputItem, closestItem_s, predictedPrice, verboseFlag=False):
    RESULT_PRICE_S = dict();
    RESULT_PRICE_S['predictedPrice'] = int(predictedPrice.values[0])
    RESULT_PRICE_S['medianPrice'] = int(0)
    RESULT_PRICE_S['meanPrice'] = int(0)
    RESULT_PRICE_S['maxPrice'] = int(0)
    RESULT_PRICE_S['minPrice'] = int(0)

    if not closestItem_s.empty:
        # Calculate required prices
        # inputPrice           = predictedPrice.values[0]
        inputSquare = inputItem['total_square']
        inputFloorNumber = inputItem['floor_number']
        inputNumberOfFloors = inputItem['number_of_floors']
        # inputPricePerSquare  = inputPrice/inputSquare

        closestPrice_s = np.array(list(map(float, closestItem_s['price'].values)))
        closestSquare_s = np.array(list(map(float, closestItem_s['total_square'].values)))
        closestFloorNumber_s = np.array(list(map(float, closestItem_s['floor_number'].values)))
        closestNumberOfFloors_s = np.array(list(map(float, closestItem_s['number_of_floors'].values)))
        closestPricePerSquare_s = closestPrice_s / closestSquare_s

        pricePerSquareMedian = np.median(closestPricePerSquare_s)
        pricePerSquareMean = np.mean(closestPricePerSquare_s)
        pricePerSquareMax = np.max(closestPricePerSquare_s)
        pricePerSquareMin = np.min(closestPricePerSquare_s)

        RESULT_PRICE_S['predictedPrice'] = int(predictedPrice.values[0])
        RESULT_PRICE_S['medianPrice'] = int(pricePerSquareMedian * inputSquare)
        RESULT_PRICE_S['meanPrice'] = int(pricePerSquareMean * inputSquare)
        RESULT_PRICE_S['maxPrice'] = int(pricePerSquareMax * inputSquare)
        RESULT_PRICE_S['minPrice'] = int(pricePerSquareMin * inputSquare)

    return RESULT_PRICE_S


def testMachineLearningModel(Model, preprocessorX, preprocessorY, dataFrame, droppedColumns=[]):
    import warnings
    warnings.filterwarnings('ignore')

    dataFrame.drop(labels=droppedColumns, inplace=True)

    X_dataFrame = dataFrame;
    X_numpy = X_dataFrame.values;

    X_numpy = X_numpy.reshape(1, -1)
    X_numpy = preprocessorX.transform(X_numpy)

    Y_numpy = Model.predict(X_numpy)

    Y_numpy = preprocessorY.inverse_transform(Y_numpy.reshape(1, -1))

    return pd.DataFrame(Y_numpy)


def testModel(Model, dataFrame):
    import warnings
    warnings.filterwarnings('ignore')

    X_dataFrame = dataFrame;
    index = X_dataFrame.index;
    X_values = X_dataFrame.values;
    Y_values = np.array(Model.predict(X_values))

    return pd.DataFrame(data=Y_values, index=index, columns=['price'])


args = parser.parse_args()

inputQuery = args.query
modelFileName = args.model

limitsFileName = args.limits
inputPrice = args.price

databaseName = args.database
tableName = args.table
verboseFlag = args.verbose

# Load tne model
with open(modelFileName, 'rb') as fid:
    modelPacket = cPickle.load(fid)

    REGRESSION_MODEL = modelPacket['model']
    PREPROCESSOR_X = modelPacket['preprocessorX']
    PREPROCESSOR_Y = modelPacket['preprocessorY']
    MODEL_FEATURE_NAMES = modelPacket['feature_names']
    MODEL_FEATURE_DEFAULTS = modelPacket['feature_defaults']

# Process query
userQuery = eval("dict({})".format(inputQuery))
defaultQuery = MODEL_FEATURE_DEFAULTS
inputQuery = defaultQuery;
inputQuery.update(userQuery)

inputDataFrame = pd.DataFrame(data=inputQuery, index=[0])

inputDataFrame = limitDataUsingLimitsFromFilename(inputDataFrame, limitsFileName)
inputDataSize = len(inputDataFrame.index)

if inputDataSize > 0:  # Check that input data is correct
    for i in range(inputDataSize):
        inputItem = inputDataFrame.iloc[i]
        inputItemForModel = inputDataFrame[MODEL_FEATURE_NAMES].iloc[i]

        predicted_Y = testMachineLearningModel(REGRESSION_MODEL, PREPROCESSOR_X,
                                                                           PREPROCESSOR_Y, inputItemForModel)

        predicted_dS = pd.DataFrame(PREPROCESSOR_X.scale_.reshape(1, -1), columns=MODEL_FEATURE_NAMES)

        print("Predicted price:  {:9.2f}".format(predicted_Y.iloc[0].values[0]))
        predicted_dS_ = predicted_dS.iloc[0].to_dict()
        print(
            "Predicted scales:" + ",".join([" {}={:7.4f}".format(key, value) for key, value in predicted_dS_.items()]))

        if verboseFlag:
            inputTolerances = {name: abs(values[0]) for name, values in predicted_dX.iteritems()}
            closestItem_s = getClosestItemsInDatabase(inputItem, databaseName, tableName, inputTolerances)
            closestItem_s = getTopKClosestItems(inputItem, closestItem_s, PREPROCESSOR_X, MODEL_FEATURE_NAMES, topk=5)
            RESULT_PRICE_S = processClosestItems(inputItem, closestItem_s, predicted_Y, verboseFlag=verboseFlag)

            print("Closest   items: ")
            for _, closestItem in closestItem_s[MODEL_FEATURE_NAMES + ['price', ]].iterrows():
                print(closestItem)
                print("")

            print("Median    price: {:,}".format(RESULT_PRICE_S['medianPrice']))
            print("Mean      price: {:,}".format(RESULT_PRICE_S['meanPrice']))
            print("Max       price: {:,}".format(RESULT_PRICE_S['maxPrice']))
            print("Min       price: {:,}".format(RESULT_PRICE_S['minPrice']))

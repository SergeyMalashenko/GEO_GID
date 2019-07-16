#!/usr/bin/env python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import pandas            as pd
import numpy             as np
from sklearn.preprocessing      import RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors

from sqlalchemy import create_engine
from datetime   import datetime

import argparse
import json
import time
import math

from commonModel import limitDataUsingProcentiles
from commonModel import FLOAT_COLUMNS, INT_COLUMNS, STR_COLUMNS, DATE_COLUMNS, TARGET_COLUMN

from collections import Counter

all_columns = FLOAT_COLUMNS+INT_COLUMNS
pd.options.display.max_columns = len(all_columns)
class loadDataFrame(object):
    def __call__(self, databasePath, limitsFileName, tableName):
        engine = create_engine(databasePath, encoding='utf8')
        with open(limitsFileName, "r") as read_file:
            processedLimits = json.load(read_file)

        sql_query = """SELECT * FROM {} WHERE """.format(tableName)

        sql_query += """ AND """.join(
            "{1} <= {0} AND {0} <= {2}".format(
                field, processedLimits[field]['min'], processedLimits[field]['max']) for field in
            processedLimits.keys())

        if 'floor_number' and 'number_of_floors' in processedLimits.keys():
            sql_query += """ AND floor_number <= number_of_floors """
        if 'total_square' and 'living_square' and 'kitchen_square' in processedLimits.keys():
            sql_query += """ AND living_square + kitchen_square <= total_square""".format(tableName)

        resultValues = pd.read_sql_query(sql_query, engine)

        if 'publication_date' in resultValues.columns :
            resultValues = resultValues.sort_values(by=['publication_date'])

        subset = ['longitude', 'latitude', 'total_square', 'number_of_rooms', 'number_of_floors', 'floor_number']
        resultValues.drop_duplicates(subset=subset, keep='last', inplace=True)
        return resultValues

def clearDataFromAnomalies( inputDataFrame ):
    clf = IsolationForest(n_estimators=20, max_features=(len(all_columns)-2), n_jobs=-1)
    inputDataFrame['pricePerSquare'] = (inputDataFrame['price'] / inputDataFrame['total_square'])
    all_columns_new = all_columns+['pricePerSquare']
    all_columns_new.remove('distance_to_metro')
    all_columns_new.remove('floor_number')

    all_columns_new.remove('price')

    inputDataFrame_matrix = inputDataFrame[all_columns_new].to_numpy()
    clf.fit(inputDataFrame_matrix)
    inputDataFrame_scores = clf.predict(inputDataFrame_matrix)
    f_anomalies = open("anomalies.txt", 'w')
    indexes_to_delete = []
    indexes_inlier = []
    inputDataFrameMedians = inputDataFrame[all_columns_new].median()
    outlierDataFrame =  inputDataFrame.copy()
    for i in range(0, len(inputDataFrame_scores)):
        data_score = inputDataFrame_scores[i]
        if data_score < 0:
            deviation_of_outlier_max = -1
            for feature in (all_columns_new):
                deviation_of_outlier_current = math.fabs((inputDataFrame.iloc[i][feature]-inputDataFrameMedians[feature])/inputDataFrame.iloc[i][feature])
                if deviation_of_outlier_current > deviation_of_outlier_max:
                    deviation_of_outlier_max = deviation_of_outlier_current
                    dev_feature = feature
            f_anomalies.write('House fias id: {0}, reason: {1} is {2}, when median is {3}.\n'.format(str(inputDataFrame.iloc[i]['fias_id']), str(dev_feature), str(inputDataFrame.iloc[i][dev_feature]),str(inputDataFrameMedians[dev_feature])))
            indexes_to_delete.append(i)
        if data_score > 0:
            indexes_inlier.append(i)
    inputDaraFrame = inputDataFrame.drop(inputDataFrame.index[indexes_to_delete])
    outlierDataFrame = outlierDataFrame.drop(outlierDataFrame.index[indexes_inlier])
    outlierDataFrame[all_columns_new].hist(bins=100)
    plt.show()

    f_anomalies.close()

    #neigh = NearestNeighbors(n_jobs=-1)
    #neigh.fit(inputDaraFrame[all_columns_new])
    #outlierDataFrame_subset=outlierDataFrame[['longitude','latitude']].query('56.307 < longitude and longitude < 56.32 and 43.98 < latitude and latitude > 44.03')
    #print(neigh.kneighbors(outlierDataFrame_subset,return_distance=False))
    return inputDaraFrame


parser = argparse.ArgumentParser()
#parser.add_argument("--database", type=str, default="mysql://sr:A4y8J6r4@149.154.71.73:3310/sr_dev" )
#parser.add_argument("--database", type=str, default="mysql://sr:password@portal.smartrealtor.pro:3306/smartrealtor" )
#parser.add_argument("--database", type=str, default="mysql://root:Intemp200784@127.0.0.1/smartRealtor" )
#parser.add_argument("--database", type=str, default="mysql://root:Intemp200784@127.0.0.1/smartRealtor?unix_socket=/var/run/mysqld/mysqld.sock" )
parser.add_argument("--database", type=str, default="mysql://root:UWmnjxPdN5ywjEcN@188.120.245.195:3306/domprice_dev1_v2" )
#parser.add_argument("--input_table"   , type=str, default="real_estate_from_ads_api" )
parser.add_argument("--input_table"   , type=str, default="src_ads_raw" )
#parser.add_argument("--limits"  , type=str, default="input/MoscowLimits.json" )
parser.add_argument("--output_table"   , type=str, default="real_estate_from_ads_api_processed" )
#parser.add_argument("--limits"  , type=str, default="input/KazanLimits.json" )
parser.add_argument("--limits"  , type=str, default="input/NizhnyNovgorodLimits.json" )
#parser.add_argument("--limits"  , type=str, default="input/SaintPetersburgLimits.json" )
args = parser.parse_args()

input_tableName = args.input_table
databaseName = args.database
limitsName   = args.limits

inputDataFrame = None
inputDataFrame = loadDataFrame()(databaseName, limitsName, input_tableName )


print("Statistics of load data frame:")
print(inputDataFrame[all_columns].describe())


inputDataFrame = limitDataUsingProcentiles(inputDataFrame)

print("Statistics of data frame limited using procentiles:")
print(inputDataFrame[all_columns].describe())

inputDataFrame = clearDataFromAnomalies(inputDataFrame)

print("Statistics of data frame after cleaning from anomalies:")
print(inputDataFrame[all_columns].describe())

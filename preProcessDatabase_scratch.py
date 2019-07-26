#!/usr/bin/env python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import pandas            as pd
import numpy             as np
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import LocalOutlierFactor
from sqlalchemy import create_engine
from datetime import datetime

import argparse
import json
import time
import math

from commonModel import limitDataUsingProcentiles
from commonModel import FLOAT_COLUMNS, INT_COLUMNS, STR_COLUMNS, DATE_COLUMNS, TARGET_COLUMN

from collections import Counter

all_columns = FLOAT_COLUMNS + INT_COLUMNS
pd.options.display.max_columns = len(all_columns)+5
pd.options.display.max_colwidth = 256

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

        if ('floor_number' in processedLimits.keys()) and ('number_of_floors' in processedLimits.keys()):
            sql_query += """ AND floor_number <= number_of_floors """
        if ('total_square' in processedLimits.keys()) and ('living_square' in processedLimits.keys()) and (
                'kitchen_square' in processedLimits.keys()):
            sql_query += """ AND living_square + kitchen_square <= total_square""".format(tableName)

        resultValues = pd.read_sql_query(sql_query, engine)

        if 'publication_date' in resultValues.columns:
            resultValues = resultValues.sort_values(by=['publication_date'])

        subset = ['longitude', 'latitude', 'total_square', 'number_of_rooms', 'number_of_floors', 'floor_number']
        resultValues.drop_duplicates(subset=subset, keep='last', inplace=True)
        return resultValues


def clearDataFromAnomalies(inputDataFrame):
    clf = LocalOutlierFactor(contamination=0.05,n_jobs=-1)
    inputDataFrame['pricePerSquare'] = (inputDataFrame['price'] / inputDataFrame['total_square'])

    all_columns_new = all_columns + ['pricePerSquare']

    all_columns_new.remove('distance_to_metro')
    all_columns_new.remove('floor_number')
    all_columns_new.remove('price')

    inputDataFrame_numpy = inputDataFrame[all_columns_new].to_numpy()
    inputDataFrame_scores = clf.fit_predict(inputDataFrame_numpy)
    """
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
    f_anomalies.close()
    """
    inliersDataFrame = inputDataFrame[inputDataFrame_scores == 1]
    outliersDataFrame = inputDataFrame[inputDataFrame_scores == -1]
    outliersDataFrame[all_columns_new].plot.scatter('longitude','latitude')
    outliersDataFrame[all_columns_new].hist(bins=100)

    f_outliers = open("outliers.txt", "w")
    f_outliers.write(str(outliersDataFrame[['source_url']]))
    f_outliers.flush()
    f_outliers.close()
    #outliersDataFrame.groupby(outliersDataFrame.publication_date.dt.month).count().plot(kind="bar")
    #print( outliersDataFrame[all_columns_new + ['publication_date']].describe( include=['number','datetime']) )

    plt.show()

    neigh = NearestNeighbors(algorithm='kd_tree',n_jobs=-1)
    neigh.fit(inliersDataFrame[all_columns_new])
    #outliersDataFrame_subset = outliersDataFrame.query('2012 < exploitation_start_year and exploitation_start_year < 2015')
    outliersDataFrame_subset=outliersDataFrame
    #print(outliersDataFrame_subset)
    outliersNeighbors=neigh.kneighbors(outliersDataFrame_subset[all_columns_new],return_distance=False)
    f_neighbors = open('outliers.txt','w')
    for i in range (0,len(outliersNeighbors)):
        #print('Outlier: {0}\n'.format(str(outliersDataFrame_subset.iloc[i][all_columns_new])))
        f_neighbors.write('Outlier:\n {0}\n'.format(str(outliersDataFrame_subset.iloc[i][all_columns+['source_url']])))
        #f_neighbors.write('Neighbors:\n {0}\n'.format(str(inliersDataFrame.iloc[outliersNeighbors[i]][all_columns+['pricePerSquare']])))
    f_neighbors.close()
    strangeHouses = outliersDataFrame_subset.query('exploitation_start_year < 1960 and number_of_floors > 9 or exploitation_start_year < 1980 and number_of_floors > 19 or exploitation_start_year < 1970 and number_of_floors > 14')
    print('Strange houses in outliers: {0}'.format(str(len(strangeHouses))))
    outliersDataFrame[all_columns_new + ['publication_date']].plot.scatter('exploitation_start_year', 'number_of_floors')
    plt.show()
    return inliersDataFrame


parser = argparse.ArgumentParser()

parser.add_argument("--database", type=str,
                    default="mysql://root:password@188.120.245.195:3306/domprice_dev1_v2")
# parser.add_argument("--input_table"   , type=str, default="real_estate_from_ads_api" )
parser.add_argument("--input_table", type=str, default="src_ads_raw_16")
# parser.add_argument("--limits"  , type=str, default="input/MoscowLimits.json" )
parser.add_argument("--output_table", type=str, default="")
parser.add_argument("--limits"  , type=str, default="input/KazanLimits.json" )
#parser.add_argument("--limits", type=str, default="input/NizhnyNovgorodLimits.json")
#parser.add_argument("--limits"  , type=str, default="input/SaintPetersburgLimits.json" )
args = parser.parse_args()

inputTableName = args.input_table
databaseName = args.database
limitsName = args.limits
outputTableName = args.output_table
inputDataFrame = None
inputDataFrame = loadDataFrame()(databaseName, limitsName, inputTableName)

print("Statistics of load data frame:")
print(inputDataFrame[all_columns].describe())

inputDataFrame = limitDataUsingProcentiles(inputDataFrame)
inputDataFrame['exploitation_start_year'].hist(bins=100)

if outputTableName == "":
    outputTableName = inputTableName + "_processed"

print(outputTableName)
#plt.show()


print("Statistics of data frame limited using procentiles:")
print(inputDataFrame[all_columns].describe())

strangeHouses = inputDataFrame.query('exploitation_start_year < 1960 and number_of_floors > 9 or exploitation_start_year < 1980 and number_of_floors > 19 or exploitation_start_year < 1970 and number_of_floors > 14')
print('Strange houses before cleaning anomalies: {0}'.format(str(len(strangeHouses))))

inputDataFrame = clearDataFromAnomalies(inputDataFrame)

print("Statistics of data frame after cleaning from anomalies:")
print(inputDataFrame[all_columns].describe())
engine = create_engine(databaseName, encoding='utf8')
inputDataFrame.to_sql(name=outputTableName,con=engine,if_exists='replace')

"""
f_strange_houses=open("strange_houses.txt","w")
strangeHouses = inputDataFrame.query('exploitation_start_year < 1960 and number_of_floors > 9 or exploitation_start_year < 1980 and number_of_floors > 19 or exploitation_start_year < 1970 and number_of_floors > 14')
pd.options.display.max_rows = len(strangeHouses)
f_strange_houses.write('{}\n'.format(str(strangeHouses[['longitude','latitude','new','exploitation_start_year','number_of_floors','fias_id','source_url']])))
"""

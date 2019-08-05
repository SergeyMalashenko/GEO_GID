#!/usr/bin/env python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import pandas            as pd
import numpy             as np

from sklearn.preprocessing import RobustScaler
from sklearn.ensemble      import IsolationForest
from sklearn.neighbors     import LocalOutlierFactor
from sklearn.neighbors     import NearestNeighbors
from sklearn.neighbors     import KDTree, BallTree
from sklearn.neighbors     import DistanceMetric
from sklearn.preprocessing import MinMaxScaler

from sqlalchemy import create_engine
from datetime   import datetime
from tqdm       import tqdm

import argparse
import json
import time
import math

from commonModel import limitDataUsingProcentiles
from commonModel import FLOAT_COLUMNS, INT_COLUMNS, STR_COLUMNS, DATE_COLUMNS, TARGET_COLUMN

from collections import Counter

all_columns = FLOAT_COLUMNS + INT_COLUMNS
pd.options.display.max_columns = len(all_columns)+5

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

class saveDataFrame(object):
    def __call__(self, dataFrame, databasePath, tableName):
        engine = create_engine(databasePath, encoding='utf8')
        engine.execute("DROP TABLE IF EXISTS {}".format( tableName ) )
        dataFrame.to_sql( tableName, con=engine )
"""
def clearDataFromAnomalies(inputDataFrame):
    clf = IsolationForest(behaviour='new',contamination=0.0125,n_jobs=-1)
    inputDataFrame['pricePerSquare'] = (inputDataFrame['price'] / inputDataFrame['total_square'])

    all_columns_new = all_columns + ['pricePerSquare']
    all_columns_new.remove('distance_to_metro')
    all_columns_new.remove('floor_number'     )
    all_columns_new.remove('price'            )
    
    inputDataFrame_numpy = inputDataFrame[all_columns_new].to_numpy()
    inputDataFrame_scores = clf.fit_predict(inputDataFrame_numpy)
    
    inliersDataFrame  = inputDataFrame[inputDataFrame_scores ==  1]
    outliersDataFrame = inputDataFrame[inputDataFrame_scores == -1]

    return inliersDataFrame, outliersDataFrame
"""
def clearDataFromAnomalies(inputDataFrame):
    clf = LocalOutlierFactor(contamination=0.05,n_jobs=-1, n_neighbors=10)
    inputDataFrame['pricePerSquare'] = (inputDataFrame['price'] / inputDataFrame['total_square'])

    all_columns_new = all_columns + ['pricePerSquare']
    all_columns_new.remove('distance_to_metro')
    all_columns_new.remove('floor_number')
    all_columns_new.remove('price')

    inputDataFrame_numpy = inputDataFrame[all_columns_new].to_numpy()
    inputDataFrame_scores = clf.fit_predict(inputDataFrame_numpy)
    
    inliersDataFrame  = inputDataFrame[inputDataFrame_scores ==  1]
    outliersDataFrame = inputDataFrame[inputDataFrame_scores == -1]
    
    return inliersDataFrame, outliersDataFrame

def smoothDataFrame( inputDataFrame, distance_limit = 0.005 ):
    outputDataFrame = inputDataFrame.copy()
    
    features = ['longitude','latitude','number_of_floors','number_of_rooms','exploitation_start_year']
    inputDataFeatures = inputDataFrame[ features ].values
    scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
    inputDataFeatures = scaler.fit_transform(inputDataFeatures)
    searchTree = KDTree( inputDataFeatures )
    
    totalSize = inputDataFeatures.shape[0] 
    
    totalMedianPricePerSquare_s = np.zeros( totalSize )
    totalMeanPricePerSquare_s   = np.zeros( totalSize )
    totalMedianPrice_s          = np.zeros( totalSize )
    totalMeanPrice_s            = np.zeros( totalSize )
    
    for index in tqdm( range( inputDataFeatures.shape[0] ) ):
        inputData = inputDataFeatures[index+0:index+1] 
        
        currentPrice                 = inputDataFrame['price'                  ].values[index]
        currentTotalSquare           = inputDataFrame['total_square'           ].values[index]
        currentKitchenSquare         = inputDataFrame['kitchen_square'         ].values[index]
        currentNumberOfRooms         = inputDataFrame['number_of_rooms'        ].values[index]
        currentNumberOfFloors        = inputDataFrame['number_of_floors'       ].values[index]
        currentExploitationStartYear = inputDataFrame['exploitation_start_year'].values[index]
        
        index_s, distance_s = searchTree.query_radius( inputData, distance_limit, count_only=False, return_distance=True )
        index_s    = index_s   [0].astype(np.int32  )
        distance_s = distance_s[0].astype(np.float32)
        #distance_s, index_s = searchTree.query( inputData, k=5 )
        #distance_s, index_s = searchTree.query( inputData, r=0.05, k=5 )
        
        Price_s                 = inputDataFrame['price'                  ].values[index_s] 
        totalSquare_s           = inputDataFrame['total_square'           ].values[index_s] 
        numberOfFloor_s         = inputDataFrame['number_of_floors'       ].values[index_s]
        numberOfRooms_s         = inputDataFrame['number_of_rooms'        ].values[index_s] 
        exploitationStartYear_s = inputDataFrame['exploitation_start_year'].values[index_s] 
        
        mask = distance_s < distance_limit;
        
        if np.sum( mask ) >= 4:
            totalPricePerSquare_s = Price_s / totalSquare_s
            
            totalMedianPricePerSquare = np.median( totalPricePerSquare_s )
            totalMeanPricePerSquare   = np.mean  ( totalPricePerSquare_s )
            
            totalMedianPricePerSquare_s[ index ] = totalMedianPricePerSquare 
            totalMeanPricePerSquare_s  [ index ] = totalMeanPricePerSquare  
            
            totalMedianPrice_s         [ index ] = totalMedianPricePerSquare*currentTotalSquare 
            totalMeanPrice_s           [ index ] = totalMeanPricePerSquare  *currentTotalSquare 
    
    mask = totalMedianPrice_s > 0
    outputDataFrame['price'] = totalMedianPrice_s
    outputDataFrame = outputDataFrame[ mask ]
    
    totalMedianPricePerSquare_s = totalMedianPricePerSquare_s[ mask ]
    totalMeanPricePerSquare_s   = totalMeanPricePerSquare_s  [ mask ]
    
    return outputDataFrame

parser = argparse.ArgumentParser()
parser.add_argument("--database"    , type=str, default="mysql://root:UWmnjxPdN5ywjEcN@188.120.245.195:3306/domprice_dev1_v2" )

parser.add_argument("--input_table" , type=str, default="src_ads_raw_52" )
parser.add_argument("--output_table", type=str, default="src_ads_raw_52_processed" )
parser.add_argument("--limits"      , type=str, default="input/NizhnyNovgorodLimits.json" )

#parser.add_argument("--input_table" , type=str, default="src_ads_raw_16" )
#parser.add_argument("--output_table", type=str, default="src_ads_raw_16_processed" )
#parser.add_argument("--limits"      , type=str, default="input/KazanLimits.json" )

#parser.add_argument("--input_table" , type=str, default="src_ads_raw_78" )
#parser.add_argument("--output_table", type=str, default="src_ads_raw_78_processed" )
#parser.add_argument("--limits"      , type=str, default="input/SaintPetersburgLimits.json" )

#parser.add_argument("--input_table" , type=str, default="src_ads_raw_77" )
#parser.add_argument("--output_table", type=str, default="src_ads_raw_77_processed" )
#parser.add_argument("--limits"      , type=str, default="input/MoscowLimits.json" )

args = parser.parse_args()

inputTableName  = args.input_table
outputTableName = args.output_table
databaseName    = args.database
limitsName      = args.limits

inputDataFrame = loadDataFrame()(databaseName, limitsName, inputTableName)
print("Statistics of load data frame:" )
print( inputDataFrame[all_columns].describe() )

inputDataFrame = limitDataUsingProcentiles(inputDataFrame)
print("Statistics of data frame limited using procentiles:")
print( inputDataFrame[all_columns].describe())

inliersDataFrame, outliersDataFrame = clearDataFromAnomalies(inputDataFrame)
print("Statistics of data frame after cleaning from anomalies:")
#print( inliersDataFrame [all_columns].describe() )
#print( outliersDataFrame[all_columns].describe() )

#outliersDataFrame[all_columns].hist( bins=200 )
#plt.savefig('Moscow_hist.png')
#plt.show()

#outliersDataFrame[all_columns].plot.scatter(x='longitude', y='latitude', c='DarkBlue')
#plt.savefig('Moscow_scatter.png')

outputDataFrame = smoothDataFrame( inliersDataFrame )

print("Result data frame:")
print( outputDataFrame.describe() )

saveDataFrame()( outputDataFrame, databaseName, outputTableName )



#plt.show()


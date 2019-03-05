#!/usr/bin/env python
import matplotlib.pyplot as plt
import pandas            as pd
import numpy             as np

from sqlalchemy import create_engine
from datetime   import datetime

import argparse

from commonModel import limitDataUsingLimitsFromFilename, loadDataFrame
from commonModel import FLOAT_COLUMNS, INT_COLUMNS, STR_COLUMNS, DATE_COLUMNS, TARGET_COLUMN

from sklearn.neighbors     import KDTree, BallTree
from sklearn.neighbors     import DistanceMetric
from sklearn.preprocessing import MinMaxScaler

parser = argparse.ArgumentParser()
#parser.add_argument("--database", type=str, default="mysql://sr:A4y8J6r4@149.154.71.73:3310/sr_dev" )
#parser.add_argument("--database", type=str, default="mysql://sr:password@portal.smartrealtor.pro:3306/smartrealtor" )
#parser.add_argument("--database", type=str, default="mysql://root:Intemp200784@127.0.0.1/smartRealtor" )
parser.add_argument("--database", type=str, default="mysql://root:Intemp200784@127.0.0.1/smartRealtor?unix_socket=/var/run/mysqld/mysqld.sock" )
parser.add_argument("--table"   , type=str, default="real_estate_from_ads_api" )

#parser.add_argument("--limits"  , type=str, default="input/NizhnyNovgorodLimits.json" )
parser.add_argument("--limits"  , type=str, default="input/MoscowLimits.json" )
parser.add_argument("--debug"   , type=bool,default=False)

args   = parser.parse_args()

databaseName = args.database
tableName    = args.table
limitsName   = args.limits
debugFlag    = args.debug

outputTable  = "moscow_processed_data"

inputDataFrame = None
inputDataFrame = loadDataFrame()( databaseName, tableName )

inputDataFrame = limitDataUsingLimitsFromFilename( inputDataFrame, limitsName )
inputDataFrame = inputDataFrame.select_dtypes(include=['number'])

features = ['longitude','latitude','exploitation_start_year']
inputDataFeatures = inputDataFrame[ features ].values
scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
inputDataFeatures = scaler.fit_transform(inputDataFeatures)
searchTree = KDTree( inputDataFeatures )

totalSize = inputDataFeatures.shape[0] 

totalMedianPricePerSquare_s = np.zeros( totalSize )
totalMeanPricePerSquare_s   = np.zeros( totalSize )
totalMedianPrice_s          = np.zeros( totalSize )
totalMeanPrice_s            = np.zeros( totalSize )

outputDataFrame = inputDataFrame.copy()

if debugFlag :
    for index in np.random.choice( inputDataFeatures.shape[0], 10 ):
        inputData = inputDataFeatures[index+0:index+1] 
        
        currentPrice                 = inputDataFrame['price'                  ].values[index]
        currentTotalSquare           = inputDataFrame['total_square'           ].values[index]
        currentKitchenSquare         = inputDataFrame['kitchen_square'         ].values[index]
        currentNumberOfRooms         = inputDataFrame['number_of_rooms'        ].values[index]
        currentExploitationStartYear = inputDataFrame['exploitation_start_year'].values[index]
        
        distance_s, index_s = searchTree.query( inputData, k=7 )
        #distance_s, index_s = searchTree.query( inputData, r=0.05, k=5 )
        
        Price_s                 = inputDataFrame['price'                  ].values[index_s] 
        totalSquare_s           = inputDataFrame['total_square'           ].values[index_s] 
        numberOfRooms_s         = inputDataFrame['number_of_rooms'        ].values[index_s] 
        exploitationStartYear_s = inputDataFrame['exploitation_start_year'].values[index_s] 
        
        print( "-> currentPrice                 {:9.2f}".format( currentPrice                ) )
        print( "-> currentTotalSquare           {:9.2f}".format( currentTotalSquare          ) )
        print( "-> currentExploitationStartYear {:9.2f}".format( currentExploitationStartYear) )
        
        mask = distance_s < 0.03;
        
        if np.sum( mask ) >= 3:
            totalPricePerSquare_s = Price_s / totalSquare_s
            
            totalMedianPricePerSquare = np.median( totalPricePerSquare_s )
            totalMeanPricePerSquare   = np.mean  ( totalPricePerSquare_s )
            
            totalMedianPricePerSquare_s[ index ] = totalMedianPricePerSquare 
            totalMeanPricePerSquare_s  [ index ] = totalMeanPricePerSquare  
            
            totalMedianPrice_s         [ index ] = totalMedianPricePerSquare*currentTotalSquare 
            totalMeanPrice_s           [ index ] = totalMeanPricePerSquare  *currentTotalSquare
            


else :
    for index in range( inputDataFeatures.shape[0] ):
        inputData = inputDataFeatures[index+0:index+1] 
        
        currentPrice                 = inputDataFrame['price'                  ].values[index]
        currentTotalSquare           = inputDataFrame['total_square'           ].values[index]
        currentKitchenSquare         = inputDataFrame['kitchen_square'         ].values[index]
        currentNumberOfRooms         = inputDataFrame['number_of_rooms'        ].values[index]
        currentExploitationStartYear = inputDataFrame['exploitation_start_year'].values[index]
        
        distance_s, index_s = searchTree.query( inputData, k=5 )
        #distance_s, index_s = searchTree.query( inputData, r=0.05, k=5 )
        
        Price_s                 = inputDataFrame['price'                  ].values[index_s] 
        totalSquare_s           = inputDataFrame['total_square'           ].values[index_s] 
        numberOfRooms_s         = inputDataFrame['number_of_rooms'        ].values[index_s] 
        exploitationStartYear_s = inputDataFrame['exploitation_start_year'].values[index_s] 
        
        mask = distance_s < 0.03;
        
        if np.sum( mask ) >= 3:
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
    
    engine = create_engine( databaseName )
    engine.execute("DROP TABLE IF EXISTS {}".format( outputTable ) )
    outputDataFrame.to_sql(outputTable, con=engine)
    
    fig, ax = plt.subplots()
    ax.set_title('medianPrice')
    ax.hist( totalMedianPricePerSquare_s, bins=20, color='r')
    plt.show()



#!/usr/bin/env python
import matplotlib.pyplot as plt
import pandas            as pd
import numpy             as np

from sqlalchemy import create_engine
from datetime   import datetime

from tqdm       import tqdm

import argparse

from commonModel import limitDataUsingLimitsFromFilename, loadDataFrame
from commonModel import FLOAT_COLUMNS, INT_COLUMNS, STR_COLUMNS, DATE_COLUMNS, TARGET_COLUMN

from sklearn.neighbors     import KDTree, BallTree
from sklearn.neighbors     import DistanceMetric
from sklearn.preprocessing import MinMaxScaler

parser = argparse.ArgumentParser()
parser.add_argument("--database", type=str, default="mysql://root:UWmnjxPdN5ywjEcN@188.120.245.195:3306/domprice_dev1_v2" )
parser.add_argument("--table"   , type=str, default="real_estate_from_ads_api" )
#parser.add_argument("--limits"  , type=str, default="input/NizhnyNovgorodLimits.json" )
parser.add_argument("--limits"  , type=str, default="input/KazanLimits.json" )
parser.add_argument("--verbose" , type=bool,default=False)

args   = parser.parse_args()

inputDatabase = args.database
inputTable    = args.table

limitsName    = args.limits
verboseFlag   = args.verbose

outputDatabase = "mysql://root:Intemp200784@127.0.0.1/smartRealtor?unix_socket=/var/run/mysqld/mysqld.sock" 
#outputTable    = "nizhny_novgorod_processed_data"
outputTable    = "kazan_processed_data"

inputDataFrame = None
inputDataFrame = loadDataFrame()( inputDatabase, inputTable, ['number','datetime'] )

print( inputDataFrame[DATE_COLUMNS].describe() )
inputDataFrame = limitDataUsingLimitsFromFilename( inputDataFrame, limitsName )

print( inputDataFrame.describe() )

#Drop duplicates
subset = ['price','total_square','number_of_rooms', 'floor_number', 'number_of_floors', 'longitude','latitude' ]
inputDataFrame.sort_values('created_at', inplace=True )
inputDataFrame.drop_duplicates(subset=subset, keep='last', inplace=True)

print( inputDataFrame.describe() )

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

outputDataFrame = inputDataFrame.copy()

print( "Before->" )
print( outputDataFrame[ FLOAT_COLUMNS+INT_COLUMNS ].describe() )

distance_limit = 0.005;

if verboseFlag :
    for index in np.random.choice( inputDataFeatures.shape[0], 10 ):
        inputData = inputDataFeatures[index+0:index+1] 
        
        currentPrice                 = inputDataFrame['price'                  ].values[index]
        currentTotalSquare           = inputDataFrame['total_square'           ].values[index]
        currentKitchenSquare         = inputDataFrame['kitchen_square'         ].values[index]
        currentNumberOfFloors        = inputDataFrame['number_of_floors'       ].values[index]
        currentNumberOfRooms         = inputDataFrame['number_of_rooms'        ].values[index]
        currentExploitationStartYear = inputDataFrame['exploitation_start_year'].values[index]
        
        index_s, distance_s = searchTree.query_radius( inputData, distance_limit, count_only=False, return_distance=True )
        index_s    = index_s   [0].astype(np.int32  )
        distance_s = distance_s[0].astype(np.float32)
        #distance_s, index_s = searchTree.query( inputData, k=5 )
        #distance_s, index_s = searchTree.query( inputData, r=0.05, k=5 )
        
        Price_s                 = inputDataFrame['price'                  ].values[index_s] 
        totalSquare_s           = inputDataFrame['total_square'           ].values[index_s] 
        numberOfRooms           = inputDataFrame['number_of_floors'       ].values[index_s]
        numberOfRooms_s         = inputDataFrame['number_of_rooms'        ].values[index_s] 
        exploitationStartYear_s = inputDataFrame['exploitation_start_year'].values[index_s] 
        
        print( "-> currentPrice                 {:9.2f}".format( currentPrice                ) )
        print( "-> currentTotalSquare           {:9.2f}".format( currentTotalSquare          ) )
        print( "-> currentExploitationStartYear {:9.2f}".format( currentExploitationStartYear) )
        
        mask = distance_s < distance_limit;
        
        if np.sum( mask ) >= 5:
            totalPricePerSquare_s = Price_s / totalSquare_s
            
            totalMedianPricePerSquare = np.median( totalPricePerSquare_s )
            totalMeanPricePerSquare   = np.mean  ( totalPricePerSquare_s )
            
            totalMedianPricePerSquare_s[ index ] = totalMedianPricePerSquare 
            totalMeanPricePerSquare_s  [ index ] = totalMeanPricePerSquare  
            
            totalMedianPrice_s         [ index ] = totalMedianPricePerSquare*currentTotalSquare 
            totalMeanPrice_s           [ index ] = totalMeanPricePerSquare  *currentTotalSquare
            
            print( "-> currentMedianPrice        {:9.2f}".format( totalMedianPricePerSquare*currentTotalSquare ) )
            print( "-> currentMeanPrice          {:9.2f}".format( totalMeanPricePerSquare  *currentTotalSquare ) )
        print("")
else :
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
        
        if np.sum( mask ) >= 5:
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
    
    engine = create_engine( outputDatabase )
    engine.execute("DROP TABLE IF EXISTS {}".format( outputTable ) )
    outputDataFrame.to_sql(outputTable, con=engine)
    print("After<-")
    print( outputDataFrame[ FLOAT_COLUMNS+INT_COLUMNS ].describe() )

    fig, ax = plt.subplots()
    ax.set_title('medianPrice')
    ax.hist( totalMedianPricePerSquare_s, bins=100, color='r')
    plt.show()



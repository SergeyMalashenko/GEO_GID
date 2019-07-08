#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import itertools
import datetime
import argparse
import types
import torch
import timeit
import json
import math
import sys

from sqlalchemy import create_engine

from commonModel import FLOAT_COLUMNS, INT_COLUMNS, STR_COLUMNS, TARGET_COLUMN
from commonModel import limitDataUsingLimitsFromFilename
from commonModel import limitDataUsingProcentiles

from commonModel import LinearNet

import datetime

def getClosestItemsInDatabase( engine, inputQuery, inputDataBase, inputTable, inputDeltas, limitPublicationDate=30 ):
    inputDeltasFields = set( inputDeltas.index ) 
    inputQueryFields  = set( inputQuery .index )
    
    processedDeltas = {field: inputDeltas[field] for field in inputDeltasFields.intersection(inputQueryFields)}
    
    processedLimits = dict()
    for (field, delta) in processedDeltas.items():
        processedLimits[field] = (
            inputQuery[field] - abs(delta),
            inputQuery[field] + abs(delta))
   
    sql_query = """SELECT * FROM {} WHERE """.format(inputTable)
    sql_query += """ AND """.join(
        "{1} <= {0} AND {0} <= {2}".format(
            field, min_value, max_value) for (
            field, (min_value, max_value)) in processedLimits.items())
    currentDate  = datetime.date.today()
    
    deltaDate    = datetime.timedelta(days=limitPublicationDate)
    updatedDate  = currentDate - deltaDate
    sql_query += """ AND """ + """publication_date BETWEEN '{}' AND '{}'  """.format( updatedDate, currentDate )
    
    resultValues = pd.read_sql_query(sql_query, engine)
    subset = ['longitude','latitude','total_square','number_of_rooms','number_of_floors','floor_number']
    resultValues.drop_duplicates(subset=subset, keep='first', inplace=True)
    
    return resultValues

def getTopKClosestItems( inputItem, closestItem_s, inputScales, inputTopK=5):
    if not closestItem_s.empty:
        inputScalesFields = inputScales.keys()
        
        processedInputItem     = inputItem    [ inputScalesFields ]
        processedClosestItem_s = closestItem_s[ inputScalesFields ]
        
        inputScales_numpy            = inputScales           .values
        processedInputItem_numpy     = processedInputItem    .values
        processedClosestItem_s_numpy = processedClosestItem_s.values
        processedInputItem_numpy     = processedInputItem_numpy.reshape(1, -1)
        
        processedResult_s_numpy = (processedClosestItem_s_numpy - processedInputItem_numpy)*inputScales_numpy
        processedResult_s_numpy = np.linalg.norm( processedResult_s_numpy, axis=1)

        index_s = processedResult_s_numpy.argsort()[:inputTopK]
        return closestItem_s.iloc[index_s]
    else:
        return closestItem_s

def getSimilarObjectsMain( engine, tableName, userQuery, userScales, userDeltas, outputTopK, outputFeatures ): 
    userQuery  = pd.Series( data=userQuery  )
    userScales = pd.Series( data=userScales )
    userDeltas = pd.Series( data=userDeltas )
    
    closestItem_s = getClosestItemsInDatabase( engine, userQuery, databaseName , tableName , userDeltas)
    closestItem_s = getTopKClosestItems      (         userQuery, closestItem_s, userScales, outputTopK)
    
    closestItem_s = closestItem_s[ outputFeatures ]
    
    return closestItem_s  

def parseArguments() :
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--query"   , type=str)
    parser.add_argument("--database", type=str)
    parser.add_argument("--table"   , type=str)
    parser.add_argument("--scales"  , type=str)
    parser.add_argument("--deltas"  , type=str)
    
    parser.add_argument("--output_features", type=str, default='id,longitude,latitude,total_square,number_of_floors,number_of_rooms,exploitation_start_year,publication_date')
    parser.add_argument("--output_format"  , type=str, default='json')
    parser.add_argument("--output_topk"    , type=int, default=5)
    parser.add_argument("--verbose", action="store_true")
    
    args = parser.parse_args()

    userQuery  = eval("dict({})".format(args.query ))
    userScales = eval("dict({})".format(args.scales))
    userDeltas = eval("dict({})".format(args.deltas))
    
    databaseName = args.database
    tableName    = args.table
    
    outputFeatures = args.output_features.split(",")
    outputFormat   = args.output_format
    outputTopK     = args.output_topk
    
    verboseFlag = args.verbose
    
    return databaseName, tableName, userQuery, userScales, userDeltas, outputFeatures, outputFormat, outputTopK

if __name__ == "__main__":
    databaseName, tableName, userQuery, userScales, userDeltas, outputFeatures, outputFormat, outputTopK = parseArguments()
    
    engine        = create_engine( databaseName )
    closestItem_s = getSimilarObjectsMain( engine, tableName, userQuery, userScales, userDeltas, outputTopK, outputFeatures )

    if outputFormat == 'json' :
        def json_serial(obj):
            if isinstance(obj, (datetime.datetime,datetime.date )):
                return obj.isoformat()
            raise TypeError ("Type %s not serializable" % type(obj))
        json_output = closestItem_s.to_dict( orient='records' )
        print( json.dumps( json_output, default=json_serial, sort_keys=True, indent=4, separators=(',', ': ')) )
    else :
        print( closestItem_s )


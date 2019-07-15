#!/usr/bin/env python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import pandas            as pd
import numpy             as np
from sklearn.preprocessing      import RobustScaler

from sqlalchemy import create_engine
from datetime   import datetime

import argparse
import json
import time


from commonModel import limitDataUsingLimitsFromFilename, loadDataFrame
from commonModel import FLOAT_COLUMNS, INT_COLUMNS, STR_COLUMNS, DATE_COLUMNS, TARGET_COLUMN

from collections import Counter

parser = argparse.ArgumentParser()
#parser.add_argument("--database", type=str, default="mysql://sr:A4y8J6r4@149.154.71.73:3310/sr_dev" )
#parser.add_argument("--database", type=str, default="mysql://sr:password@portal.smartrealtor.pro:3306/smartrealtor" )
#parser.add_argument("--database", type=str, default="mysql://root:Intemp200784@127.0.0.1/smartRealtor" )
#parser.add_argument("--database", type=str, default="mysql://root:Intemp200784@127.0.0.1/smartRealtor?unix_socket=/var/run/mysqld/mysqld.sock" )
parser.add_argument("--database", type=str, default="mysql://root:UWmnjxPdN5ywjEcN@188.120.245.195:3306/domprice_dev1_v2" )
parser.add_argument("--table"   , type=str, default="real_estate_from_ads_api" )
#parser.add_argument("--table"   , type=str, default="src_ads_raw" )
parser.add_argument("--limits"  , type=str, default="input/MoscowLimits.json" )
#parser.add_argument("--limits"  , type=str, default="input/KazanLimits.json" )
#parser.add_argument("--limits"  , type=str, default="input/NizhnyNovgorodLimits.json" )
#parser.add_argument("--limits"  , type=str, default="input/SaintPetersburgLimits.json" )
args   = parser.parse_args()

databaseName = args.database
tableName    = args.table
limitsName   = args.limits


engine = create_engine( databaseName, encoding='utf8' )
#engine = create_engine( databaseName, encoding='iso-8859-1' )
table_s = pd.read_sql("SHOW TABLES;", engine )
print("TABLE NAMES")
print( table_s )

column_s = pd.read_sql( 'DESCRIBE {}'.format( tableName ), engine )
print(column_s)

start_time = time.time()
with open(limitsName, "r") as read_file:
	processedLimits = json.load(read_file)

sql_query = """SELECT * FROM {} WHERE """.format(tableName)

sql_query += """ AND """.join(
        "{1} <= {0} AND {0} <= {2}".format(
            field, processedLimits[field]['min'], processedLimits[field]['max']) for field in processedLimits.keys())

resultValues = pd.read_sql_query(sql_query, engine)
subset = ['longitude', 'latitude', 'total_square', 'number_of_rooms', 'number_of_floors', 'floor_number']
resultValues.drop_duplicates(subset=subset, keep='first', inplace=True)


print("--- %s seconds ---" % (time.time() - start_time))

mask = True
resultValues['pricePerSquare'] = resultValues['price'] / resultValues['total_square']

pricePerSquareValues = resultValues['pricePerSquare'].values

robustScaler = RobustScaler(quantile_range=(10, 90))
robustScaler.fit(pricePerSquareValues.reshape((-1, 1)))
pricePerSquareValues = robustScaler.transform(pricePerSquareValues.reshape((-1, 1))).reshape(-1)

mask = (pricePerSquareValues > -1) & (pricePerSquareValues < 1) & mask

resultValues = resultValues[mask]

statistics_price = resultValues['pricePerSquare'].describe()
print(statistics_price)

resultValues.hist(column='pricePerSquare',bins=100)

plt.show()


#!/usr/bin/env python3
import matplotlib.pyplot as plt
import pandas            as pd
import numpy             as np

from sqlalchemy import create_engine
from datetime   import datetime

import argparse

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

#parser.add_argument("--limits"  , type=str, default="input/MoscowLimits.json" )
parser.add_argument("--limits"  , type=str, default="input/NizhnyNovgorodLimits.json" )
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
print( column_s )

#dataFrame = pd.read_sql('SELECT * FROM smartRealtor.real_estate_from_ads_api;', engine )
dataFrame = pd.read_sql_table( tableName, engine, schema='domprice_dev1_v2')
#dataFrame = pd.read_sql('SELECT price,longitude,latitude,total_square,kitchen_square,living_square,number_of_rooms,floor_number,number_of_floors,exploitation_start_year FROM smartRealtor.real_estate_from_ads_api;', engine )

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
	print( dataFrame.describe() )

def check_float( x ):
	try:
		float(x)
	except ValueError:
		return False
	return True
def check_int( x ):
	try:
		int(x)
	except ValueError:
		return False
	return True
def check_str( x ):
	return isinstance(x, str)
def check_datetime( x ):
	return isinstance(x, datetime)

FLOAT_COLUMNS = FLOAT_COLUMNS + ['distance_to_metro',]

def check_row( row ):
	check_s = True
	for columnName in FLOAT_COLUMNS :
		if columnName in row : check_s = check_s and check_float( row[ columnName ] )
	for columnName in INT_COLUMNS:
		if columnName in row : check_s = check_s and check_int  ( row[ columnName ] )
	for columnName in STR_COLUMNS:
		if columnName in row : check_s = check_s and check_str  ( row[ columnName ] )
	for columnName in DATE_COLUMNS:
		if columnName in row : check_s = check_s and check_datetime( row[ columnName ] )
	return check_s
#Check types
dataFrame = dataFrame[ dataFrame.apply( check_row , axis=1 ) ]
#Convert values
for columnName in FLOAT_COLUMNS:
	if columnName in dataFrame.columns : dataFrame[ columnName ] = dataFrame[ columnName ].astype( np.float32 )
for columnName in INT_COLUMNS:
	if columnName in dataFrame.columns : dataFrame[ columnName ] = dataFrame[ columnName ].astype( np.int32   )
for columnName in DATE_COLUMNS:
	if columnName in dataFrame.columns : dataFrame[ columnName ] = pd.to_datetime( dataFrame[ columnName ] )
#Limit values
dataFrame = limitDataUsingLimitsFromFilename( dataFrame, limitsName )
print( "\n{}\n".format( limitsName) )
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
	print( dataFrame.describe() )

#Drop duplicates
subset=['price', 'total_square', 'number_of_rooms','longitude','latitude' ]	
dataFrame.drop_duplicates(subset=subset, keep='first', inplace=True)	
print("Data without duplicates")
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
	print( dataFrame.describe() )

print( np.min( dataFrame['created_at'].values ), np.max( dataFrame['created_at'].values )) 

fig, ax = plt.subplots()
ax.set_title('CreatedAt')
ax.hist(dataFrame['created_at'].values,40,color='r')
plt.show()

bins = range(5,105)
bins = [i * 0.5e4 for i in bins ]

fig, ax = plt.subplots()
pricePerSquare  = ( dataFrame['price'] / dataFrame['total_square'] ).values
ax.set_title('Price/Square')
ax.hist( pricePerSquare, bins=bins, color='r')
plt.show()

bins = range(0,10)
bins = [i * 1000 for i in bins ]

fig, ax = plt.subplots()
distanceToMetro  = ( dataFrame['distance_to_metro'] ).values
ax.set_title('distanceToMetro')
ax.hist( distanceToMetro, bins=bins, color='r')
plt.show()

fig, ax = plt.subplots()
distanceToMetro  = ( dataFrame['distance_to_metro'] ).values
ax.set_title('distanceToMetro')
ax.hist( distanceToMetro, bins=bins, color='r')
plt.show()

fig, ax = plt.subplots()
longitude_s = ( dataFrame['longitude'] ).values
latitude_s  = ( dataFrame['latitude' ] ).values
ax.set_title('longitude/latitude')
ax.scatter(longitude_s, latitude_s)
plt.show()

print( Counter(dataFrame['wall_material'].values ))

#to_timestamp = np.vectorize(lambda x: x.timestamp())
#time_stamp_s = to_timestamp( dataFrame['created_at'].values )
#print( np.histogram(time_stamp_s) )

#name_s = [ "analytics", "appraisers", "customer", "estimator", "estimator_document", "estimator_document_scan", "evaluation_order", "evaluation_report", "express_estimation", "house", "house_group", "metro", "migration_versions", "nn", "nn_neiro", "real_estate", "real_estate_analog", "real_estate_analog_foto", "real_estate_document", "real_estate_document_scan", "real_estate_from_ads_api", "real_estate_owner", "real_estate_photo", "report_form", "user"]
#name_s = [ "real_estate_from_ads_api"]
#for name in name_s :
#	data = pd.read_sql( "SELECT * from INFORMATION_SCHEMA.COLUMNS where TABLE_NAME='{}'".format(name), engine )
#	print('NAME: ', name)
#	print( data )

#sql_query = """SELECT * FROM nn_neiro WHERE 56.28826 <= latitude AND latitude <= 56.290259999999996 AND 44.07104 <= longitude AND longitude <= 44.07304"""
#sql_query = """SELECT * FROM nn_neiro WHERE SQRT(latitude*latitude + longitude*longitude) < 50"""
#
#resultValues = pd.read_sql_query( sql_query, engine)
#
#print( resultValues )
#to_timestamp = np.vectorize(lambda x: x.timestamp() )
#time_stamps = to_timestamp( dataFrame['re_created_at'].values )
#np.histogram(time_stamps)
#dataFrame = loadDataFrame()( databaseName, tableName )
#dataFrame = limitDataUsingLimitsFromFilename( dataFrame, limitsFileName )

#print( dataFrame.describe() )
#print( dataFrame            )

#print( np.histogram( dataFrame['re_created_at'].values ) )
#print( np.min( dataFrame['re_created_at'].values ), np.max( dataFrame['re_created_at'].values ) )

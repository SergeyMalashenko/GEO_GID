#!/usr/bin/env python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn.preprocessing     import StandardScaler
from sklearn.neighbors import NearestNeighbors
from os.path import join, dirname
from dotenv import load_dotenv
import os

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
import json

from sqlalchemy  import create_engine
from commonModel import FLOAT_COLUMNS, INT_COLUMNS, STR_COLUMNS, TARGET_COLUMN
from commonModel import limitDataUsingLimitsFromFilename
from commonModel import limitDataUsingProcentiles

pd.set_option('display.width'      , 500 )
pd.set_option('display.max_rows'   , 500 )
pd.set_option('display.max_columns', 500 )
class loadDataFrame(object):
    def __call__(self, databasePath, limitsFileName, tableName, date_update, date_download):
        engine = create_engine(databasePath)
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
            sql_query += """ AND living_square + kitchen_square <= total_square"""
        sql_query += """ AND ( date_download BETWEEN '{1}' AND '{0}' OR date_update BETWEEN '{2}' AND '{0}' ) """.format( datetime.date.today(), date_download, date_update )
        resultValues = pd.read_sql_query(sql_query, engine)
        if 'date_download' in resultValues.columns:
            resultValues = resultValues.sort_values(by=['date_download'])

        subset = ['longitude', 'latitude', 'total_square', 'number_of_rooms', 'number_of_floors', 'floor_number']
        resultValues.drop_duplicates(subset=subset, keep='last', inplace=True)
        """
        f_cheap_houses = open("cheap_houses.txt","w")
        f_cheap_houses.write(str(cheapHouses[['exploitation_start_year', 'price','source_url','publication_date','pricePerSquare','longitude','latitude']]))
        f_cheap_houses.flush()
        f_cheap_houses.close()
        """
        #resultValues = limitDataUsingProcentiles(resultValues)
        #otherHouses = resultValues.loc[resultValues['building_type'] == 'Другое']
        #f_other_houses = open("other_houses.txt", "w")

        #f_other_houses.write(str(otherHouses[['exploitation_start_year', 'price', 'source_url', 'publication_date',
                                               #'longitude', 'latitude']]))
        #f_other_houses.flush()
        #f_other_houses.close()
        return resultValues
parser = argparse.ArgumentParser()
parser.add_argument("--query", type=str, default="longitude=43.985530, latitude=56.269815, total_square=63, living_square=42, kitchen_square=7, number_of_rooms=3, floor_number=3, number_of_floors=5, exploitation_start_year=1969" )
parser.add_argument("--city", type=str, default='Nizhny Novgorod')
parser.add_argument("--database", type=str, default="")
parser.add_argument("--features", type=str, default="")
parser.add_argument("--top_k", type=str, default="")
parser.add_argument("--date_update", type=str, default="")
parser.add_argument("--date_download", type=str, default="")
args = parser.parse_args()
# Create .env file path.
dotenv_path = join(dirname(__file__), '.env')

# Load file from the path.
load_dotenv(dotenv_path)
# Accessing variables.
inputDatabase = args.database
featureNames = args.features
top_k = args.top_k
date_update = args.date_update
date_download = args.date_download

if inputDatabase == "":
    inputDatabase = os.getenv('DATABASE_URL',"")
    if inputDatabase == "":
        print("Error: database is not defined")
if featureNames == "":
    featureNames = os.getenv('FEATURES',"")
    if featureNames == "":
        print("Error: features are not defined")
featureNames = featureNames.split(',')
if top_k == "":
    top_k = os.getenv('TOP_K',"")
    if top_k == "":
        print("Error: top_k is not defined")

date_update = os.getenv('DATE_UPDATE',"")
date_download = os.getenv('DATE_DOWNLOAD',"")
#default settings for date:
if date_update == '':
    date_update = datetime.date.today() - datetime.timedelta(days=30)
else:
    date_update = datetime.datetime.strptime(date_update, '%Y-%m-%d')
if date_download == '':
    date_download = datetime.date.today() - datetime.timedelta(days=30)
else:
    date_download = datetime.datetime.strptime(date_download, '%Y-%m-%d')
city = args.city
inputTableDict = {'Nizhny Novgorod': 'real_estate_from_ads_api_52','Kazan': 'real_estate_from_ads_api_16',
                  'Saint Petersburg': 'real_estate_from_ads_api_78','Moscow': 'real_estate_from_ads_api_77'}
limitsDict = {'Nizhny Novgorod': 'input/NizhnyNovgorodLimits.json','Kazan': 'input/KazanLimits.json',
                  'Saint Petersburg': 'input/SaintPetersburgLimits.json','Moscow': 'input/MoscowLimits.json'}
modelsDict = {'Nizhny Novgorod': 'analogs_model/analogsNizhnyNovgorod.pkl','Kazan': 'analogs_model/analogsKazan.pkl',
                  'Saint Petersburg': 'analogs_model/analogsSaintPetersburg.pkl','Moscow': 'analogs_model/analogsMoscow.pkl'}
inputTable = inputTableDict[city]
limitsFileName   = limitsDict[city]
modelFileName  = modelsDict[city]


inputQuery = args.query

inputDataFrame = None
inputDataFrame = loadDataFrame()(inputDatabase, limitsFileName, inputTable, date_update, date_download)

inputDataFrame = limitDataUsingProcentiles(inputDataFrame)
print(inputDataFrame.describe())
#inputDataFrame = pd.read_csv("NizhnyNovgorod_dataframe.csv")
#print(inputDataFrame.describe())

if inputQuery == "":
    inputDataFrame = inputDataFrame[featureNames]
    DataFrame_values = inputDataFrame.values
    FEATURE_DEFAULTS = ((inputDataFrame.max() + inputDataFrame.min()) * 0.5).to_dict()
    preprocessor = StandardScaler()
    preprocessor.fit(DataFrame_values)
    DataFrame_values = preprocessor.transform(DataFrame_values)
    neigh = NearestNeighbors(algorithm='kd_tree',n_jobs=-1)
    neigh.fit(DataFrame_values)

    modelPacket = dict()
    modelPacket['model'] = neigh
    modelPacket['preprocessor'] = preprocessor
    modelPacket['feature_names'] = featureNames
    modelPacket['feature_defaults'] = FEATURE_DEFAULTS
    if modelFileName != "":
        with open(modelFileName, 'wb') as fid:
            cPickle.dump(modelPacket, fid)
else:
    with open(modelFileName, 'rb') as fid:
        modelPacket = cPickle.load(fid)

        NEIGH_MODEL = modelPacket['model']
        PREPROCESSOR = modelPacket['preprocessor']
        MODEL_FEATURE_NAMES = modelPacket['feature_names']
        MODEL_FEATURE_DEFAULTS = modelPacket['feature_defaults']

    # Process query
    userQuery = eval("dict({})".format(inputQuery))
    defaultQuery = MODEL_FEATURE_DEFAULTS
    inputQuery = defaultQuery
    inputQuery.update(userQuery)

    DataFrame = pd.DataFrame(data=inputQuery, index=[0])

    DataFrame = limitDataUsingLimitsFromFilename(DataFrame, limitsFileName)
    inputDataSize = len(DataFrame.index)
    if inputDataSize > 0:  # Check that input data is correct
        for i in range(inputDataSize):
            inputItem = DataFrame.iloc[i]
            inputItemForModel = DataFrame[MODEL_FEATURE_NAMES].iloc[i]
            inputItemForModel_values = PREPROCESSOR.transform(inputItemForModel.values.reshape(1, -1))

            inputItemNeighbors_indexes = NEIGH_MODEL.radius_neighbors(inputItemForModel_values,radius=0.05)
            inputItemNeighbors_indexes = np.array(inputItemNeighbors_indexes)

            index_s = inputItemNeighbors_indexes[0][0].argsort()
            print(inputItemNeighbors_indexes[0][0][index_s])

            #inputItemNeighbors = inputDataFrame.iloc[inputItemNeighbors_indexes.flatten()]

            inputItemNeighbors = inputDataFrame.iloc[inputItemNeighbors_indexes[1][0][index_s][:int(top_k)]]
            print('inputItem: {0}\n'.format(str(inputItemForModel)))
            print('Neighbors:\n {0}\n'.format(
                    str(inputItemNeighbors[MODEL_FEATURE_NAMES + ['price']])))

            print("Median    price: {:,}".format(inputItemNeighbors['price'].median()))
            print("Mean      price: {:,}".format(inputItemNeighbors['price'].mean()))
            print("Max       price: {:,}".format(inputItemNeighbors['price'].max()))
            print("Min       price: {:,}".format(inputItemNeighbors['price'].min()))

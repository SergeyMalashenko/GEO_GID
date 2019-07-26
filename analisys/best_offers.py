#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import pandas            as pd
import numpy             as np

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing     import StandardScaler
from sklearn.preprocessing      import RobustScaler
from sqlalchemy import create_engine
from datetime   import datetime
import argparse
import json

import warnings

from pylab import rcParams





import datetime
color = ['orangered','dodgerblue','forestgreen','gold','c','crimson','chocolate','tomato','coral']
warnings.filterwarnings('ignore')

pd.options.display.max_columns = 52
pd.options.display.max_colwidth = 256

def limitDataUsingProcentiles(dataFrame):
    if 'price' in dataFrame.columns:
        mask = True

        pricePerSquare = (dataFrame['price'] / dataFrame['total_square'])
        pricePerSquareValues = pricePerSquare.values

        robustScaler = RobustScaler(quantile_range=(5, 95))
        robustScaler.fit(pricePerSquareValues.reshape((-1, 1)))
        pricePerSquareValues = robustScaler.transform(pricePerSquareValues.reshape((-1, 1))).reshape(-1)

        mask = (pricePerSquareValues > -1) & (pricePerSquareValues < 1) & mask

        dataFrame = dataFrame[mask]

    return dataFrame


def theBestoffer(start_date, end_date,inputDataFrame):
    month = datetime.timedelta(days=7)
    mask = (inputDataFrame['publication_date'] > end_date - month) & (inputDataFrame['publication_date'] <= end_date) & (inputDataFrame['exploitation_start_year'] > 1980)
    inputDataFrame = inputDataFrame.loc[mask]
    inputDataFrame['pricePerSquare'] = (inputDataFrame['price'] / inputDataFrame['total_square'])
    all_columns_new = all_columns
    all_columns_new.remove('price')
    inputDataFrame = inputDataFrame.loc[inputDataFrame['city_district'].values != '']
    Districts = np.unique(inputDataFrame['city_district'].values)
    print(Districts)
    f_bestOffer = open("best_offer.txt", "w")
    for cityDistrict in Districts:
        f_bestOffer.write('\n\n{}:\n'.format(cityDistrict))
        cityDataFrame = inputDataFrame.loc[inputDataFrame['city_district'] == cityDistrict]
        print('\n\n{}:\n'.format(cityDistrict))
        cityDataFrame.index = np.arange(len(cityDataFrame))
        scaler = StandardScaler()
        cityDataFrame_numpy = cityDataFrame[all_columns_new].to_numpy()
        scaler.fit(cityDataFrame_numpy)
        cityDataFrame_numpy_scaled = scaler.transform(cityDataFrame_numpy)
        neigh = NearestNeighbors(n_neighbors=10, radius=0.7, algorithm='kd_tree', n_jobs=-1)
        neigh.fit(cityDataFrame_numpy_scaled)

        for index in cityDataFrame.index:
            #sampleDataFrame = scaler.transform(np.reshape(cityDataFrame.loc[index][all_columns_new],(-1,1)))
            sampleDataFrame = scaler.transform(cityDataFrame.loc[index][all_columns_new].to_numpy().reshape(1, -1))

            rng = neigh.radius_neighbors(sampleDataFrame, return_distance=False)[0]
            neighbors = np.asarray(rng)
            if len(neighbors)<1:
                continue
            sampleNeighbors = cityDataFrame.loc[neighbors]
            cityDataFrame.loc[index, 'priceDeviation']=(cityDataFrame.loc[index, 'pricePerSquare']-sampleNeighbors['pricePerSquare'].median())/sampleNeighbors['pricePerSquare'].median()
        cityDataFrame=cityDataFrame.loc[cityDataFrame['priceDeviation'].idxmin()]
        f_bestOffer.write(str(cityDataFrame[['exploitation_start_year', 'price','source_url','publication_date','priceDeviation','longitude','latitude']]))
        f_bestOffer.write(' Median price of analogs: {}'.format(str(sampleNeighbors['price'].median())))
        f_bestOffer.flush()
    f_bestOffer.close()


class loadDataFrame(object):
    def __call__(self, databasePath, limitsFileName, tableName):
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
            sql_query += """ AND living_square + kitchen_square <= total_square""".format(tableName)

        resultValues = pd.read_sql_query(sql_query, engine)

        if 'publication_date' in resultValues.columns:
            resultValues = resultValues.sort_values(by=['publication_date'])

        subset = ['longitude', 'latitude', 'total_square', 'number_of_rooms', 'number_of_floors', 'floor_number']
        resultValues.drop_duplicates(subset=subset, keep='last', inplace=True)
        """
        f_cheap_houses = open("cheap_houses.txt","w")
        f_cheap_houses.write(str(cheapHouses[['exploitation_start_year', 'price','source_url','publication_date','pricePerSquare','longitude','latitude']]))
        f_cheap_houses.flush()
        f_cheap_houses.close()
        """
        resultValues = limitDataUsingProcentiles(resultValues)

        return resultValues
parser = argparse.ArgumentParser()

parser.add_argument("--database", type=str, default="mysql://root:password@188.120.245.195:3306/domprice_dev1_v2?charset=utf8" )
#parser.add_argument("--input_table"   , type=str, default="real_estate_from_ads_api" )
parser.add_argument("--input_table"   , type=str, default="src_ads_raw_52" )
#parser.add_argument("--limits"  , type=str, default="input/MoscowLimits.json" )
#parser.add_argument("--output_table"   , type=str, default="real_estate_from_ads_api_processed" )
#parser.add_argument("--limits"  , type=str, default="input/KazanLimits.json" )
parser.add_argument("--limits"  , type=str, default="input/NizhnyNovgorodLimits.json" )

parser.add_argument("--start_date"  , type=str, default='2018-07-01' )
parser.add_argument("--end_date"  , type=str, default='2019-07-18')
#parser.add_argument("--limits"  , type=str, default="input/SaintPetersburgLimits.json" )
args = parser.parse_args()

input_tableName = args.input_table
databaseName = args.database
limitsName   = args.limits
start_date = datetime.datetime.strptime(args.start_date,'%Y-%m-%d')
end_date = datetime.datetime.strptime(args.end_date,'%Y-%m-%d')

inputDataFrame = None
inputDataFrame = loadDataFrame()(databaseName, limitsName, input_tableName )


theBestoffer(start_date,end_date,inputDataFrame)
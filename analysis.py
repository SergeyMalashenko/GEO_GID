#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import matplotlib.pyplot as plt
import pandas            as pd
import numpy             as np


from sqlalchemy import create_engine
from datetime   import datetime

import argparse
import json
import time
from pylab import rcParams
rcParams['figure.figsize'] = 15, 10
import matplotlib.dates as mdates
from commonModel import limitDataUsingProcentiles
from commonModel import FLOAT_COLUMNS, INT_COLUMNS, STR_COLUMNS, DATE_COLUMNS, TARGET_COLUMN

from collections import Counter

import datetime
color = ['orangered','dodgerblue','forestgreen','gold','c','crimson','chocolate','tomato','coral']

all_columns = FLOAT_COLUMNS+INT_COLUMNS
pd.options.display.max_columns = 52

def cityDistrictDiagram(start_date,end_date,inputDataFrame):
    mask = (inputDataFrame['publication_date'] > start_date) & (inputDataFrame['publication_date']  <= end_date)
    inputDataFrame = inputDataFrame.loc[mask]
    inputDataFrame = inputDataFrame.groupby(['city_district']).size().reset_index(name='count_of_city_district')
    print(inputDataFrame[['city_district','count_of_city_district']])
    df = pd.DataFrame({'Территориальная структура рынка готового жилья': inputDataFrame['count_of_city_district'].values }, index = inputDataFrame['city_district'])
    df = df[df.index != '']
    df.plot.pie(y='Территориальная структура рынка готового жилья', figsize=(5, 5),autopct='%1.1f%%')
    plt.savefig('Территориальная структура рынка готового жилья.png')


def numberOfRoomsDiagram(start_date,end_date,inputDataFrame):
    mask = (inputDataFrame['publication_date'] > start_date) & (inputDataFrame['publication_date'] <= end_date)
    inputDataFrame = inputDataFrame.loc[mask]
    inputDataFrame = inputDataFrame.groupby(['number_of_rooms']).size().reset_index(name='count_of_number_of_rooms')
    print(inputDataFrame[['number_of_rooms', 'count_of_number_of_rooms']])
    df = pd.DataFrame({'Структура рынка готового жилья по количеству комнат': [inputDataFrame.iloc[0]['count_of_number_of_rooms'],inputDataFrame.iloc[1]['count_of_number_of_rooms'],inputDataFrame.iloc[2]['count_of_number_of_rooms'],inputDataFrame.iloc[3]['count_of_number_of_rooms'], inputDataFrame.loc[inputDataFrame['number_of_rooms'] > 4, 'count_of_number_of_rooms'].sum()]}, index=['1-комн.','2-комн.','3-комн.','4-комн.','>4-комн.'])
    df.plot.pie(y='Структура рынка готового жилья по количеству комнат', figsize=(5, 5),autopct='%1.1f%%')
    plt.savefig('Структура рынка готового жилья по количеству комнат.png')


def buildingTypeDiagram(start_date,end_date,inputDataFrame):
    mask = (inputDataFrame['publication_date'] > start_date) & (inputDataFrame['publication_date'] <= end_date)
    inputDataFrame = inputDataFrame.loc[mask]
    inputDataFrame = inputDataFrame.groupby(['building_type']).size().reset_index(name='count_of_building_type')
    print(inputDataFrame[['building_type', 'count_of_building_type']])
    df = pd.DataFrame({'Структура рынка по типам готового жилья': inputDataFrame['count_of_building_type'].values}, index=inputDataFrame['building_type'])
    df = df[df.index != '']
    df.plot.pie(y='Структура рынка по типам готового жилья', figsize=(5, 5),autopct='%1.1f%%')
    plt.savefig('Структура рынка по типам готового жилья.png')


def dynamicsOfMeanPrice(start_date,end_date,inputDataFrame):
    mask = (inputDataFrame['publication_date'] > start_date) & (inputDataFrame['publication_date'] <= end_date)
    inputDataFrame = inputDataFrame.loc[mask]
    inputDataFrame['pricePerSquare'] = (inputDataFrame['price'] / inputDataFrame['total_square'])
    inputDataFrame = inputDataFrame.groupby(['publication_date']).mean()
    inputDataFrame = inputDataFrame.groupby(pd.Grouper(freq='MS')).mean()
    print(inputDataFrame['pricePerSquare'])
    fig, ax = plt.subplots()
    bars =  ax.bar(x=inputDataFrame.index, height=inputDataFrame['pricePerSquare'],width=25,color=['cadetblue','skyblue'],align='center')
    for p in ax.patches:
        ax.annotate(int(np.round(p.get_height(), decimals=0)), (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 10), fontweight='bold', fontsize=8, textcoords='offset points')
    deviation_of_price = (inputDataFrame.iloc[len(inputDataFrame)-1]['pricePerSquare']-inputDataFrame.iloc[0]['pricePerSquare'])/inputDataFrame.iloc[0]['pricePerSquare']
    plt.text(x=0.95,y=0.975,s="{:.1%}".format(deviation_of_price),fontsize=12,fontweight='bold', bbox=dict(facecolor='red', alpha=0.5),transform=ax.transAxes)
    plt.ylabel('руб./кв.м.',fontweight='bold')
    months = mdates.MonthLocator()
    months_fmt = mdates.DateFormatter('%m-%Y')
    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(months_fmt)
    ax.xaxis.set_minor_locator(months)
    ax.xaxis.set_major_formatter(months_fmt)
    plt.title('Изменение средней цены предложения на рынке готового жилья',fontweight='bold')
    plt.savefig('Изменение средней цены предложения на рынке готового жилья.png')


def buildingTypeAndMeanPrice(start_date,end_date,inputDataFrame):
    mask = (inputDataFrame['publication_date'] > start_date) & (inputDataFrame['publication_date'] <= end_date)
    inputDataFrame = inputDataFrame.loc[mask]
    inputDataFrame['pricePerSquare'] = (inputDataFrame['price'] / inputDataFrame['total_square'])
    inputDataFrame = inputDataFrame.groupby(['building_type']).mean()
    print(inputDataFrame['pricePerSquare'])
    inputDataFrame = inputDataFrame[inputDataFrame.index != '']
    fig, ax = plt.subplots()
    bars = ax.bar(x=inputDataFrame.index, height=inputDataFrame['pricePerSquare'],color=color)
    for p in ax.patches:
        ax.annotate(int(np.round(p.get_height(), decimals=0)), (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 10), fontweight='bold', fontsize=8, textcoords='offset points')
    plt.ylabel('руб./кв.м.',fontweight='bold')
    plt.savefig('Cредняя цена предложения по типам готового жилья Нижнего Новгорода.png')


def buildingTypeAndMeanPriceCityDistricts(start_date,end_date,inputDataFrame):
    mask = (inputDataFrame['publication_date'] > start_date) & (inputDataFrame['publication_date'] <= end_date)
    inputDataFrame = inputDataFrame.loc[mask]
    inputDataFrame['pricePerSquare'] = (inputDataFrame['price'] / inputDataFrame['total_square'])
    inputDataFrame = inputDataFrame.groupby(['city_district','building_type','number_of_rooms']).mean()
    print(inputDataFrame['pricePerSquare'])
    inputDataFrame = inputDataFrame.drop('', level='city_district')
    inputDataFrame = inputDataFrame.drop('', level='building_type')
    inputDataFrame = inputDataFrame.reset_index(level='number_of_rooms')
    Districts = inputDataFrame.index.get_level_values('city_district').drop_duplicates()

    for cityDistrict in Districts:
        cityDataFrame = inputDataFrame.loc[(cityDistrict)]

        buildingTypes = cityDataFrame.index.drop_duplicates()

        zero_data = np.zeros(shape=(len(buildingTypes), 4))
        df = pd.DataFrame(zero_data, columns=['1-комн.', '2-комн.', '3-комн.', '4-комн. и больше'],index=buildingTypes)

        for buildingType in buildingTypes:

            for i in range(1, 4):
                if buildingType in cityDataFrame.loc[cityDataFrame['number_of_rooms'] == i].index:
                    smallDataFrame = cityDataFrame.loc[(buildingType)]
                    df.loc[buildingType, '{0}-комн.'.format(i)] = smallDataFrame.loc[smallDataFrame['number_of_rooms'] == i]['pricePerSquare'].to_numpy().astype(float)

            if buildingType in cityDataFrame.loc[cityDataFrame['number_of_rooms'] >= 4].index:
                smallDataFrame = cityDataFrame.loc[(buildingType)]
                df.loc[buildingType, '4-комн. и больше'] = smallDataFrame.loc[smallDataFrame['number_of_rooms'] >= 4]['pricePerSquare'].mean()

        print(df)
        ax = df.plot.bar(width=1,color=color)
        for p in ax.patches:
            ax.annotate(int(np.round(p.get_height(),decimals=0)), (p.get_x()+p.get_width()/2., p.get_height()), ha='center', va='center', xytext=(0, 10),fontweight='bold',fontsize=8, textcoords='offset points')
        plt.ylabel('руб./кв.м.', fontweight='bold')
        plt.legend()
        plt.title('{} район средняя цена предложения по типам квартир и числу комнат'.format(cityDistrict), fontweight='bold')
        plt.savefig('{} район средняя цена предложения по типам квартир и числу комнат.png'.format(cityDistrict))

def theBestoffer(end_date,inputDataFrame):
    week = datetime.timedelta(days=7)
    mask = (inputDataFrame['publication_date'] > (end_date - week)) & (inputDataFrame['publication_date'] <= end_date)
    inputDataFrame = inputDataFrame.loc[mask]
    inputDataFrame['pricePerSquare'] = (inputDataFrame['price'] / inputDataFrame['total_square'])
    inputDataFrame = inputDataFrame.groupby(['city_district', 'building_type','exploitation_start_year']).min()
    inputDataFrame = inputDataFrame.drop('', level='city_district')
    inputDataFrame = inputDataFrame.drop('', level='building_type')
    inputDataFrame = inputDataFrame.reset_index(level='exploitation_start_year')
    Districts = inputDataFrame.index.get_level_values('city_district').drop_duplicates()
    f_bestOffer = open("best_offer.txt", "w")
    for cityDistrict in Districts:
        f_bestOffer.write('\n\n{}:\n'.format(cityDistrict))
        cityDataFrame = inputDataFrame.loc[(cityDistrict)]
        cityDataFrame = cityDataFrame.reset_index(level='building_type')
        cityDataFrame.drop_duplicates(subset='building_type', keep='last', inplace=True)
        print(cityDataFrame[['building_type','exploitation_start_year', 'pricePerSquare','source_url']])
        f_bestOffer.write(str(cityDataFrame[['building_type','exploitation_start_year', 'pricePerSquare','source_url','publication_date']]))
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
        return resultValues


parser = argparse.ArgumentParser()
#parser.add_argument("--database", type=str, default="mysql://sr:A4y8J6r4@149.154.71.73:3310/sr_dev" )
#parser.add_argument("--database", type=str, default="mysql://sr:password@portal.smartrealtor.pro:3306/smartrealtor" )
#parser.add_argument("--database", type=str, default="mysql://root:Intemp200784@127.0.0.1/smartRealtor" )
#parser.add_argument("--database", type=str, default="mysql://root:Intemp200784@127.0.0.1/smartRealtor?unix_socket=/var/run/mysqld/mysqld.sock" )
parser.add_argument("--database", type=str, default="mysql://root:UWmnjxPdN5ywjEcN@188.120.245.195:3306/domprice_dev1_v2?charset=utf8" )
#parser.add_argument("--input_table"   , type=str, default="real_estate_from_ads_api" )
parser.add_argument("--input_table"   , type=str, default="src_ads_raw" )
#parser.add_argument("--limits"  , type=str, default="input/MoscowLimits.json" )
#parser.add_argument("--output_table"   , type=str, default="real_estate_from_ads_api_processed" )
#parser.add_argument("--limits"  , type=str, default="input/KazanLimits.json" )
parser.add_argument("--limits"  , type=str, default="input/NizhnyNovgorodLimits.json" )

parser.add_argument("--start_date"  , type=str, default='2018-07-01' )
parser.add_argument("--end_date"  , type=str, default='2019-07-01')
#parser.add_argument("--limits"  , type=str, default="input/SaintPetersburgLimits.json" )
args = parser.parse_args()

input_tableName = args.input_table
databaseName = args.database
limitsName   = args.limits
start_date = datetime.datetime.strptime(args.start_date,'%Y-%m-%d')
end_date = datetime.datetime.strptime(args.end_date,'%Y-%m-%d')

inputDataFrame = None
inputDataFrame = loadDataFrame()(databaseName, limitsName, input_tableName )


cityDistrictDiagram(start_date,end_date,inputDataFrame)
#numberOfRoomsDiagram(start_date,end_date,inputDataFrame)
#buildingTypeDiagram(start_date,end_date,inputDataFrame)
#buildingTypeAndMeanPrice(start_date,end_date,inputDataFrame)
#dynamicsOfMeanPrice(start_date,end_date,inputDataFrame)
#buildingTypeAndMeanPriceCityDistricts(start_date,end_date,inputDataFrame)
#theBestoffer(end_date,inputDataFrame)
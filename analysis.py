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

import warnings

from pylab import rcParams
rcParams['figure.figsize'] = 15, 10
import matplotlib.dates as mdates


import datetime
color = ['orangered','dodgerblue','forestgreen','gold','c','crimson','chocolate','tomato','coral']
warnings.filterwarnings('ignore')
pd.options.display.max_columns = 52
pd.options.display.max_colwidth = 256

Districts_Moscow = {'Северный АО': ['Аэропорт','Беговой','Бескудниковский','Войковский','Головинский','Восточное Дегунино','Западное Дегунино',
                                    'Дмитровский','Коптево','Левобережный','Молжаниновский','Савеловский','Сокол','Тимирязевский','Ховрино','Хорошевский'],
                    'Северо-Восточный АО': ['Алексеевский','Алтуфьевский','Бабушкинский','Бибирево','Бутырский','Лианозово','Лосиноостровский','Марфино',
                                            'Марьина роща','Медведково','Останкийский','Отрадное','Ростокино','Свиблово','Северный','Ярославский'],
                    'Восточный АО': ['Богородское','Вешняки','Восточный','Гольяново','Ивановское','Измайлово','Косино-Ухтомский','Метрогородок',
                                     'Новогиреево','Новокосино','Перово','Преображенское','Соколиная гора','Сокольники'],
                    'Юго-Восточный АО': ['Выхино-Жулебино','Капотня','Кузьминки','Лефортово','Люблино','Марьино','Некрасовка','Нижегородский',
                                         'Печатники','Рязанский','Текстильщики','Южнопортовый'],
                    'Южный АО': ['Восточное Бирюлево','Западное Бирюлево','Братеево','Даниловский','Донской','Зябликово','Московоречье-Сабурово',
                                 'Нагатино-Садовники','Нагатинский затон','Нагорный',
                    'Северное Орехово-Борисово','Южное Орехово-Борисово','Царицыно','Северное Чертаново','Южное Чертаново','Центральное Чертаново'],
                    'Юго-Западный АО': ['Академический','Северное Бутово','Южное Бутово','Гагаринский','Зюзино','Коньково',
                                          'Котловка','Ломоносовский','Обручевский','Теплый Стан','Черемушки','Ясенево'],
                    'Западный АО': ['Внуково','Дорогомилово','Крылатское','Кунцево','Можайский','Ново-Переделкино','Очаково-Матвеевское',
                                    'Проспект Вернадского','Раменки','Солнцево','Тропарево-Никулино','Филевский парк','Фили-Давыдское'] ,
                    'Северо-Западный АО': ['Куркино', 'Митино', 'Покровское-Стрешнево', 'Строгино', 'Северное Тушино',
                                           'Южное Тушино', 'Хорошево-Мневники', 'Щукино'],
                    'Центральный АО': ['Арбат', 'Басманный', 'Замоскворечье', 'Красносельский', 'Мещанский',
                                       'Пресненский', 'Таганский', 'Тверской', 'Хамовники', 'Якиманка']}


def limitDataUsingProcentiles(dataFrame):
    if 'price' in dataFrame.columns:
        mask = True

        pricePerSquare = (dataFrame['price'] / dataFrame['total_square'])
        pricePerSquareValues = pricePerSquare.values

        robustScaler = RobustScaler(quantile_range=(1, 99))
        robustScaler.fit(pricePerSquareValues.reshape((-1, 1)))
        pricePerSquareValues = robustScaler.transform(pricePerSquareValues.reshape((-1, 1))).reshape(-1)

        mask = (pricePerSquareValues > -1) & (pricePerSquareValues < 1) & mask

        dataFrame = dataFrame[mask]

    return dataFrame

def cityDistrictDiagram(start_date,end_date,inputDataFrame,city):
    inputDataFrame = inputDataFrame.sort_values(by=['city_district'])
    inputDataFrame = inputDataFrame.groupby(['city_district']).size().reset_index(name='count_of_city_district')
    if city == 'Moscow':
        inputDataFrame = inputDataFrame.set_index('city_district')
        for Moscow_AO in Districts_Moscow.keys():
            Districts_AO = Districts_Moscow[Moscow_AO]
            Moscow_AODataFrame = inputDataFrame.loc[Districts_AO]
            df = pd.DataFrame(
                {'Территориальная структура рынка готового жилья': Moscow_AODataFrame['count_of_city_district'].values},
                index=Moscow_AODataFrame.index)
            df = df.dropna()
            print(df)
            df.plot.pie(y='Территориальная структура рынка готового жилья', autopct='%1.1f%%')
            plt.ylabel('', fontweight='bold')
            plt.title('Территориальная структура рынка готового жилья {}'.format(str(Moscow_AO)), fontweight='bold')
            plt.legend(loc='upper right')
            plt.savefig('/output/{0}/Территориальная структура рынка готового жилья {1}.png'.format(str(city),str(Moscow_AO)))
    else:
        print(inputDataFrame[['city_district','count_of_city_district']])
        df = pd.DataFrame({'Территориальная структура рынка готового жилья': inputDataFrame['count_of_city_district'].values }, index = inputDataFrame['city_district'])
        df = df[df.index != '']
        df.plot.pie(y='Территориальная структура рынка готового жилья', autopct='%1.1f%%')
        plt.ylabel('', fontweight='bold')
        plt.title('Территориальная структура рынка готового жилья', fontweight='bold')
        plt.legend(loc='upper right')
        if city == '':
            plt.show()
        else:
            plt.savefig('output/{0}/Территориальная структура рынка готового жилья.png'.format(str(city)))


def numberOfRoomsDiagram(start_date,end_date,inputDataFrame,city):
    inputDataFrame = inputDataFrame.groupby(['number_of_rooms']).size().reset_index(name='count_of_number_of_rooms')
    print(inputDataFrame[['number_of_rooms', 'count_of_number_of_rooms']])
    df = pd.DataFrame({'Структура рынка готового жилья по количеству комнат': [inputDataFrame.iloc[0]['count_of_number_of_rooms'],inputDataFrame.iloc[1]['count_of_number_of_rooms'],inputDataFrame.iloc[2]['count_of_number_of_rooms'],inputDataFrame.iloc[3]['count_of_number_of_rooms'], inputDataFrame.loc[inputDataFrame['number_of_rooms'] > 4, 'count_of_number_of_rooms'].sum()]}, index=['1-комн.','2-комн.','3-комн.','4-комн.','>4-комн.'])
    df.plot.pie(y='Структура рынка готового жилья по количеству комнат', autopct='%1.1f%%')
    plt.ylabel('', fontweight='bold')
    plt.title('Структура рынка готового жилья по количеству комнат', fontweight='bold')
    if city == '':
        plt.show()
    else:
        plt.savefig('output/{0}/Структура рынка готового жилья по количеству комнат.png'.format(str(city)))


def buildingTypeDiagram(start_date,end_date,inputDataFrame,city):
    inputDataFrame = inputDataFrame.groupby(['building_type']).size().reset_index(name='count_of_building_type')
    print(inputDataFrame[['building_type', 'count_of_building_type']])
    if 'Другое' in inputDataFrame['building_type'].values:
        inputDataFrame = inputDataFrame.loc[inputDataFrame['building_type'].values!='Другое']
    df = pd.DataFrame({'Структура рынка по типам готового жилья': inputDataFrame['count_of_building_type'].values}, index=inputDataFrame['building_type'])
    df = df[df.index != '']
    df.plot.pie(y='Структура рынка по типам готового жилья',autopct='%1.2f%%',fontsize=8)
    plt.ylabel('', fontweight='bold')
    plt.title('Структура рынка по типам готового жилья', fontweight='bold')
    if city == '':
        plt.show()
    else:
        plt.savefig('output/{0}/Структура рынка по типам готового жилья.png'.format(str(city)))


def dynamicsOfMeanPrice(start_date,end_date,inputDataFrame,city):
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
    if city == '':
        plt.show()
    else:
        plt.savefig('output/{0}/Изменение средней цены предложения на рынке готового жилья.png'.format(str(city)))


def buildingTypeAndMeanPrice(start_date,end_date,inputDataFrame,city):
    inputDataFrame['pricePerSquare'] = (inputDataFrame['price'] / inputDataFrame['total_square'])
    inputDataFrame = inputDataFrame.groupby(['building_type']).mean()
    print(inputDataFrame['pricePerSquare'])
    inputDataFrame = inputDataFrame[inputDataFrame.index != '']
    inputDataFrame = inputDataFrame[inputDataFrame.index !='Другое']
    fig, ax = plt.subplots()
    bars = ax.bar(x=inputDataFrame.index, height=inputDataFrame['pricePerSquare'],color=color)
    for p in ax.patches:
        ax.annotate(int(np.round(p.get_height(), decimals=0)), (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 10), fontweight='bold', fontsize=8, textcoords='offset points')
    plt.ylabel('руб./кв.м.',fontweight='bold')
    if city == '':
        plt.show()
    else:
        plt.savefig('output/{0}/Cредняя цена предложения по типам готового жилья.png'.format(str(city)))


def buildingTypeAndMeanPriceCityDistricts(start_date,end_date,inputDataFrame,city):
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
        if 'Другое' in buildingTypes:
            buildingTypes = buildingTypes.drop('Другое')

        zero_data = np.zeros(shape=(len(buildingTypes), 4))
        df = pd.DataFrame(zero_data, columns=['1-комн.', '2-комн.', '3-комн.', '4-комн. и больше'],index=buildingTypes)

        for buildingType in buildingTypes:

            for i in range(1, 4):
                if buildingType in cityDataFrame.loc[cityDataFrame['number_of_rooms'] == i].index:
                    smallDataFrame = cityDataFrame.loc[(buildingType)]
                    if type(smallDataFrame) is pd.core.series.Series and smallDataFrame['number_of_rooms'] == i: #if this dataframe is serie
                        print(type(smallDataFrame))
                        df.loc[buildingType, '{0}-комн.'.format(i)] = smallDataFrame['pricePerSquare']
                    else:
                        df.loc[buildingType, '{0}-комн.'.format(i)] = smallDataFrame.loc[smallDataFrame['number_of_rooms'] == i]['pricePerSquare'].to_numpy().astype(float)

            if buildingType in cityDataFrame.loc[cityDataFrame['number_of_rooms'] >= 4].index:
                smallDataFrame = cityDataFrame.loc[(buildingType)]
                if type(smallDataFrame) is pd.core.series.Series:  #if this dataframe is serie
                    df.loc[buildingType, '4-комн. и больше'] = smallDataFrame['pricePerSquare']
                else:
                    df.loc[buildingType, '4-комн. и больше'] = smallDataFrame.loc[smallDataFrame['number_of_rooms'] >= 4]['pricePerSquare'].mean()
        print(df)
        ax = df.plot.bar(width=1,color=color)
        for p in ax.patches:
            ax.annotate(int(np.round(p.get_height(),decimals=0)), (p.get_x()+p.get_width()/2., p.get_height()), ha='center', va='center', xytext=(0, 10),fontweight='bold',fontsize=8, textcoords='offset points')
        plt.ylabel('руб./кв.м.', fontweight='bold')
        plt.legend()
        plt.title('{} район средняя цена предложения по типам квартир и числу комнат'.format(cityDistrict), fontweight='bold')
        if city == '':
            plt.show()
        else:
            plt.savefig('output/{0}/{1} район средняя цена предложения по типам квартир и числу комнат.png'.format(str(city),cityDistrict))



class loadDataFrame(object):
    def __call__(self, databasePath, limitsFileName, tableName,start_date,end_date):
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
        sql_query += """ AND """ + """publication_date BETWEEN '{}' AND '{}'  """.format( start_date, end_date )

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
        #otherHouses = resultValues.loc[resultValues['building_type'] == 'Другое']
        #f_other_houses = open("other_houses.txt", "w")

        #f_other_houses.write(str(otherHouses[['exploitation_start_year', 'price', 'source_url', 'publication_date',
                                               #'longitude', 'latitude']]))
        #f_other_houses.flush()
        #f_other_houses.close()
        return resultValues


parser = argparse.ArgumentParser()

parser.add_argument("--database", type=str, default="mysql://root:UWmnjxPdN5ywjEcN@188.120.245.195:3306/domprice_dev1_v2?charset=utf8" )
parser.add_argument("--input_table"   , type=str, default="src_ads_raw_77" )
parser.add_argument("--limits"  , type=str, default="input/MoscowLimits.json" )

#parser.add_argument("--limits"  , type=str, default="input/KazanLimits.json" )
#parser.add_argument("--limits"  , type=str, default="input/NizhnyNovgorodLimits.json" )
#parser.add_argument("--limits"  , type=str, default="input/SaintPetersburgLimits.json" )
parser.add_argument("--start_date"  , type=str, default='2018-07-18' )
parser.add_argument("--end_date", type=str, default='2019-07-18')
parser.add_argument("--city", type=str, default='Moscow')

args = parser.parse_args()

input_tableName = args.input_table
databaseName = args.database
limitsName   = args.limits
start_date = datetime.datetime.strptime(args.start_date,'%Y-%m-%d')
end_date = datetime.datetime.strptime(args.end_date,'%Y-%m-%d')
city = args.city

inputDataFrame = None
inputDataFrame = loadDataFrame()(databaseName, limitsName, input_tableName,start_date,end_date )


cityDistrictDiagram(start_date,end_date,inputDataFrame,city)
numberOfRoomsDiagram(start_date,end_date,inputDataFrame,city)
buildingTypeDiagram(start_date,end_date,inputDataFrame,city)
buildingTypeAndMeanPrice(start_date,end_date,inputDataFrame,city)
dynamicsOfMeanPrice(start_date,end_date,inputDataFrame,city)
buildingTypeAndMeanPriceCityDistricts(start_date,end_date,inputDataFrame,city)

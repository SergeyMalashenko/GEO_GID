#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from os.path import join, dirname
from dotenv import load_dotenv
import os
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
rcParams['figure.figsize'] = 15, 12
plt.rcParams.update({'font.size': 16})
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

def cityDistrictDiagram(inputDataFrame,city,output_Folder):
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
            df = df.apply(lambda x: 100 * x / float(x.sum()))
            print(df)
            df = df.sort_values(by='Территориальная структура рынка готового жилья',ascending=False)
            with open('{2}/{0}/Территориальная структура рынка готового жилья {1}.json'.format(str(city),str(Moscow_AO),
                                                            str(output_Folder)), "w", encoding="utf-8") as write_file:
                json.dump(df.to_json(force_ascii=False,double_precision=2), write_file, ensure_ascii=False)
            df.plot.pie(y='Территориальная структура рынка готового жилья', autopct='%1.1f%%')

            plt.ylabel('', fontweight='bold')
            plt.title('Территориальная структура рынка готового жилья {}'.format(str(Moscow_AO)), fontweight='bold', fontsize = 20)

            plt.tick_params(axis='both', which='major', labelsize=16)
            plt.tight_layout()
            plt.savefig('{2}/{0}/Территориальная структура рынка готового жилья {1}.png'.format(str(city),str(Moscow_AO),str(output_Folder)))
    else:
        df = pd.DataFrame({'Территориальная структура рынка готового жилья': inputDataFrame['count_of_city_district'].values }, index = inputDataFrame['city_district'])
        df = df[df.index != '']
        df = df.apply(lambda x: 100 * x / float(x.sum()))
        df = df.sort_values(by='Территориальная структура рынка готового жилья',ascending=False)
        print(df)
        with open('{1}/{0}/Территориальная структура рынка готового жилья.json'.format(str(city),str(output_Folder)),
                  "w", encoding="utf-8") as write_file:
            json.dump(df.to_json(force_ascii=False,double_precision=2), write_file, ensure_ascii=False)
        df.plot.pie(y='Территориальная структура рынка готового жилья', autopct='%1.1f%%')
        plt.ylabel('', fontweight='bold')
        plt.title('Территориальная структура рынка готового жилья', fontweight='bold', fontsize = 20)
        plt.legend(label=None)
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.tight_layout()
        if city == '':
            plt.show()
        else:
            plt.savefig('{1}/{0}/Территориальная структура рынка готового жилья.png'.format(str(city),str(output_Folder)))


def numberOfRoomsDiagram(inputDataFrame,city,output_Folder):
    inputDataFrame = inputDataFrame.groupby(['number_of_rooms']).size().reset_index(name='count_of_number_of_rooms')
    print(inputDataFrame[['number_of_rooms', 'count_of_number_of_rooms']])
    df = pd.DataFrame({'Структура рынка готового жилья по количеству комнат': [inputDataFrame.iloc[0]['count_of_number_of_rooms'],inputDataFrame.iloc[1]['count_of_number_of_rooms'],inputDataFrame.iloc[2]['count_of_number_of_rooms'],inputDataFrame.iloc[3]['count_of_number_of_rooms'], inputDataFrame.loc[inputDataFrame['number_of_rooms'] > 4, 'count_of_number_of_rooms'].sum()]}, index=['1-комн.','2-комн.','3-комн.','4-комн.','>4-комн.'])
    df.plot.pie(y='Структура рынка готового жилья по количеству комнат', autopct='%1.1f%%')
    df = df.apply(lambda x: 100 * x / float(x.sum()))
    print(df)
    df = df.sort_values(by='Структура рынка готового жилья по количеству комнат',ascending=False)
    with open('{1}/{0}/Структура рынка готового жилья по количеству комнат.json'.format(str(city),str(output_Folder)), "w", encoding="utf-8") as write_file:
        json.dump(df.to_json(force_ascii=False,double_precision=2), write_file, ensure_ascii=False)
    plt.ylabel('', fontweight='bold')
    plt.title('Структура рынка готового жилья по количеству комнат', fontweight='bold', fontsize = 20)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.tight_layout()
    if city == '':
        plt.show()
    else:
        plt.savefig('{1}/{0}/Структура рынка готового жилья по количеству комнат.png'.format(str(city),str(output_Folder)))


def buildingTypeDiagram(inputDataFrame,city,output_Folder):
    inputDataFrame = inputDataFrame.groupby(['building_type']).size().reset_index(name='count_of_building_type')
    print(inputDataFrame[['building_type', 'count_of_building_type']])
    if 'Другое' in inputDataFrame['building_type'].values:
        inputDataFrame = inputDataFrame.loc[inputDataFrame['building_type'].values!='Другое']
    df = pd.DataFrame({'Структура рынка по типам готового жилья': inputDataFrame['count_of_building_type'].values}, index=inputDataFrame['building_type'])
    df = df[df.index != '']
    df = df.apply(lambda x: 100 * x / float(x.sum()))

    fig, ax = plt.subplots()
    plt.tick_params(axis='both', which='major', labelsize=12)
    i = 0
    for tick in ax.get_xaxis().get_major_ticks():
        if (i % 2) == 0:
            tick.set_pad(26.)
        else:
            tick.set_pad(4.)
        tick.label1 = tick._get_text1()
        i += 1
    bars = ax.bar(x=df.index, height=df['Структура рынка по типам готового жилья'],
                  color=color, align='center')
    for p in ax.patches:
        ax.annotate("{:.2f}%".format(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 10), fontweight='bold', fontsize=12,
                    textcoords='offset points')
    df = df.sort_values(by='Структура рынка по типам готового жилья', ascending=False)
    with open('{1}/{0}/Структура рынка по типам готового жилья.json'.format(str(city), str(output_Folder)), "w",
              encoding="utf-8") as write_file:
        json.dump(df.to_json(force_ascii=False, double_precision=2), write_file, ensure_ascii=False)
    plt.ylabel('', fontweight='bold')
    plt.legend()
    plt.yticks(labels=None)
    plt.title('Структура рынка по типам готового жилья', fontweight='bold', fontsize = 20)
    plt.tight_layout()
    if city == '':
        plt.show()
    else:
        plt.savefig('{1}/{0}/Структура рынка по типам готового жилья.png'.format(str(city),str(output_Folder)))


def dynamicsOfMeanPrice(inputDataFrame,city,output_Folder):
    inputDataFrame['pricePerSquare'] = (inputDataFrame['price'] / inputDataFrame['total_square'])
    inputDataFrame = inputDataFrame.groupby(['publication_date']).mean()
    inputDataFrame = inputDataFrame.groupby(pd.Grouper(freq='MS')).mean()
    print(inputDataFrame['pricePerSquare'])

    fig, ax = plt.subplots()
    bars =  ax.bar(x=inputDataFrame.index, height=inputDataFrame['pricePerSquare'],width=25,color=['cadetblue','skyblue'],align='center')
    for p in ax.patches:
        ax.annotate(int(np.round(p.get_height(), decimals=0)), (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 10), fontweight='bold', fontsize=12, textcoords='offset points')
    deviation_of_price = (inputDataFrame.iloc[len(inputDataFrame)-1]['pricePerSquare']-inputDataFrame.iloc[0]['pricePerSquare'])/inputDataFrame.iloc[0]['pricePerSquare']
    plt.text(x=0.95,y=0.975,s="{:.1%}".format(deviation_of_price), fontsize = 16,fontweight='bold', bbox=dict(facecolor='red', alpha=0.5),transform=ax.transAxes)
    plt.ylabel('руб./кв.м.',fontweight='bold', fontsize = 16)
    plt.tick_params(axis='both', which='major', labelsize=10)
    months = mdates.MonthLocator()
    months_fmt = mdates.DateFormatter('%m-%Y')
    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(months_fmt)
    ax.xaxis.set_minor_locator(months)
    ax.xaxis.set_major_formatter(months_fmt)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.xticks(rotation='vertical')
    plt.title('Изменение средней цены предложения на рынке готового жилья',fontweight='bold', fontsize = 20)
    plt.tight_layout()
    if city == '':
        plt.show()
    else:
        plt.savefig('{1}/{0}/Изменение средней цены предложения на рынке готового жилья.png'.format(str(city),str(output_Folder)))
    inputDataFrame.index = inputDataFrame.index.strftime("%Y-%m")
    with open('{1}/{0}/Изменение средней цены предложения на рынке готового жилья.json'.format(str(city),str(output_Folder)), "w", encoding="utf-8") as write_file:
        json.dump(inputDataFrame['pricePerSquare'].to_json(force_ascii=False,date_format='iso',double_precision=0), write_file, ensure_ascii=False)


def buildingTypeAndMeanPrice(inputDataFrame,city,output_Folder):
    inputDataFrame['pricePerSquare'] = (inputDataFrame['price'] / inputDataFrame['total_square'])
    inputDataFrame = inputDataFrame.groupby(['building_type']).mean()
    print(inputDataFrame['pricePerSquare'])
    inputDataFrame = inputDataFrame[inputDataFrame.index != '']
    inputDataFrame = inputDataFrame[inputDataFrame.index !='Другое']
    with open('{1}/{0}/Cредняя цена предложения по типам готового жилья.json'.format(str(city),str(output_Folder)), "w", encoding="utf-8") as write_file:
        json.dump(inputDataFrame['pricePerSquare'].to_json(force_ascii=False,double_precision=0), write_file, ensure_ascii=False)

    fig, ax = plt.subplots()
    plt.tick_params(axis='both', which='major', labelsize=12)
    i = 0
    for tick in ax.get_xaxis().get_major_ticks():
        if (i % 2) == 0:
            tick.set_pad(26.)
        else:
            tick.set_pad(4.)
        tick.label1 = tick._get_text1()
        i+=1
    bars = ax.bar(x=inputDataFrame.index, height=inputDataFrame['pricePerSquare'],color=color)
    for p in ax.patches:
        ax.annotate(int(np.round(p.get_height(), decimals=0)), (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 10), fontweight='bold', fontsize=16, textcoords='offset points')
    plt.ylabel('руб./кв.м.',fontweight='bold', fontsize = 16)
    inputDataFrame = inputDataFrame.sort_values(by='pricePerSquare',ascending=False)
    print(inputDataFrame['pricePerSquare'])

    plt.title('Cредняя цена предложения по типам готового жилья', fontweight='bold', fontsize=20)
    plt.tight_layout()
    if city == '':
        plt.show()
    else:
        plt.savefig('{1}/{0}/Cредняя цена предложения по типам готового жилья.png'.format(str(city),str(output_Folder)))

def cityDistrictAndMeanPrice(inputDataFrame,city,output_Folder):
    inputDataFrame['pricePerSquare'] = (inputDataFrame['price'] / inputDataFrame['total_square'])
    inputDataFrame = inputDataFrame.groupby(['city_district']).mean()
    print(inputDataFrame['pricePerSquare'])
    inputDataFrame = inputDataFrame[inputDataFrame.index != '']
    if city == 'Moscow':
        for Moscow_AO in Districts_Moscow.keys():
            Districts_AO = Districts_Moscow[Moscow_AO]
            Moscow_AODataFrame = inputDataFrame.loc[Districts_AO]
        fig, ax = plt.subplots()
        plt.tick_params(axis='both', which='major', labelsize=15)
        i = 0
        for tick in ax.get_xaxis().get_major_ticks():
            if (i % 2) == 0:
                tick.set_pad(26.)
            else:
                tick.set_pad(4.)
            tick.label1 = tick._get_text1()
            i += 1
        bars = ax.bar(x=Moscow_AODataFrame.index, height=Moscow_AODataFrame['pricePerSquare'],color=color)
        for p in ax.patches:
            ax.annotate(int(np.round(p.get_height(), decimals=0)), (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 10), fontweight='bold', fontsize=16, textcoords='offset points')
        plt.ylabel('руб./кв.м.',fontweight='bold', fontsize = 16)
        inputDataFrame = inputDataFrame.sort_values(by='pricePerSquare',ascending=False)
        print(inputDataFrame['pricePerSquare'])
        with open('{1}/{0}/Cредняя цена предложения по административным районам {2}.json'.format(str(city),str(output_Folder),str(Moscow_AO)), "w", encoding="utf-8") as write_file:
            json.dump(inputDataFrame['pricePerSquare'].to_json(force_ascii=False,double_precision=0), write_file, ensure_ascii=False)
        plt.title('Cредняя цена предложения по административным районам {}'.format(str(Moscow_AO)), fontweight='bold', fontsize=20)
        i = 0
        for tick in ax.get_xaxis().get_major_ticks():
            if (i % 2) == 0:
                tick.set_pad(26.)
            else:
                tick.set_pad(4.)
            tick.label1 = tick._get_text1()
            i += 1
        plt.tight_layout()
        plt.savefig(
            '{1}/{0}/Cредняя цена предложения по административным районам {2}.png'.format(str(city), str(output_Folder),str(Moscow_AO)))
    else:
        fig, ax = plt.subplots()
        plt.tick_params(axis='both', which='major', labelsize=15)
        bars = ax.bar(x=inputDataFrame.index, height=inputDataFrame['pricePerSquare'], color=color)
        for p in ax.patches:
            ax.annotate(int(np.round(p.get_height(), decimals=0)), (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 10), fontweight='bold', fontsize=16,
                        textcoords='offset points')
        plt.ylabel('руб./кв.м.', fontweight='bold', fontsize=16)
        inputDataFrame = inputDataFrame.sort_values(by='pricePerSquare', ascending=False)
        print(inputDataFrame['pricePerSquare'])
        with open('{1}/{0}/Cредняя цена предложения по административным районам.json'.format(str(city),
                                                                                             str(output_Folder)), "w",
                  encoding="utf-8") as write_file:
            json.dump(inputDataFrame['pricePerSquare'].to_json(force_ascii=False, double_precision=0), write_file,
                      ensure_ascii=False)
        plt.title('Cредняя цена предложения по административным районам', fontweight='bold', fontsize=20)
        i = 0
        for tick in ax.get_xaxis().get_major_ticks():
            if (i % 2) == 0:
                tick.set_pad(26.)
            else:
                tick.set_pad(4.)
            tick.label1 = tick._get_text1()
            i += 1
        plt.tight_layout()
        if city == '':
            plt.show()
        else:
            plt.savefig('{1}/{0}/Cредняя цена предложения по административным районам.png'.format(str(city),str(output_Folder)))

def buildingTypeAndMeanPriceCityDistricts(inputDataFrame,city,output_Folder):
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
        with open('{2}/{0}/{1} район средняя цена предложения по типам квартир и числу комнат.json'.format(str(city),cityDistrict,str(output_Folder)),"w",
                  encoding="utf-8") as write_file:
            json.dump(df.to_json(force_ascii=False,double_precision=0), write_file, ensure_ascii=False)
        for p in ax.patches:
            ax.annotate(int(np.round(p.get_height(),decimals=0)), (p.get_x()+p.get_width()/2., p.get_height()), ha='center', va='center', xytext=(0, 10),fontweight='bold',fontsize=9, textcoords='offset points')
        plt.ylabel('руб./кв.м.', fontweight='bold', fontsize = 16)
        plt.legend()
        plt.xlabel('')
        plt.tick_params(axis='both', which='major', labelsize=13)
        plt.xticks(rotation='horizontal')
        plt.title('{} район средняя цена предложения по типам квартир и числу комнат'.format(cityDistrict), fontweight='bold', fontsize = 16)
        i = 0
        for tick in ax.get_xaxis().get_major_ticks():
            if (i % 2) == 0:
                tick.set_pad(26.)
            else:
                tick.set_pad(4.)
            tick.label1 = tick._get_text1()
            i += 1
        plt.tight_layout()
        if city == '':
            plt.show()
        else:
            plt.savefig('{2}/{0}/{1} район средняя цена предложения по типам квартир и числу комнат.png'.format(str(city),cityDistrict,str(output_Folder)))



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

parser.add_argument("--start_date"  , type=str, default='')
parser.add_argument("--end_date", type=str, default='')
parser.add_argument("--city", type=str, default='Kazan')

args = parser.parse_args()
# Create .env file path.
dotenv_path = join(dirname(__file__), '.env')

# Load file from the path.
load_dotenv(dotenv_path)
# Accessing variables.
databaseName = os.getenv('DATABASE_URL')
output_Folder = os.getenv('OUTPUT_FOLDER')
start_date = args.start_date
end_date = args.end_date
#default settings for date:
if end_date == '':
    end_date = datetime.date.today()
else:
    end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')
if start_date == '':
    start_date = end_date - datetime.timedelta(days=365)
else:
    start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')

city = args.city
inputTableDict = {'Nizhny Novgorod': 'src_ads_raw_52','Kazan': 'src_ads_raw_16',
                  'Saint Petersburg': 'src_ads_raw_78','Moscow': 'src_ads_raw_77'}
limitsDict = {'Nizhny Novgorod': 'input/NizhnyNovgorodLimits.json','Kazan': 'input/KazanLimits.json',
                  'Saint Petersburg': 'input/SaintPetersburgLimits.json','Moscow': 'input/MoscowLimits.json'}
input_tableName = inputTableDict[city]
limitsName   = limitsDict[city]

inputDataFrame = None
inputDataFrame = loadDataFrame()(databaseName, limitsName, input_tableName,start_date,end_date )
d = {'count_ads': len(inputDataFrame.index)}
with open('{1}/{0}/metadata.json'.format(str(city), str(output_Folder)),
          "w", encoding="utf-8") as write_file:
    json.dump(d, write_file, ensure_ascii=False)
cityDistrictDiagram(inputDataFrame,city,output_Folder)
numberOfRoomsDiagram(inputDataFrame,city,output_Folder)
buildingTypeDiagram(inputDataFrame,city,output_Folder)
buildingTypeAndMeanPrice(inputDataFrame,city,output_Folder)
cityDistrictAndMeanPrice(inputDataFrame,city,output_Folder)
dynamicsOfMeanPrice(inputDataFrame,city,output_Folder)
buildingTypeAndMeanPriceCityDistricts(inputDataFrame,city,output_Folder)

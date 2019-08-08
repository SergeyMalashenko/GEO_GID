# MODELNAME_=best_model/modelNizhnyNovgorod_0.0001.pkl
# MODELNAME_=modelNizhnyNovgorod_.pkl
export MODELNAME_ML="XGBoost/XGBoost_NizhnyNovgorod.pkl"
export MODELNAME_NN="temp/modelNizhnyNovgorod.pkl"
export DATABASE_="mysql://root:passwordN@188.120.245.195:3306/domprice_dev1_v2"
export TABLE_="src_ads_raw_52_processed"
export MODE_="--verbose"
export OUTPUT_FILENAME="output_NizhnyNovgorod.txt"


echo Link: https://www.avito.ru/nizhniy_novgorod/kvartiry/3-k_kvartira_63_m_35_et._1229410388 $MODE_--database $DATABASE_ --table $TABLE_ > $OUTPUT_FILENAME
echo  RealPrice: 3500000 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo XGBoost >> $OUTPUT_FILENAME
./testModelMachineLearning.py --model $MODELNAME_ML --query "longitude=43.985530, latitude=56.269815, total_square=63, living_square=42, kitchen_square=7, number_of_rooms=3, floor_number=3, number_of_floors=5, exploitation_start_year=1969" --limits input/NizhnyNovgorodLimits.json  $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo Neuron Network >> $OUTPUT_FILENAME
./testModel.py --model $MODELNAME_NN --query "longitude=43.985530, latitude=56.269815, total_square=63, living_square=42, kitchen_square=7, number_of_rooms=3, floor_number=3, number_of_floors=5, exploitation_start_year=1969" --limits input/NizhnyNovgorodLimits.json  $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME

echo Link: https://www.avito.ru/nizhniy_novgorod/kvartiry/2-k_kvartira_75_m_312_et._1330600258 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo RealPrice: 6900000 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo XGBoost >> $OUTPUT_FILENAME
./testModelMachineLearning.py --model $MODELNAME_ML   --query "longitude=43.956893, latitude=56.334829, total_square=75, living_square=56, kitchen_square=11, number_of_rooms=2, floor_number=3, number_of_floors=12, exploitation_start_year=2007" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo Neuron Network >> $OUTPUT_FILENAME
./testModel.py --model $MODELNAME_NN --query "longitude=43.956893, latitude=56.334829, total_square=75, living_square=56, kitchen_square=11, number_of_rooms=2, floor_number=3, number_of_floors=12, exploitation_start_year=2007" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME

echo Link: https://www.avito.ru/nizhniy_novgorod/kvartiry/1-k_kvartira_34.5_m_49_et._1188266556 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo RealPrice: 2050000 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo XGBoost >> $OUTPUT_FILENAME
./testModelMachineLearning.py --model $MODELNAME_ML --query "longitude=43.800627, latitude=56.369655, total_square=34.5, living_square=17, kitchen_square=7.5, number_of_rooms=1, floor_number=4, number_of_floors=9, exploitation_start_year=1983" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo Neuron Network >> $OUTPUT_FILENAME
./testModel.py --model $MODELNAME_NN --query "longitude=43.800627, latitude=56.369655, total_square=34.5, living_square=17, kitchen_square=7.5, number_of_rooms=1, floor_number=4, number_of_floors=9, exploitation_start_year=1983" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME

echo Link: https://www.avito.ru/nizhniy_novgorod/kvartiry/2-k_kvartira_44_m_15_et._1021003671 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo RealPrice: 1950000 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo XGBoost >> $OUTPUT_FILENAME
./testModelMachineLearning.py --model $MODELNAME_ML  --query "longitude=43.909747, latitude=56.273713, total_square=44, living_square=25, kitchen_square=6, number_of_rooms=2, floor_number=1, number_of_floors=5, exploitation_start_year=1964" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo Neuron Network >> $OUTPUT_FILENAME
./testModel.py --model $MODELNAME_NN --query "longitude=43.909747, latitude=56.273713, total_square=44, living_square=25, kitchen_square=6, number_of_rooms=2, floor_number=1, number_of_floors=5, exploitation_start_year=1964" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME

echo Link: https://www.avito.ru/nizhniy_novgorod/kvartiry/1-k_kvartira_33_m_610_et._1390717765 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo RealPrice: 2380000 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo XGBoost >> $OUTPUT_FILENAME
./testModelMachineLearning.py --model $MODELNAME_ML  --query "longitude=43.841315, latitude=56.249576, total_square=33, living_square=17.2, kitchen_square=8.3, number_of_rooms=1, floor_number=6, number_of_floors=10, exploitation_start_year=1995" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo Neuron Network >> $OUTPUT_FILENAME
./testModel.py --model $MODELNAME_NN --query "longitude=43.841315, latitude=56.249576, total_square=33, living_square=17.2, kitchen_square=8.3, number_of_rooms=1, floor_number=6, number_of_floors=10, exploitation_start_year=1995" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME

echo Link: https://www.avito.ru/nizhniy_novgorod/kvartiry/3-k_kvartira_63_m_29_et._1269454476 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo RealPrice: 4500000 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo XGBoost >> $OUTPUT_FILENAME
./testModelMachineLearning.py --model $MODELNAME_ML  --query "longitude=43.845803, latitude=56.253039, total_square=63, living_square=39, kitchen_square=9, number_of_rooms=3, floor_number=2, number_of_floors=9, exploitation_start_year=1985" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo Neuron Network >> $OUTPUT_FILENAME
./testModel.py --model $MODELNAME_NN --query "longitude=43.845803, latitude=56.253039, total_square=63, living_square=39, kitchen_square=9, number_of_rooms=3, floor_number=2, number_of_floors=9, exploitation_start_year=1985" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME

echo Link: https://www.avito.ru/nizhniy_novgorod/kvartiry/2-k_kvartira_52_m_89_et._1504332021 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo RealPrice: 3250000 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo XGBoost >> $OUTPUT_FILENAME
./testModelMachineLearning.py --model $MODELNAME_ML  --query "longitude=43.934205, latitude=56.326687, total_square=52, living_square=30, kitchen_square=9.5, number_of_rooms=2, floor_number=8, number_of_floors=9, exploitation_start_year=1985" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo Neuron Network >> $OUTPUT_FILENAME
./testModel.py --model $MODELNAME_NN --query "longitude=43.934205, latitude=56.326687, total_square=52, living_square=30, kitchen_square=9.5, number_of_rooms=2, floor_number=8, number_of_floors=9, exploitation_start_year=1985" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME

echo Link: https://www.avito.ru/nizhniy_novgorod/kvartiry/1-k_kvartira_41_m_710_et._1242583545 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo RealPrice: 3550000 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo XGBoost >> $OUTPUT_FILENAME
./testModelMachineLearning.py --model $MODELNAME_ML  --query "longitude=43.966681, latitude=56.237371, total_square=41, living_square=18, kitchen_square=8, number_of_rooms=1, floor_number=7, number_of_floors=10, exploitation_start_year=2009" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo Neuron Network >> $OUTPUT_FILENAME
./testModel.py --model $MODELNAME_NN --query "longitude=43.966681, latitude=56.237371, total_square=41, living_square=18, kitchen_square=8, number_of_rooms=1, floor_number=7, number_of_floors=10, exploitation_start_year=2009" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME

echo Link: https://www.avito.ru/nizhniy_novgorod/kvartiry/2-k_kvartira_37.1_m_22_et._1252074608 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo RealPrice: 1900000 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo XGBoost >> $OUTPUT_FILENAME
./testModelMachineLearning.py --model $MODELNAME_ML  --query "longitude=43.848245, latitude=56.227080, total_square=37.1, living_square=27.4, kitchen_square=4.6, number_of_rooms=2, floor_number=2, number_of_floors=2, exploitation_start_year=1962" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo Neuron Network >> $OUTPUT_FILENAME
./testModel.py --model $MODELNAME_NN --query "longitude=43.848245, latitude=56.227080, total_square=37.1, living_square=27.4, kitchen_square=4.6, number_of_rooms=2, floor_number=2, number_of_floors=2, exploitation_start_year=1962" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME

echo Link: https://www.avito.ru/nizhniy_novgorod/kvartiry/1-k_kvartira_42.7_m_610_et._1311304542 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo RealPrice: 3600000 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo XGBoost >> $OUTPUT_FILENAME
./testModelMachineLearning.py --model $MODELNAME_ML  --query "longitude=43.936267, latitude=56.325305, total_square=42.7, living_square=18.6, kitchen_square=13.2, number_of_rooms=1, floor_number=6, number_of_floors=10, exploitation_start_year=2007" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo Neuron Network >> $OUTPUT_FILENAME
./testModel.py --model $MODELNAME_NN --query "longitude=43.936267, latitude=56.325305, total_square=42.7, living_square=18.6, kitchen_square=13.2, number_of_rooms=1, floor_number=6, number_of_floors=10, exploitation_start_year=2007" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME

echo Link: https://www.avito.ru/nizhniy_novgorod/kvartiry/1-k_kvartira_32.5_m_49_et._1625915692 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo RealPrice: 2200000 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo XGBoost >> $OUTPUT_FILENAME
./testModelMachineLearning.py --model $MODELNAME_ML  --query "longitude=43.858459, latitude=56.264565, total_square=32.5, living_square=17.1, kitchen_square=6.8, number_of_rooms=1, floor_number=4, number_of_floors=9, exploitation_start_year=1980" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo Neuron Network >> $OUTPUT_FILENAME
./testModel.py --model $MODELNAME_NN --query "longitude=43.858459, latitude=56.264565, total_square=32.5, living_square=17.1, kitchen_square=6.8, number_of_rooms=1, floor_number=4, number_of_floors=9, exploitation_start_year=1980" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME

echo Link: https://www.avito.ru/nizhniy_novgorod/kvartiry/2-k_kvartira_50.1_m_59_et._1613178940 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo RealPrice: 3650000 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo XGBoost >> $OUTPUT_FILENAME
./testModelMachineLearning.py --model $MODELNAME_ML  --query "longitude=44.071471, latitude=56.297274, total_square=50.1, living_square=30, kitchen_square=9, number_of_rooms=2, floor_number=5, number_of_floors=9, exploitation_start_year=1989" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo Neuron Network >> $OUTPUT_FILENAME
./testModel.py --model $MODELNAME_NN  --query "longitude=44.071471, latitude=56.297274, total_square=50.1, living_square=30, kitchen_square=9, number_of_rooms=2, floor_number=5, number_of_floors=9, exploitation_start_year=1989" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME

echo Link: https://www.avito.ru/nizhniy_novgorod/kvartiry/1-k_kvartira_40_m_1617_et._1198204435 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo RealPrice: 3450000 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo XGBoost >> $OUTPUT_FILENAME
./testModelMachineLearning.py --model $MODELNAME_ML  --query "longitude=44.069970, latitude=56.314628, total_square=40, living_square=10.5, kitchen_square=17, number_of_rooms=1, floor_number=16, number_of_floors=17, exploitation_start_year=2008" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo Neuron Network >> $OUTPUT_FILENAME
./testModel.py --model $MODELNAME_NN --query "longitude=44.069970, latitude=56.314628, total_square=40, living_square=10.5, kitchen_square=17, number_of_rooms=1, floor_number=16, number_of_floors=17, exploitation_start_year=2008" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME

echo Link: https://www.avito.ru/nizhniy_novgorod/kvartiry/2-k_kvartira_42_m_45_et._1542642597 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo RealPrice: 2850000 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo XGBoost >> $OUTPUT_FILENAME
./testModelMachineLearning.py --model $MODELNAME_ML  --query "longitude=44.020939, latitude=56.292668, total_square=42, living_square=27.3, kitchen_square=6, number_of_rooms=2, floor_number=4, number_of_floors=5, exploitation_start_year=1975" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo Neuron Network >> $OUTPUT_FILENAME
./testModel.py --model $MODELNAME_NN --query "longitude=44.020939, latitude=56.292668, total_square=42, living_square=27.3, kitchen_square=6, number_of_rooms=2, floor_number=4, number_of_floors=5, exploitation_start_year=1975" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME

echo Link: https://www.avito.ru/nizhniy_novgorod/kvartiry/1-k_kvartira_31_m_55_et._1464877757 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo RealPrice: 1900000 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo XGBoost >> $OUTPUT_FILENAME
./testModelMachineLearning.py --model $MODELNAME_ML  --query "longitude=43.984330, latitude=56.274786, total_square=31, living_square=17, kitchen_square=6, number_of_rooms=1, floor_number=5, number_of_floors=5, exploitation_start_year=1962" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo Neuron Network >> $OUTPUT_FILENAME
./testModel.py --model $MODELNAME_NN --query "longitude=43.984330, latitude=56.274786, total_square=31, living_square=17, kitchen_square=6, number_of_rooms=1, floor_number=5, number_of_floors=5, exploitation_start_year=1962" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME

echo Link: https://www.avito.ru/nizhniy_novgorod/kvartiry/3-k_kvartira_120.8_m_411_et._1593622798 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo RealPrice: 8100000 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo XGBoost >> $OUTPUT_FILENAME
./testModelMachineLearning.py --model $MODELNAME_ML --query "longitude=43.976535, latitude=56.300300, total_square=120.8, living_square=63.1, kitchen_square=21.8, number_of_rooms=3, floor_number=4, number_of_floors=11, exploitation_start_year=2017" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo Neuron Network >> $OUTPUT_FILENAME
./testModel.py --model $MODELNAME_NN --query "longitude=43.976535, latitude=56.300300, total_square=120.8, living_square=63.1, kitchen_square=21.8, number_of_rooms=3, floor_number=4, number_of_floors=11, exploitation_start_year=2017" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME

echo Link: https://www.avito.ru/nizhniy_novgorod/kvartiry/3-k_kvartira_80.8_m_48_et._1561196678 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo RealPrice: 6400000 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo XGBoost >> $OUTPUT_FILENAME
./testModelMachineLearning.py --model $MODELNAME_ML  --query "longitude=43.978952, latitude=56.303685, total_square=80.8, living_square=45.4, kitchen_square=14.7, number_of_rooms=3, floor_number=4, number_of_floors=8, exploitation_start_year=2003" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo Neuron Network >> $OUTPUT_FILENAME
./testModel.py --model $MODELNAME_NN --query "longitude=43.978952, latitude=56.303685, total_square=80.8, living_square=45.4, kitchen_square=14.7, number_of_rooms=3, floor_number=4, number_of_floors=8, exploitation_start_year=2003" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME

echo Link: https://www.avito.ru/nizhniy_novgorod/kvartiry/1-k_kvartira_45_m_25_et._1043145473 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo RealPrice: 3750000 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo XGBoost >> $OUTPUT_FILENAME
./testModelMachineLearning.py --model $MODELNAME_ML --query "longitude=43.988275, latitude=56.312874, total_square=45, living_square=28, kitchen_square=7, number_of_rooms=1, floor_number=2, number_of_floors=5, exploitation_start_year=1962" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo Neuron Network >> $OUTPUT_FILENAME
./testModel.py --model $MODELNAME_NN --query "longitude=43.988275, latitude=56.312874, total_square=45, living_square=28, kitchen_square=7, number_of_rooms=1, floor_number=2, number_of_floors=5, exploitation_start_year=1962" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME

echo Link: https://www.avito.ru/nizhniy_novgorod/kvartiry/3-k_kvartira_55.9_m_25_et._1555009776 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo RealPrice: 4700000 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo XGBoost >> $OUTPUT_FILENAME
./testModelMachineLearning.py --model $MODELNAME_ML  --query "longitude=43.984681, latitude=56.324555, total_square=55.9, living_square=41, kitchen_square=6.1, number_of_rooms=3, floor_number=2, number_of_floors=5, exploitation_start_year=1965" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo Neuron Network >> $OUTPUT_FILENAME
./testModel.py --model $MODELNAME_NN --query "longitude=43.984681, latitude=56.324555, total_square=55.9, living_square=41, kitchen_square=6.1, number_of_rooms=3, floor_number=2, number_of_floors=5, exploitation_start_year=1965" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME

echo Link: https://www.avito.ru/nizhniy_novgorod/kvartiry/4-k_kvartira_135_m_48_et._1673043252 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo RealPrice: 17000000 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo XGBoost >> $OUTPUT_FILENAME
./testModelMachineLearning.py --model $MODELNAME_ML  --query "longitude=44.022726, latitude=56.326732, total_square=135, living_square=82, kitchen_square=15, number_of_rooms=4, floor_number=4, number_of_floors=8, exploitation_start_year=2000" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo Neuron Network >> $OUTPUT_FILENAME
./testModel.py --model $MODELNAME_NN  --query "longitude=44.022726, latitude=56.326732, total_square=135, living_square=82, kitchen_square=15, number_of_rooms=4, floor_number=4, number_of_floors=8, exploitation_start_year=2000" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME

echo Link: https://www.avito.ru/nizhniy_novgorod/kvartiry/2-k_kvartira_55.5_m_66_et._1241801222 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo RealPrice: 4300000 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo XGBoost >> $OUTPUT_FILENAME
./testModelMachineLearning.py --model $MODELNAME_ML  --query "longitude=44.020036, latitude= 56.317545, total_square=55.5, living_square=18, kitchen_square=7.7, number_of_rooms=2, floor_number=6, number_of_floors=6, exploitation_start_year=1992" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo Neuron Network >> $OUTPUT_FILENAME
./testModel.py --model $MODELNAME_NN  --query "longitude=44.020036, latitude= 56.317545, total_square=55.5, living_square=18, kitchen_square=7.7, number_of_rooms=2, floor_number=6, number_of_floors=6, exploitation_start_year=1992" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME

echo Link: https://www.avito.ru/nizhniy_novgorod/kvartiry/1-k_kvartira_49_m_1616_et._1608356794 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo RealPrice: 4950000 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo XGBoost >> $OUTPUT_FILENAME
./testModelMachineLearning.py --model $MODELNAME_ML  --query "longitude=44.02627, latitude=56.308913, total_square=49, living_square=21, kitchen_square=11, number_of_rooms=1, floor_number=16, number_of_floors=16, exploitation_start_year=2012" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo Neuron Network >> $OUTPUT_FILENAME
./testModel.py --model $MODELNAME_NN  --query "longitude=44.02627, latitude=56.308913, total_square=49, living_square=21, kitchen_square=11, number_of_rooms=1, floor_number=16, number_of_floors=16, exploitation_start_year=2012" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME

echo Link: https://www.avito.ru/nizhniy_novgorod/kvartiry/4-k_kvartira_59.5_m_25_et._1272454785 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo RealPrice: 3150000 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo XGBoost >> $OUTPUT_FILENAME
./testModelMachineLearning.py --model $MODELNAME_ML  --query "longitude=44.037541, latitude=56.302567, total_square=59.5, living_square=43.1, kitchen_square= 6, number_of_rooms=4, floor_number=2, number_of_floors=5, exploitation_start_year=1991" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo Neuron Network >> $OUTPUT_FILENAME
./testModel.py --model $MODELNAME_NN  --query "longitude=44.037541, latitude=56.302567, total_square=59.5, living_square=43.1, kitchen_square= 6, number_of_rooms=4, floor_number=2, number_of_floors=5, exploitation_start_year=1991" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME

echo Link: https://www.avito.ru/nizhniy_novgorod/kvartiry/2-k_kvartira_64_m_117_et._836612789 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo RealPrice: 4200000 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo XGBoost >> $OUTPUT_FILENAME
./testModelMachineLearning.py --model $MODELNAME_ML  --query "longitude=44.070233, latitude=56.284499, total_square=64, living_square=32, kitchen_square=14, number_of_rooms=2, floor_number=1, number_of_floors=17, exploitation_start_year=2011" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo Neuron Network >> $OUTPUT_FILENAME
./testModel.py --model $MODELNAME_NN  --query "longitude=44.070233, latitude=56.284499, total_square=64, living_square=32, kitchen_square=14, number_of_rooms=2, floor_number=1, number_of_floors=17, exploitation_start_year=2011" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME

echo Link: https://www.avito.ru/nizhniy_novgorod/kvartiry/2-k_kvartira_61.5_m_99_et._1293035736 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo RealPrice: 4900000 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo XGBoost >> $OUTPUT_FILENAME
./testModelMachineLearning.py --model $MODELNAME_ML  --query "longitude=44.0730544, latitude=56.286023, total_square=61.5, living_square=33.9, kitchen_square=10, number_of_rooms=2, floor_number=9, number_of_floors=9, exploitation_start_year=2002" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo Neuron Network >> $OUTPUT_FILENAME
./testModel.py --model $MODELNAME_NN --query "longitude=44.0730544, latitude=56.286023, total_square=61.5, living_square=33.9, kitchen_square=10, number_of_rooms=2, floor_number=9, number_of_floors=9, exploitation_start_year=2002" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME

echo Link: https://www.avito.ru/nizhniy_novgorod/kvartiry/1-k_kvartira_46.3_m_1621_et._1669757591 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo RealPrice: 4400000 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo XGBoost >> $OUTPUT_FILENAME
./testModelMachineLearning.py --model $MODELNAME_ML  --query "longitude=44.032556, latitude=56.282717, total_square=46.3, living_square=18.4, kitchen_square=16.1, number_of_rooms=1, floor_number=16, number_of_floors=21, exploitation_start_year=2011" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo Neuron Network >> $OUTPUT_FILENAME
./testModel.py --model $MODELNAME_NN  --query "longitude=44.032556, latitude=56.282717, total_square=46.3, living_square=18.4, kitchen_square=16.1, number_of_rooms=1, floor_number=16, number_of_floors=21, exploitation_start_year=2011" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME

echo Link: https://www.avito.ru/nizhniy_novgorod/kvartiry/3-k_kvartira_60_m_19_et._981420953 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo RealPrice: 3950000 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo XGBoost >> $OUTPUT_FILENAME
./testModelMachineLearning.py --model $MODELNAME_ML  --query "longitude=44.050732, latitude=56.279253, total_square=60, living_square=40, kitchen_square=7, number_of_rooms=3, floor_number=1, number_of_floors=9, exploitation_start_year=1976" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo Neuron Network >> $OUTPUT_FILENAME
./testModel.py --model $MODELNAME_NN --query "longitude=44.050732, latitude=56.279253, total_square=60, living_square=40, kitchen_square=7, number_of_rooms=3, floor_number=1, number_of_floors=9, exploitation_start_year=1976" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME

echo Link: https://www.avito.ru/nizhniy_novgorod/kvartiry/2-k_kvartira_40_m_1617_et._1477685850 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo RealPrice: 3300000 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo XGBoost >> $OUTPUT_FILENAME
./testModelMachineLearning.py --model $MODELNAME_ML  --query "longitude=43.862366, latitude=56.217824, total_square=40, living_square=22, kitchen_square=9.3, number_of_rooms=2, floor_number=16, number_of_floors=17, exploitation_start_year=2014" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo Neuron Network >> $OUTPUT_FILENAME
./testModel.py --model $MODELNAME_NN --query "longitude=43.862366, latitude=56.217824, total_square=40, living_square=22, kitchen_square=9.3, number_of_rooms=2, floor_number=16, number_of_floors=17, exploitation_start_year=2014" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME

echo Link: https://www.avito.ru/nizhniy_novgorod/kvartiry/2-k_kvartira_36.3_m_12_et._1241062712 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo RealPrice: 1700000 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo XGBoost >> $OUTPUT_FILENAME
./testModelMachineLearning.py --model $MODELNAME_ML  --query "longitude=43.856176, latitude=56.231997, total_square=36.3, living_square=26.2, kitchen_square=4.5, number_of_rooms=2, floor_number=1, number_of_floors=2, exploitation_start_year=1969" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo Neuron Network >> $OUTPUT_FILENAME
./testModel.py --model $MODELNAME_NN  --query "longitude=43.856176, latitude=56.231997, total_square=36.3, living_square=26.2, kitchen_square=4.5, number_of_rooms=2, floor_number=1, number_of_floors=2, exploitation_start_year=1969" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME

echo Link: https://www.avito.ru/nizhniy_novgorod/kvartiry/3-k_kvartira_61.4_m_29_et._1382936975 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo RealPrice: 3900000 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo XGBoost >> $OUTPUT_FILENAME
./testModelMachineLearning.py --model $MODELNAME_ML --query "longitude=43.850486, latitude=56.259593, total_square=61.4, living_square=38.8, kitchen_square=7.6, number_of_rooms=3, floor_number=2, number_of_floors=9, exploitation_start_year=1981" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo Neuron Network >> $OUTPUT_FILENAME
./testModel.py --model $MODELNAME_NN  --query "longitude=43.850486, latitude=56.259593, total_square=61.4, living_square=38.8, kitchen_square=7.6, number_of_rooms=3, floor_number=2, number_of_floors=9, exploitation_start_year=1981" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME

echo Link: https://www.avito.ru/nizhniy_novgorod/kvartiry/2-k_kvartira_45_m_79_et._1685701443 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo RealPrice: 2300000 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo XGBoost >> $OUTPUT_FILENAME
./testModelMachineLearning.py --model $MODELNAME_ML  --query "longitude=43.874560, latitude=56.269533, total_square=45, living_square=23, kitchen_square=9, number_of_rooms=2, floor_number=7, number_of_floors=9, exploitation_start_year=1984" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo Neuron Network >> $OUTPUT_FILENAME
./testModel.py --model $MODELNAME_NN  --query "longitude=43.874560, latitude=56.269533, total_square=45, living_square=23, kitchen_square=9, number_of_rooms=2, floor_number=7, number_of_floors=9, exploitation_start_year=1984" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME

echo Link: https://www.avito.ru/nizhniy_novgorod/kvartiry/3-k_kvartira_61_m_99_et._1175215663 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo RealPrice: 4600000 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo XGBoost >> $OUTPUT_FILENAME
./testModelMachineLearning.py --model $MODELNAME_ML  --query "longitude=44.0730544, latitude=56.286023, total_square=61, living_square=42, kitchen_square=9, number_of_rooms=3, floor_number=9, number_of_floors=9, exploitation_start_year=1978" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo Neuron Network >> $OUTPUT_FILENAME
./testModel.py --model $MODELNAME_NN --query "longitude=44.0730544, latitude=56.286023, total_square=61, living_square=42, kitchen_square=9, number_of_rooms=3, floor_number=9, number_of_floors=9, exploitation_start_year=1978" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME

echo Link: https://www.avito.ru/nizhniy_novgorod/kvartiry/2-k_kvartira_39.6_m_12_et._1238778699 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo RealPrice: 1430000 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo XGBoost >> $OUTPUT_FILENAME
./testModelMachineLearning.py --model $MODELNAME_ML --query "longitude=43.906721, latitude= 56.275687, total_square=39.6, living_square=29.1, kitchen_square=6, number_of_rooms=2, floor_number=1, number_of_floors=2, exploitation_start_year=1958" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo Neuron Network >> $OUTPUT_FILENAME
./testModel.py --model $MODELNAME_NN  --query "longitude=43.906721, latitude= 56.275687, total_square=39.6, living_square=29.1, kitchen_square=6, number_of_rooms=2, floor_number=1, number_of_floors=2, exploitation_start_year=1958" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME

echo Link: https://www.avito.ru/nizhniy_novgorod/kvartiry/3-k_kvartira_60_m_39_et._1231342662 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo RealPrice: 5000000 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo XGBoost >> $OUTPUT_FILENAME
./testModelMachineLearning.py --model $MODELNAME_ML --query "longitude=43.918013, latitude=56.273438, total_square=60, living_square=48, kitchen_square=7, number_of_rooms=3, floor_number=3, number_of_floors=9, exploitation_start_year=1965" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo Neuron Network >> $OUTPUT_FILENAME
./testModel.py --model $MODELNAME_NN --query "longitude=43.918013, latitude=56.273438, total_square=60, living_square=48, kitchen_square=7, number_of_rooms=3, floor_number=3, number_of_floors=9, exploitation_start_year=1965" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME

echo Link: https://www.avito.ru/nizhniy_novgorod/kvartiry/2-k_kvartira_50_m_22_et._1370343025 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo RealPrice: 1900000 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo XGBoost >> $OUTPUT_FILENAME
./testModelMachineLearning.py --model $MODELNAME_ML  --query "longitude=43.849635, latitude=56.288726, total_square=50, living_square=35, kitchen_square=7.2, number_of_rooms=2, floor_number=2, number_of_floors=2, exploitation_start_year=1952"--limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo Neuron Network >> $OUTPUT_FILENAME
./testModel.py --model $MODELNAME_NN --query "longitude=43.849635, latitude=56.288726, total_square=50, living_square=35, kitchen_square=7.2, number_of_rooms=2, floor_number=2, number_of_floors=2, exploitation_start_year=1952" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME

echo Link: https://www.avito.ru/nizhniy_novgorod/kvartiry/1-k_kvartira_34_m_19_et._1095079539 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo RealPrice: 1990000 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo XGBoost >> $OUTPUT_FILENAME
./testModelMachineLearning.py --model $MODELNAME_ML --query "longitude=43.870959, latitude=56.295324, total_square=34, living_square=21, kitchen_square=6, number_of_rooms=1, floor_number=1, number_of_floors=9, exploitation_start_year=1980" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo Neuron Network >> $OUTPUT_FILENAME
./testModel.py --model $MODELNAME_NN --query "longitude=43.870959, latitude=56.295324, total_square=34, living_square=21, kitchen_square=6, number_of_rooms=1, floor_number=1, number_of_floors=9, exploitation_start_year=1980" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME

echo Link: https://www.avito.ru/nizhniy_novgorod/kvartiry/2-k_kvartira_46_m_59_et._954342950 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo RealPrice: 3193000 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo XGBoost >> $OUTPUT_FILENAME
./testModelMachineLearning.py --model $MODELNAME_ML  --query "longitude=43.888160, latitude=56.310765, total_square=46, living_square=28, kitchen_square=7, number_of_rooms=2, floor_number=5, number_of_floors=9, exploitation_start_year=1982" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo Neuron Network >> $OUTPUT_FILENAME
./testModel.py --model $MODELNAME_NN  --query "longitude=43.888160, latitude=56.310765, total_square=46, living_square=28, kitchen_square=7, number_of_rooms=2, floor_number=5, number_of_floors=9, exploitation_start_year=1982" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME

echo Link: https://www.avito.ru/nizhniy_novgorod/kvartiry/1-k_kvartira_31.5_m_66_et._1298960570 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo RealPrice: 1750000 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo XGBoost >> $OUTPUT_FILENAME
./testModelMachineLearning.py --model $MODELNAME_ML  --query "longitude=43.891262, latitude=56.319972, total_square=31.5, living_square=18, kitchen_square=5, number_of_rooms=1, floor_number=6, number_of_floors=6, exploitation_start_year=1967" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo Neuron Network >> $OUTPUT_FILENAME
./testModel.py --model $MODELNAME_NN  --query "longitude=43.891262, latitude=56.319972, total_square=31.5, living_square=18, kitchen_square=5, number_of_rooms=1, floor_number=6, number_of_floors=6, exploitation_start_year=1967" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME

echo Link: https://www.avito.ru/nizhniy_novgorod/kvartiry/2-k_kvartira_44_m_15_et._1550093971 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo RealPrice: 1800000 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo XGBoost >> $OUTPUT_FILENAME
./testModelMachineLearning.py --model $MODELNAME_ML --query "longitude=43.910825, latitude=56.302707, total_square=44, living_square=30, kitchen_square=6, number_of_rooms=2, floor_number=1, number_of_floors=5, exploitation_start_year=1965" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo Neuron Network >> $OUTPUT_FILENAME
./testModel.py --model $MODELNAME_NN  --query "longitude=43.910825, latitude=56.302707, total_square=44, living_square=30, kitchen_square=6, number_of_rooms=2, floor_number=1, number_of_floors=5, exploitation_start_year=1965" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME

echo Link: https://www.avito.ru/nizhniy_novgorod/kvartiry/3-k_kvartira_94_m_1019_et._1029278901 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo RealPrice: 8000000 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo XGBoost >> $OUTPUT_FILENAME
./testModelMachineLearning.py --model $MODELNAME_ML --query "longitude=43.929871, latitude=56.349316, total_square=94, living_square=76, kitchen_square=16, number_of_rooms=3, floor_number=10, number_of_floors=19, exploitation_start_year=2012" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo Neuron Network >> $OUTPUT_FILENAME
./testModel.py --model $MODELNAME_NN --query "longitude=43.929871, latitude=56.349316, total_square=94, living_square=76, kitchen_square=16, number_of_rooms=3, floor_number=10, number_of_floors=19, exploitation_start_year=2012" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME

echo Link: https://www.avito.ru/nizhniy_novgorod/kvartiry/1-k_kvartira_38_m_512_et._1498956019 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo RealPrice: 2450000 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo XGBoost >> $OUTPUT_FILENAME
./testModelMachineLearning.py --model $MODELNAME_ML --query "longitude=43.942664, latitude=56.340652, total_square=38, living_square=16, kitchen_square=7, number_of_rooms=1, floor_number=5, number_of_floors=12, exploitation_start_year=1990" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo Neuron Network >> $OUTPUT_FILENAME
./testModel.py --model $MODELNAME_NN --query "longitude=43.942664, latitude=56.340652, total_square=38, living_square=16, kitchen_square=7, number_of_rooms=1, floor_number=5, number_of_floors=12, exploitation_start_year=1990" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME

echo Link: https://www.avito.ru/nizhniy_novgorod/kvartiry/2-k_kvartira_43.7_m_89_et._1319607728 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo RealPrice: 3350000 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo XGBoost >> $OUTPUT_FILENAME
./testModelMachineLearning.py --model $MODELNAME_ML --query "longitude=43.936459, latitude=56.333290, total_square=43.7, living_square=28, kitchen_square=7.1, number_of_rooms=2, floor_number=8, number_of_floors=9, exploitation_start_year=1978" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo Neuron Network >> $OUTPUT_FILENAME
./testModel.py --model $MODELNAME_NN --query "longitude=43.936459, latitude=56.333290, total_square=43.7, living_square=28, kitchen_square=7.1, number_of_rooms=2, floor_number=8, number_of_floors=9, exploitation_start_year=1978" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME

echo Link: https://www.avito.ru/nizhniy_novgorod/kvartiry/3-k_kvartira_52.6_m_89_et._1316775654 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo RealPrice: 3070000 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo XGBoost >> $OUTPUT_FILENAME
./testModelMachineLearning.py --model $MODELNAME_ML --query "longitude=43.932077, latitude=56.324751, total_square=52.6, living_square=37, kitchen_square=7, number_of_rooms=3, floor_number=8, number_of_floors=9, exploitation_start_year=1971" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo Neuron Network >> $OUTPUT_FILENAME
./testModel.py --model $MODELNAME_NN --query "longitude=43.932077, latitude=56.324751, total_square=52.6, living_square=37, kitchen_square=7, number_of_rooms=3, floor_number=8, number_of_floors=9, exploitation_start_year=1971" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME

echo Link: https://www.avito.ru/nizhniy_novgorod/kvartiry/2-k_kvartira_44_m_25_et._962143763 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo RealPrice: 2100000 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo XGBoost >> $OUTPUT_FILENAME
./testModelMachineLearning.py --model $MODELNAME_ML --query "longitude=43.929785, latitude=56.312325, total_square=44, living_square=27, kitchen_square=6, number_of_rooms=2, floor_number=2, number_of_floors=5, exploitation_start_year=1962" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo Neuron Network >> $OUTPUT_FILENAME
./testModel.py --model $MODELNAME_NN --query "longitude=43.929785, latitude=56.312325, total_square=44, living_square=27, kitchen_square=6, number_of_rooms=2, floor_number=2, number_of_floors=5, exploitation_start_year=1962" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME

echo Link: https://www.avito.ru/nizhniy_novgorod/kvartiry/1-k_kvartira_31_m_12_et._1400660677 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo RealPrice: 1650000 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo XGBoost >> $OUTPUT_FILENAME
./testModelMachineLearning.py --model $MODELNAME_ML --query "longitude=43.937821, latitude=56.307636, total_square=31, living_square=21, kitchen_square=5.6, number_of_rooms=1, floor_number=1, number_of_floors=2, exploitation_start_year=1950" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo Neuron Network >> $OUTPUT_FILENAME
./testModel.py --model $MODELNAME_NN --query "longitude=43.937821, latitude=56.307636, total_square=31, living_square=21, kitchen_square=5.6, number_of_rooms=1, floor_number=1, number_of_floors=2, exploitation_start_year=1950" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME

echo Link: https://www.avito.ru/nizhniy_novgorod/kvartiry/3-k_kvartira_66.5_m_23_et._1455397162 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo RealPrice: 3650000 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo XGBoost >> $OUTPUT_FILENAME
./testModelMachineLearning.py --model $MODELNAME_ML  --query "longitude=43.931659, latitude=56.305208, total_square=66.5, living_square=49.8, kitchen_square=7, number_of_rooms=3, floor_number=2, number_of_floors=3, exploitation_start_year=1958" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo Neuron Network >> $OUTPUT_FILENAME
./testModel.py --model $MODELNAME_NN --query "longitude=43.931659, latitude=56.305208, total_square=66.5, living_square=49.8, kitchen_square=7, number_of_rooms=3, floor_number=2, number_of_floors=3, exploitation_start_year=1958" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME

echo Link: https://www.avito.ru/nizhniy_novgorod/kvartiry/2-k_kvartira_48.5_m_22_et._1595877271 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo RealPrice: 2200000 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo XGBoost >> $OUTPUT_FILENAME
./testModelMachineLearning.py --model $MODELNAME_ML  --query "longitude=43.927824, latitude=56.299226, total_square=48.5, living_square=33.3, kitchen_square=6.6, number_of_rooms=2, floor_number=2, number_of_floors=2, exploitation_start_year=1952" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo Neuron Network >> $OUTPUT_FILENAME
./testModel.py --model $MODELNAME_NN --query "longitude=43.927824, latitude=56.299226, total_square=48.5, living_square=33.3, kitchen_square=6.6, number_of_rooms=2, floor_number=2, number_of_floors=2, exploitation_start_year=1952" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME

echo Link: https://www.avito.ru/nizhniy_novgorod/kvartiry/4-k_kvartira_61_m_35_et._1228603055 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo RealPrice: 3600000 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo XGBoost >> $OUTPUT_FILENAME
./testModelMachineLearning.py --model $MODELNAME_ML  --query "longitude=43.918327, latitude=56.292160, total_square=61, living_square=45, kitchen_square=6, number_of_rooms=4, floor_number=3, number_of_floors=5, exploitation_start_year=1972" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo Neuron Network >> $OUTPUT_FILENAME
./testModel.py --model $MODELNAME_NN --query "longitude=43.918327, latitude=56.292160, total_square=61, living_square=45, kitchen_square=6, number_of_rooms=4, floor_number=3, number_of_floors=5, exploitation_start_year=1972" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME

echo Link: https://www.avito.ru/nizhniy_novgorod/kvartiry/1-k_kvartira_32.2_m_210_et._1434722280 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo RealPrice: 2650000 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo XGBoost >> $OUTPUT_FILENAME
./testModelMachineLearning.py --model $MODELNAME_ML --query "longitude=43.945927, latitude=56.279606, total_square=32.2, living_square=18, kitchen_square=6.8, number_of_rooms=1, floor_number=2, number_of_floors=10, exploitation_start_year=2006" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo Neuron Network >> $OUTPUT_FILENAME
./testModel.py --model $MODELNAME_NN --query "longitude=43.945927, latitude=56.279606, total_square=32.2, living_square=18, kitchen_square=6.8, number_of_rooms=1, floor_number=2, number_of_floors=10, exploitation_start_year=2006" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME

echo Link: https://www.avito.ru/nizhniy_novgorod/kvartiry/3-k_kvartira_56_m_15_et._1344435246 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo RealPrice: 3000000 $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo XGBoost >> $OUTPUT_FILENAME
./testModelMachineLearning.py --model $MODELNAME_ML   --query "longitude=43.853220, latitude=56.231541, total_square=56, living_square=42, kitchen_square=6, number_of_rooms=3, floor_number=1, number_of_floors=5, exploitation_start_year=1971" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME
echo Neuron Network >> $OUTPUT_FILENAME
./testModel.py --model $MODELNAME_NN --query "longitude=43.853220, latitude=56.231541, total_square=56, living_square=42, kitchen_square=6, number_of_rooms=3, floor_number=1, number_of_floors=5, exploitation_start_year=1971" --limits input/NizhnyNovgorodLimits.json $MODE_ --database $DATABASE_ --table $TABLE_ >> $OUTPUT_FILENAME

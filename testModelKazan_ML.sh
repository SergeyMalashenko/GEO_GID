#export MODELNAME_=best_model/modelNizhnyNovgorod_0.0001.pkl
#export MODELNAME_=modelNizhnyNovgorod_.pkl
export MODELNAME_="XGBoost/XGBoost_Kazan.pkl"
export DATABASE_='mysql://root:UWmnjxPdN5ywjEcN@188.120.245.195:3306/domprice_dev1_v2'
export TABLE_='src_ads_raw_16_processed'
export MODE_="--verbose"

#Link: https://www.cian.ru/sale/flat/170750076   $MODE_ --database $DATABASE_ --table $TABLE_
#RealPrice: 5350000 $MODE_ --database $DATABASE_ --table $TABLE_
./testModelMachineLearning.py --model $MODELNAME_ --query 'longitude=49.181801, latitude=55.790990, total_square=72, living_square=40, kitchen_square=12, number_of_rooms=2, floor_number=10, number_of_floors=11, exploitation_start_year=2009' --limits input/KazanLimits.json  $MODE_ --database $DATABASE_ --table $TABLE_
#Link: https://www.cian.ru/sale/flat/181500852 $MODE_ --database $DATABASE_ --table $TABLE_
#RealPrice: 5080000 $MODE_ --database $DATABASE_ --table $TABLE_
./testModelMachineLearning.py --model $MODELNAME_ --query 'longitude=49.130785, latitude=55.785655, total_square=50.3, living_square=36, kitchen_square=12, number_of_rooms=1, floor_number=20, number_of_floors=25, exploitation_start_year=2014' --limits input/KazanLimits.json $MODE_ --database $DATABASE_ --table $TABLE_
#Link:  https://www.cian.ru/sale/flat/190513028   $MODE_ --database $DATABASE_ --table $TABLE_
#RealPrice: 3850000 $MODE_ --database $DATABASE_ --table $TABLE_
./testModelMachineLearning.py --model $MODELNAME_ --query 'longitude=49.077857, latitude=55.829183, total_square=54.50, living_square=35.2, kitchen_square=6, number_of_rooms=3, floor_number=3, number_of_floors=5, exploitation_start_year= 1960' --limits input/KazanLimits.json $MODE_ --database $DATABASE_ --table $TABLE_
#Link:  https://www.cian.ru/sale/flat/167423018  $MODE_ --database $DATABASE_ --table $TABLE_
#RealPrice: 4499000 $MODE_ --database $DATABASE_ --table $TABLE_
./testModelMachineLearning.py --model $MODELNAME_ --query 'longitude=49.116745, latitude=55.779788, total_square=55.30, living_square=33, kitchen_square=8.1, number_of_rooms=2, floor_number=3, number_of_floors=4, exploitation_start_year= 1917' --limits input/KazanLimits.json $MODE_ --database $DATABASE_ --table $TABLE_
#Link:  https://www.cian.ru/sale/flat/193616668 $MODE_ --database $DATABASE_ --table $TABLE_
#RealPrice: 9000000 $MODE_ --database $DATABASE_ --table $TABLE_
./testModelMachineLearning.py --model $MODELNAME_ --query 'longitude=49.065774, latitude=55.839577, total_square=127, living_square=66, kitchen_square=21.8, number_of_rooms=3, floor_number=3, number_of_floors=5, exploitation_start_year=2002' --limits input/KazanLimits.json $MODE_ --database $DATABASE_ --table $TABLE_
#Link:  https://www.cian.ru/sale/flat/183518699   $MODE_ --database $DATABASE_ --table $TABLE_
#RealPrice: 2700000 $MODE_ --database $DATABASE_ --table $TABLE_
./testModelMachineLearning.py --model $MODELNAME_ --query 'longitude=49.093146, latitude=55.860605, total_square=45.0, living_square=22, kitchen_square=10, number_of_rooms=1, floor_number=1, number_of_floors=10, exploitation_start_year=1999' --limits input/KazanLimits.json $MODE_ --database $DATABASE_ --table $TABLE_
#Link: https://www.cian.ru/sale/flat/199214936  $MODE_ --database $DATABASE_ --table $TABLE_
#RealPrice: 6249000 $MODE_ --database $DATABASE_ --table $TABLE_
./testModelMachineLearning.py --model $MODELNAME_ --query 'longitude=49.084235, latitude=55.816379, total_square=66.8, living_square=33.2, kitchen_square=12, number_of_rooms=2, floor_number=8, number_of_floors=16, exploitation_start_year=2011' --limits input/KazanLimits.json $MODE_ --database $DATABASE_ --table $TABLE_
#Link:  https://www.cian.ru/sale/flat/203984109    $MODE_ --database $DATABASE_ --table $TABLE_
#RealPrice: 2650000 $MODE_ --database $DATABASE_ --table $TABLE_
./testModelMachineLearning.py --model $MODELNAME_ --query 'longitude=49.228100, latitude=55.780618, total_square=31.7, living_square=13, kitchen_square=8, number_of_rooms=1, floor_number=5, number_of_floors=9, exploitation_start_year=1997' --limits input/KazanLimits.json $MODE_ --database $DATABASE_ --table $TABLE_
#Link: https://www.cian.ru/sale/flat/204473339  $MODE_ --database $DATABASE_ --table $TABLE_
#RealPrice: 2750000 $MODE_ --database $DATABASE_ --table $TABLE_
./testModelMachineLearning.py --model $MODELNAME_ --query 'longitude=48.896792, latitude=55.820359, total_square=50.9, living_square=28.8, kitchen_square=6.6, number_of_rooms=2, floor_number=3, number_of_floors=3, exploitation_start_year=1959' --limits input/KazanLimits.json $MODE_ --database $DATABASE_ --table $TABLE_
#Link: https://www.cian.ru/sale/flat/201907145  $MODE_ --database $DATABASE_ --table $TABLE_
#RealPrice: 3900000 $MODE_ --database $DATABASE_ --table $TABLE_
./testModelMachineLearning.py --model $MODELNAME_ --query 'longitude=49.114427, latitude=55.786187, total_square=40.00, living_square=20, kitchen_square=17, number_of_rooms=1, floor_number=1, number_of_floors=5, exploitation_start_year=1997' --limits input/KazanLimits.json $MODE_ --database $DATABASE_ --table $TABLE_
#Link: https://www.cian.ru/sale/flat/200282568  $MODE_ --database $DATABASE_ --table $TABLE_
#RealPrice: 6520000 $MODE_ --database $DATABASE_ --table $TABLE_
./testModelMachineLearning.py --model $MODELNAME_ --query 'longitude=49.148788, latitude=55.824956, total_square=72, living_square=43, kitchen_square=10.8, number_of_rooms=2, floor_number=2, number_of_floors=16, exploitation_start_year=2012' --limits input/KazanLimits.json $MODE_ --database $DATABASE_ --table $TABLE_
#Link: https://www.cian.ru/sale/flat/200418848/ $MODE_ --database $DATABASE_ --table $TABLE_
#RealPrice: 4100000 $MODE_ --database $DATABASE_ --table $TABLE_
./testModelMachineLearning.py --model $MODELNAME_ --query 'longitude=49.092463, latitude=55.855002, total_square=60, living_square=38, kitchen_square=6, number_of_rooms=3, floor_number=2, number_of_floors=5, exploitation_start_year=1961' --limits input/KazanLimits.json $MODE_ --database $DATABASE_ --table $TABLE_
#Link: https://www.cian.ru/sale/flat/201053285    $MODE_ --database $DATABASE_ --table $TABLE_
#RealPrice: 4050000 $MODE_ --database $DATABASE_ --table $TABLE_
./testModelMachineLearning.py --model $MODELNAME_ --query 'longitude=49.246030, latitude=55.753630, total_square=43, living_square=20, kitchen_square=13, number_of_rooms=1, floor_number=5, number_of_floors=10, exploitation_start_year=2013' --limits input/KazanLimits.json $MODE_ --database $DATABASE_ --table $TABLE_
#Link: https://www.cian.ru/sale/flat/202848916    $MODE_ --database $DATABASE_ --table $TABLE_
#RealPrice: 7000000 $MODE_ --database $DATABASE_ --table $TABLE_
./testModelMachineLearning.py --model $MODELNAME_ --query 'longitude=49.082034, latitude=55.817967, total_square=81.7, living_square=43.8, kitchen_square=12.7, number_of_rooms=3, floor_number=6, number_of_floors=16, exploitation_start_year=2009' --limits input/KazanLimits.json $MODE_ --database $DATABASE_ --table $TABLE_
#Link: https://www.cian.ru/sale/flat/199809896  $MODE_ --database $DATABASE_ --table $TABLE_
#RealPrice: 7000000 $MODE_ --database $DATABASE_ --table $TABLE_
./testModelMachineLearning.py --model $MODELNAME_ --query 'longitude=49.169548, latitude=55.790555, total_square=104, living_square=67, kitchen_square=15.5, number_of_rooms=3, floor_number=5, number_of_floors=9, exploitation_start_year=2004' --limits input/KazanLimits.json $MODE_ --database $DATABASE_ --table $TABLE_
#Link:  https://www.cian.ru/sale/flat/196935901 $MODE_ --database $DATABASE_ --table $TABLE_
#RealPrice: 12000000 $MODE_ --database $DATABASE_ --table $TABLE_
./testModelMachineLearning.py --model $MODELNAME_ --query 'longitude=49.111355, latitude=55.790808, total_square=122.8, living_square=80, kitchen_square=14.5, number_of_rooms=3, floor_number=5, number_of_floors=5, exploitation_start_year=1939' --limits input/KazanLimits.json $MODE_ --database $DATABASE_ --table $TABLE_
#Link: https://www.cian.ru/sale/flat/200409755 $MODE_ --database $DATABASE_ --table $TABLE_
#RealPrice: 3850000 $MODE_ --database $DATABASE_ --table $TABLE_
./testModelMachineLearning.py --model $MODELNAME_ --query 'longitude=49.076060, latitude=55.853435, total_square=69, living_square=45, kitchen_square=8, number_of_rooms=3, floor_number=3, number_of_floors=5, exploitation_start_year=1937' --limits input/KazanLimits.json $MODE_ --database $DATABASE_ --table $TABLE_
#Link: https://www.cian.ru/sale/flat/203156210  $MODE_ --database $DATABASE_ --table $TABLE_
#RealPrice: 2300000 $MODE_ --database $DATABASE_ --table $TABLE_
./testModelMachineLearning.py --model $MODELNAME_ --query 'longitude=49.085699, latitude=55.834972, total_square=33.6, living_square=22.6, kitchen_square=9.6, number_of_rooms=2, floor_number=2, number_of_floors=3, exploitation_start_year=1953' --limits input/KazanLimits.json $MODE_ --database $DATABASE_ --table $TABLE_
#Link: https://www.cian.ru/sale/flat/211543909   $MODE_ --database $DATABASE_ --table $TABLE_
#RealPrice: 2290000 $MODE_ --database $DATABASE_ --table $TABLE_
./testModelMachineLearning.py --model $MODELNAME_ --query 'longitude=49.077165, latitude=55.855457, total_square=35.5, living_square=17.5, kitchen_square=10, number_of_rooms=1, floor_number=2, number_of_floors=2, exploitation_start_year=1948' --limits input/KazanLimits.json $MODE_ --database $DATABASE_ --table $TABLE_
#Link: https://www.cian.ru/sale/flat/211548756    $MODE_ --database $DATABASE_ --table $TABLE_
#RealPrice: 3750000 $MODE_ --database $DATABASE_ --table $TABLE_
./testModelMachineLearning.py --model $MODELNAME_ --query 'longitude=49.076060, latitude=55.853435, total_square=80.2, living_square=57.7, kitchen_square=7, number_of_rooms=4, floor_number=6, number_of_floors=6, exploitation_start_year=1937' --limits input/KazanLimits.json $MODE_ --database $DATABASE_ --table $TABLE_

#export DATABASE_='mysql://sr:A4y8J6r4@149.154.71.73:3310/sr_dev' 
#export TABLE_='nn_neiro'
export DATABASE_='mysql://root:Intemp200784@127.0.0.1/smartRealtor?unix_socket=/var/run/mysqld/mysqld.sock' 
export TABLE_='real_estate_from_ads_api' 
export MODELNAME_=modelNizhnyNovgorod.pkl 
#export MODELNAME_=best_model/modelNizhnyNovgorod.pkl 
export MODE_="--verbose"
#export MODE_=""

set -x
#http://www.gipernn.ru/prodazha-kvartir/1-komnatnaya-sh-kazanskoe-d-23-id2503730
./testModel.py --model $MODELNAME_ --query 'longitude=44.0730544, latitude=56.286023, total_square=35, living_square=18, kitchen_square=9, number_of_rooms=1, floor_number=2, number_of_floors=10, exploitation_start_year=1991' --limits input/NizhnyNovgorodLimits.json
#http://www.gipernn.ru/prodazha-kvartir/2-komnatnaya-sh-kazanskoe-d-1-id2202161
./testModel.py --model $MODELNAME_ --query 'longitude=44.07629, latitude=56.29836, total_square=68, living_square=32, kitchen_square=10, number_of_rooms=2, floor_number=9, number_of_floors=16, exploitation_start_year=1999' --limits input/NizhnyNovgorodLimits.json
#http://www.gipernn.ru/prodazha-kvartir/2-komnatnaya-sh-yuzhnoe-d-50-id2409699
./testModel.py --model $MODELNAME_ --query 'longitude=43.8714109, latitude=56.22936, total_square=42.4, living_square=27.9, kitchen_square=5.4, number_of_rooms=2, floor_number=1, number_of_floors=5, exploitation_start_year=1965' --limits input/NizhnyNovgorodLimits.json
#http://www.gipernn.ru/prodazha-kvartir/1-komnatnaya-sh-yuzhnoe-d-7-id2514505
./testModel.py --model $MODELNAME_ --query 'longitude=43.8559724, latitude=56.2251936, total_square=26, living_square=15, kitchen_square=5, number_of_rooms=1, floor_number=2, number_of_floors=2, exploitation_start_year=1958' --limits input/NizhnyNovgorodLimits.json

./testModel.py --model $MODELNAME_ --query 'longitude=44.076181, latitude=56.296589, total_square=33.0, living_square=28.0, kitchen_square=8.6, number_of_rooms=1, floor_number=1, number_of_floors=9, exploitation_start_year=1983' --limits input/NizhnyNovgorodLimits.json

./testModel.py --model $MODELNAME_ --query 'longitude=44.072040, latitude=56.289260, total_square=56.0, living_square=33.0, kitchen_square=15.0, number_of_rooms=2, floor_number=7, number_of_floors=9, exploitation_start_year=2007' --limits input/NizhnyNovgorodLimits.json

./testModel.py --model $MODELNAME_ --query 'longitude=44.000999, latitude=56.310582 , total_square=100.0, living_square=57.0, kitchen_square=18.0, number_of_rooms=3, floor_number=9, number_of_floors=14, exploitation_start_year=2010' --limits input/NizhnyNovgorodLimits.json

./testModel.py --model $MODELNAME_ --query 'longitude=44.002034, latitude=56.311701, total_square=91.1, living_square=45.1, kitchen_square=20.5, number_of_rooms=2, floor_number=2, number_of_floors=9, exploitation_start_year=2003' --limits input/NizhnyNovgorodLimits.json

./testModel.py --model $MODELNAME_ --limits input/NizhnyNovgorodLimits.json --query 'longitude=44.0151902, latitude=56.2638371, total_square=59.3, living_square=40, kitchen_square=5, number_of_rooms=3, floor_number=4, number_of_floors=17, exploitation_start_year=2015' $MODE_

./testModel.py --model $MODELNAME_ --limits input/NizhnyNovgorodLimits.json --query 'longitude=43.9620856, latitude=56.3315441, total_square=49.1, living_square=35, kitchen_square=10, number_of_rooms=2, floor_number=7, number_of_floors=9, exploitation_start_year=1975' $MODE_

#https://www.gipernn.ru/prodazha-kvartir/3-komnatnaya-ul-belinskogo-d-15-id2561887
./testModel.py --model $MODELNAME_ --limits input/NizhnyNovgorodLimits.json --query 'longitude=43.9989883, latitude=56.310686, total_square=104.6, living_square=60.8, kitchen_square=15.1, number_of_rooms=3, floor_number=3, number_of_floors=14, exploitation_start_year=2010' $MODE_

set +x

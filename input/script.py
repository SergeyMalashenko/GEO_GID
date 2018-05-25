import pandas as pd
import numpy as  np

input_fileName  = 'NizhnyNovgorod.csv'
output_fileName = 'NizhnyNovgorod_test.csv'

dataFrame = pd.read_csv(
	input_fileName, 
	sep=";",
	encoding='cp1251', 
	verbose=True, 
	keep_default_na=False
).dropna(how="all")

dataFrame.drop( ['price'], axis=1, inplace=True )

dataFrame.to_csv(
	output_fileName,
	sep=";",
	encoding='cp1251',
	index=False 
)

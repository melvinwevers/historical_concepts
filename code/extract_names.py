from bs4 import BeautifulSoup as bs
import pandas as pd
from os import listdir
from os.path import isfile, join

'''
Script to extract names from raw html files from Voornamenbank
and place them in a CSV
'''

mypath = 'raw/'  # path with raw html files
files_ = [f for f in listdir(mypath) if isfile(join(mypath, f))]

male_names = []
female_names = []
for file in files_:
    html = bs(open(mypath + file))
    for row in html.select('table#topnamen-jongens tr'):
        cells = row.find_all('td')
        if not cells:
            continue
        male_names.append([cell.get_text(strip=True) for cell in cells][1])
    for row in html.select('table#topnamen-meisjes tr'):
        cells = row.find_all('td')
        if not cells:
            continue
        female_names.append([cell.get_text(strip=True) for cell in cells][1])
male_names = set(male_names)
female_names = set(female_names)

names_df = pd.DataFrame(list(zip(male_names, female_names)),
                        columns=['male_names', 'female_names'])
names_df.to_csv('names.csv', index=False)

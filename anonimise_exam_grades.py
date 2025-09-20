import pandas as pd
import glob

# specify the directory where the Excel files are located
directory = 'data/'

# use glob to find all the Excel files in the directory
excel_files = glob.glob(directory + '/*.xlsx')
excel_files_2 = glob.glob(directory + '/*.xls')

# initialize an empty DataFrame to store the concatenated data
dataframes = []

# loop through each Excel file and append it to the DataFrame
for file in excel_files:
    # read excel only from row 21
    curent_dataframe = pd.read_excel(file, skiprows=20)
    
    dataframes.append(curent_dataframe)

for file in excel_files_2:
    # read excel only from row 21
    curent_dataframe = pd.read_excel(file, skiprows=20)
    
    dataframes.append(curent_dataframe)


# concatenate all the dataframes into a single DataFrame
df = pd.concat(dataframes, ignore_index=True)

# drop any duplicate rows that may have been created by the append operation
df = df.drop_duplicates()

# print the resulting DataFrame
print(df)

# get rid of the following columns:
columns_to_delete = ['#', 'Unnamed: 1', 'Cognome', 'Nome', 'CFU',
       'Unnamed: 7', 'Svolgimento Esame', 'Domande d\'esame',
       'Data superamento', 'Nota per lo studente', 'Presa Visione', 'CDS COD.',
       'AD COD.', 'Email']

df = df.drop(columns=columns_to_delete, axis=1)

# lets anonnymize "MAtricola" assigning a mapping
mapping = {}
for i, row in enumerate(df['Matricola'].unique()):
    mapping[row] = i

df['Matricola'] = df['Matricola'].map(mapping)


# save as psi_grades.csv
df.to_csv('data/psi_grades.csv', index=False)
#%% 
import pandas as pd
import os
from spacy_language_detection import LanguageDetector
import spacy
from spacy.language import Language
from bs4 import BeautifulSoup
#%%
# 
data_source = 'diva-portal'
#data_source = 'gupea'

#%%
if data_source == 'diva-portal':
    data_path = './data/get_csv/'
    files = os.listdir(data_path)
    data = []
    for file in files:
        df = pd.read_csv(data_path + file)
        data.append(df)
        
        
    data = pd.concat(data)

    # Remove any duplicate PID values
    data = data.drop_duplicates(subset='PID')
    print(len(data))  # nr of hits

    data.dropna(subset=['Abstract'], inplace=True)
    data = data[data['Language'] == 'swe']
    data.reset_index(drop=True, inplace=True)
    data['Abstract_auto'] = None
    save_str = 'dp_data'
elif data_source == 'gupea':
    data = pd.read_csv('./data/gupea_abstracts.csv')
    data.dropna(subset=['abstract'], inplace=True)
    data.reset_index(drop=True, inplace=True)
    data.rename(columns={'abstract': 'Abstract'}, inplace=True)
    data['Abstract_auto'] = None
    save_str = 'gupea_data'
#%%

def get_lang_detector(nlp, name):
    return LanguageDetector(seed=42)  # We use the seed 42

nlp_model = spacy.load("en_core_web_sm")
Language.factory("language_detector", func=get_lang_detector)
nlp_model.add_pipe('language_detector', last=True)

#%%
abstract_auto = {}
abstract_auto_lang = {}
abstract_auto_lang_score = {}

#%%
start_i = 0

# First route, is to just add language and probability of which language
for i, d in data.iterrows():
    # Check if Abstract_auto already exists in d
    if i in abstract_auto:
        continue

    if i < start_i:
        continue
    print(i)
    bs = BeautifulSoup(d['Abstract'], 'html.parser')
    l = 0
    doc = nlp_model(bs.text)
    print(doc._.language)
    abstract_auto_text = bs.text    
    abstract_auto[i] = abstract_auto_text
    abstract_auto_lang[i] = doc._.language['language']
    abstract_auto_lang_score[i] = doc._.language['score']
    # save every 1000 rows
    if i % 1000 == 0:
        data['Abstract_auto'] = data.index.map(abstract_auto)
        data['Abstract_auto_lang'] = data.index.map(abstract_auto_lang)
        data['Abstract_auto_lang_score'] = data.index.map(abstract_auto_lang_score)
        data.to_csv(f'./data/{save_str}.csv')

#%%
data['Abstract_auto'] = data.index.map(abstract_auto)
data['Abstract_auto_lang'] = data.index.map(abstract_auto_lang)
data['Abstract_auto_lang_score'] = data.index.map(abstract_auto_lang_score)

#data['Abstract_sv'] = abstract_swe

# %%
data.to_csv(f'./data/{save_str}_step-autolang.csv')

#%%
data = pd.read_csv(f'./data/{save_str}_step-autolang.csv') 

#%%
# Remove rows in english that are 99% sure
data = data[~((data['Abstract_auto_lang'] == 'en') & (data['Abstract_auto_lang_score'] > 0.99))]

data_sure = data[(data['Abstract_auto_lang'] == 'sv') & (data['Abstract_auto_lang_score'] > 0.99)]
data_unsure = data[~((data['Abstract_auto_lang'] == 'sv') & (data['Abstract_auto_lang_score'] > 0.99))]

#%%
# Go through data_unsure and find the sections in swedish
abstract_auto_unsure = {}
for i, d in data_unsure.iterrows():
    if i in abstract_auto_unsure:
        continue
    print(i)
    abstract_auto_unsure[i] = []
    bs = BeautifulSoup(d['Abstract'], 'html.parser')
    doc = nlp_model(bs.text)
    for sent in doc.sents:
        if sent._.language['language'] == 'sv':
            abstract_auto_unsure[i].append(sent.text)
    abstract_auto_unsure[i] = ' '.join(abstract_auto_unsure[i])
    
    
# %%
data_sure['abstract_preproc'] = data_sure['Abstract_auto']
data_unsure['abstract_preproc'] = data_unsure.index.map(abstract_auto_unsure)

#%% combine data
data = pd.concat([data_sure, data_unsure])
data['Abstract'] = data['Abstract'].str.replace('\n', ' ')
data.to_csv(f'./data/{save_str}_step-autolang_s2.csv')


# %%

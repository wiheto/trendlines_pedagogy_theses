#%%
import pandas as pd
import re
from tqdm import tqdm


#%%
df = pd.read_csv('./data/data_step-autolang_s2.csv', index_col=[0])
df.dropna(subset=['abstract_preproc'], inplace=True)
df.drop(columns=['Abstract_auto_lang', 'Abstract_auto', 'Abstract', 'abstract_preproc'], inplace=True)

#%%
df.columns
# %%
for n in df.Name:
    print(n)
# %%
# We want to grab all text between (...), there can be multiple, and only keep them
row = df.loc[11]
re.findall(r'\((.*?)\)', row['Name'])
# %%
# We want to grab all text between "[0-9]" (including the square brackets), there are multiple, and get all text prior to that string back until punctuation of ;, , or ( 
codes_store = []
str_store = []
#bad_codes = ['[1991]', '[409174]']
for i, row in tqdm(df.iterrows()):
    # First check if there is a semicolon. Spit on that. 
    # Then check if there is a [ before a (. If so, remove up until the ( 
    # Then recombine the string
    name = row['Name']
    name = name.split(';')
    # Find index of first ( and [
    new_name = []
    for n in name: 
        if '(' in n and '[' in n:
            bracket_i = n.index('(')
            square_i = n.index('[')
            if bracket_i > square_i:
                n = n[bracket_i-1:]
        new_name.append(n)
    name = ';'.join(new_name)    
    codes = re.findall(r'\[[0-9]*?\]', name)
    # and for each of these codes get all text prior to that string back until punctuation of ;, , or ( 
    after_code = name
    for code in codes:
        split_text = after_code.split(code)
        before_code = split_text[0]
        if ';' in before_code:
            before_code = before_code.split(';')[1]
        after_code = code.join(split_text[1:])
        exceptions = ['-)', '-2013)', 'CeHum)', 'UU Innovation)', 'UU Samverkan)']
        exceptions_past = ['Anjelica) ']
        if '(' in before_code:
            # Check if before_code starts with any of the exceptions
            skip = 0
            # If before_code is [A-Z]*) then is an exception
            if before_code.split('(')[1].isupper() and before_code.split('(')[1].strip().endswith(')'):
                skip = 1
            for e in exceptions: 
                if before_code.split('(')[1].startswith(e):
                    skip = 1
            for e in exceptions_past:
                if  before_code.split('(')[1] == e:
                    before_code = before_code.split('(')[2]
                    skip = 1
            if skip == 0:
                before_code = before_code.split('(')[1]
        # Remove trailing whitespace
        before_code = before_code.replace(', ', '')
        before_code = before_code.strip()
        code = code.replace('[', '').replace(']', '')
        code = int(code)
        codes_store.append(code)
        str_store.append(before_code)
    

#%%
df_uni = pd.DataFrame(data={'Uni': sorted(list(set(str_store)))})
df_uni.to_csv('./data/affiliations.csv')
# %%

# %%
# Get numbers per affiliation 
affiliation = pd.read_csv('./data/affiliations.csv', index_col=[0])
affiliation = affiliation[affiliation['Keep'] == 1]
# %%
# Loop through afifliations and count in df
aff_df = {}
for i, row in affiliation.iterrows():
    aff_df[row['Uni']] = df['Name'].str.contains(row['Uni'], case=False)
    
aff_df = pd.DataFrame(aff_df)
# %%
aff_df.sum()
aff_df.sum().to_csv('./data/affiliation_numbers.csv')
# %%
aff_df.transpose().sum().value_counts()
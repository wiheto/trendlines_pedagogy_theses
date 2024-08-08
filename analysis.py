#%%
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pingouin as pg 
import numpy as np
import json

#%%
## General funcitons
def load_data():
    # diva-portal data
    df = pd.read_csv('./data/dp_data_step-autolang_s2.csv')
    df = df.dropna(subset=['abstract_preproc'])
    # gupea data
    df_gupea = pd.read_csv('./data/gupea_data_step-autolang_s2.csv', index_col=[0])
    df_gupea['affiliation'] = 'University of Gothenburg' 
    # get same column names
    df['abstract'] = df['abstract_preproc']
    df['year'] = df['Year']
    df['title'] = df['Title']   
    df['keywords'] = df['Keywords']
    df['affiliation'] = df['Name']
    df_gupea['abstract'] = df_gupea['abstract_preproc']
    # Combine df and df gupea in abstract year title and keywords only
    df = df[['abstract', 'year', 'title', 'keywords', 'affiliation']]
    df_gupea = df_gupea[['abstract', 'year', 'title', 'keywords', 'affiliation']]
    df = pd.concat([df, df_gupea])
    # Only take after year 2005 
    # (OBS there is one 13 here)
    df = df[df['year'] >= 2005]
    df = df[df['year'] < 2024]
    # make all lowercase
    df['abstract'] = df['abstract'].str.lower()
    df['title'] = df['title'].str.lower()
    df['keywords'] = df['keywords'].str.lower().replace(np.nan, '')
    df['affiliation'] = df['affiliation'].str.lower()
    df['text'] = df['abstract'] + ' ' + df['title'] + ' ' + df['keywords']
    df.reset_index(inplace=True)
    return df


def load_match_affiliations(affiliation_path, df, keywords, sortby, multiple_hit_policy=None, mcolname=None, deprio_col=None, restrict_hist=True):
    affiliations = pd.read_csv(affiliation_path, index_col=[0])
    affiliations = affiliations[affiliations['Keep'] == 1]
    affiliations['Uni'] = affiliations['Uni'].str.lower()
    # Add one row that is university of gothenburg
    affiliations = affiliations.append({'Uni': 'university of gothenburg'}, ignore_index=True)
    duplicate_terms = ['högskolan i jönköping', 'lärarhögskolan i stockholm','växjö universitet', 'högskolan i kalmar',  'malmö högskola', 'mälardalens högskola', 'lärarhögskolan vid umeå universitet (LH)', 'teologiska högskolan stockholm', 'linné universitetet']
    replace_terms = ['jönköping university', 'stockholms universitet','linné universitetet', 'linné universitetet', 'malmö universitet', 'mälardalens universitet', 'umeå universitet', 'enskilda högskolan stockholm', 'linnéuniversitetet']
    for i, dt in enumerate(duplicate_terms):
        df['affiliation'] = df['affiliation'].str.lower().str.replace(dt, replace_terms[i], case=False)
    for keyword in keywords:
        affiliations[keyword] = 0
    if multiple_hit_policy:
        affiliations[mcolname] = 0
        
    affiliations['total'] = 0

    for af in affiliations['Uni']:
        af_hit = 0
        tdf = df[df['affiliation'].str.lower().str.contains(af, case=False)]
        if isinstance(keywords, list):   
            key_store = {}    
            for keyword in keywords:
                # At the row where Uni = af, update
                key_store[keyword] = tdf['text'].str.contains(keyword, case=False)
            key_store = pd.DataFrame(key_store)
            if multiple_hit_policy:
                key_store = apply_multiple_hit_policy(key_store, keywords, multiple_hit_policy, mcolname, deprio_col)
                newkeys = keywords + [mcolname]
            else: 
                newkeys = keywords
            for keyword in newkeys:
                key_per_af = key_store[keyword].sum()
                affiliations.loc[affiliations['Uni'] == af, keyword] += key_per_af
                af_hit += key_per_af
            affiliations.loc[affiliations['Uni'] == af, 'Not mentioned'] = sum(key_store.sum(axis=1)==0)      

        elif isinstance(keywords, dict):
            key_store = {}
            for keyword in keywords:
                key_store[keyword] = np.zeros(len(tdf))
                for kw in keywords[keyword]:
                    key_store[keyword] += tdf['text'].str.contains(kw, case=False)
                # Make sure only one hit per keyword
                key_store[keyword][key_store[keyword]>1] = 1
            key_store = pd.DataFrame(key_store)
            if multiple_hit_policy:
                key_store = apply_multiple_hit_policy(key_store, list(keywords.keys()), multiple_hit_policy, mcolname, deprio_col)
                if mcolname is not None:
                    newkeys = list(keywords.keys()) + [mcolname]
                else: 
                    newkeys = list(keywords.keys())
            else: 
                newkeys = list(keywords.keys())
            for keyword in newkeys:
                key_per_af = key_store[keyword].sum()
                af_hit += key_per_af
                affiliations.loc[affiliations['Uni'] == af, keyword] += key_per_af
            affiliations.loc[affiliations['Uni'] == af, 'Not mentioned'] = sum(key_store.sum(axis=1)==0)      
        print(af)
        print(af_hit)
    
    total = np.zeros(len(affiliations))
    for key in newkeys:
        total += affiliations[key]
    total += affiliations['Not mentioned']
    affiliations['total'] = total
    
    mentioned_sum = np.zeros(len(affiliations))
    for keyword in newkeys:
        mentioned_sum += affiliations[keyword]
        affiliations[keyword] = affiliations[keyword] / affiliations['total']
        
    affiliations['Not mentioned'] = affiliations['Not mentioned'] / affiliations['total']   
    # Only take those with more than 50 hits
    if restrict_hist == True:
        affiliations = affiliations[affiliations['total']>50]
    affiliations.sort_values(sortby, inplace=True, ascending=False)
    # Hard code the english names
    # The dictionary of universities
    universities_dict = {
        'högskolan i borås': 'University of Borås',
        'uppsala universitet': 'Uppsala University',
        'karlstads universitet': 'Karlstad University',
        'linköpings universitet': 'Linköping University',
        'university of gothenburg': 'University of Gothenburg',
        'högskolan väst': 'University West',
        'högskolan kristianstad': 'Kristianstad University',
        'mälardalens universitet': 'Mälardalen University',
        'örebro universitet': 'Örebro University',
        'umeå universitet': 'Umeå University',
        'linnéuniversitetet': 'Linnaeus University',
        'högskolan i halmstad': 'Halmstad University',
        'mittuniversitetet': 'Mid Sweden University',
        'luleå tekniska universitet': 'Luleå University of Technology',
        'stockholms universitet': 'Stockholm University',
        'högskolan i gävle': 'University of Gävle',
        'kungl. musikhögskolan': 'Royal College of Music in Stockholm',
        'jönköping university': 'Jönköping University',
        'högskolan dalarna': 'Dalarna University',
        'malmö universitet': 'Malmö University',
        'kth': 'KTH Royal Institute of Technology',
        'högskolan i skövde': 'University of Skövde',
        'konstfack': 'University of Arts, Crafts and Design',
        'södertörns högskola': 'Södertörn University',
        'blekinge tekniska högskola': 'Blekinge Institute of Technology',
        'högskolan på gotland': 'Campus Gotland',
        'försvarshögskolan': 'Swedish Defence University',
        'stockholms konstnärliga högskola': 'Stockholm University of the Arts',
        'enskilda högskolan stockholm': 'University College Stockholm'
    }
    # Map the English names to the 'uni' column and create a new 'uni_eng' column
    affiliations['Uni_eng'] = affiliations['Uni'].map(universities_dict)
    affiliations.reset_index(inplace=True, drop=True)
    return affiliations

def apply_multiple_hit_policy(keyword_df, keywords, policy, columnname, deprio_col=None):
    if policy == 'multiple_column':
        mcol = keyword_df[keywords].sum(axis=1)
        mcol[mcol <= 1] = 0 
        mcol[mcol > 1] = 1
        keyword_df[columnname] = mcol
        # Set all columns to 0 where multiple hit policy is 1
        for kw in keywords:
            keyword_df[kw][keyword_df[columnname] == 1] = 0
        mdf = keyword_df
    elif policy == 'heuristic_takemax_w_deprioritize':
        mdf = keyword_df.copy()
        mdf[deprio_col][mdf[deprio_col]>0] = 0.5
        mdf['None mentioned'] = 0
        mdf['None mentioned'][mdf.sum(axis=1) == 0] = 1
        if isinstance(keywords, dict):
            nk = list(keywords.keys()) + ['None mentioned']
        else:
            nk = keywords + ['None mentioned']
        mdf['max'] = mdf[nk].idxmax(axis=1)
        for kw in nk:
            mdf[kw] = 0
        for i, row in mdf.iterrows():
            mdf[row['max']][i] = 1
        mdf.drop('max', axis=1, inplace=True)
    elif policy == 'return_jaccard':
        # Return a len(keyword) * len(keyword) where each cell is the number of times two keywords are mentioned together
        overlap = np.zeros((len(keywords), len(keywords)))
        union = np.zeros((len(keywords), len(keywords)))
        for i, key_i in enumerate(keywords):
            for j, key_j in enumerate(keywords):
                if i == j:
                    continue
                overlap[i, j] += np.sum(keyword_df[key_i] * keyword_df[key_j])
                union[i, j] += np.sum(np.max([keyword_df[key_i], keyword_df[key_j]], axis=0))
        mdf = pd.DataFrame(data=overlap/union, columns=keywords, index=keywords)

    elif policy == 'return_SzymkSimp':
        # Return a len(keyword) * len(keyword) where each cell is the number of times two keywords are mentioned together
        overlap = np.zeros((len(keywords), len(keywords)))
        smallest_set = np.zeros((len(keywords), len(keywords)))
        for i, key_i in enumerate(keywords):
            for j, key_j in enumerate(keywords):
                if i == j:
                    continue
                overlap[i, j] += np.sum(keyword_df[key_i] * keyword_df[key_j])
                smallest_set[i, j] = np.min([keyword_df[key_i].sum(), keyword_df[key_j].sum()])
        mdf = pd.DataFrame(data=overlap/smallest_set, columns=keywords, index=keywords)
    else:
        mdf = keyword_df
    return mdf

def match_keywords(df, keywords, multiple_hit_policy=None, mcolname='Multiple', deprio_col=None):
    key_match = {}
    if isinstance(keywords, list):
        for keyword in keywords:
            key_match[keyword] = df['text'].str.lower().str.contains(keyword, case=False)
        kdf = pd.DataFrame(key_match)
    elif isinstance(keywords, dict):
        kdf = pd.DataFrame(index=df.index)
        for kws in keywords:
            kdf[kws] = np.zeros(len(df))
            for kw in keywords[kws]:
                kdf[kws] += df['text'].str.lower().str.count(kw)
    # Make max 1 for all policies but heuristic_takemax_w_deprioritize
    if multiple_hit_policy != 'heuristic_takemax_w_deprioritize':
        kdf[kdf>1] = 1
    # Multiple hit policy
    kdf = apply_multiple_hit_policy(kdf, keywords, multiple_hit_policy, mcolname, deprio_col)
    # Add year
    kdf['year'] = df['year']
    # make keyword numeric instead of bool
    for keyword in keywords:
        kdf[keyword] = kdf[keyword].astype(int)
    return kdf

# %%

def plot_keywords(kdf, df, keywords, colors, sname, return_trends=False, legend_offset=(0, 0), y1_tickspacing=0.1):

    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    count_gdf = kdf.groupby('year').sum()
    mean_gdf = kdf.groupby('year').sum()
    for keyword in keywords:
        # Have to divide by len of grouped df because of multiple hits possible
        mean_gdf[keyword] = mean_gdf[keyword] / df.groupby('year').count()['abstract'].values
    for i, keyword in enumerate(keywords):
        ax[0].plot(count_gdf.index, count_gdf[keyword], label=keyword, color=colors[i])
    # Percent per year
    for i, keyword in enumerate(keywords):
        ax[1].plot(mean_gdf.index, mean_gdf[keyword], color=colors[i])    
    ax[0].set_ylabel('Number')
    ax[1].set_ylabel('Percentage')
    ax[0].set_ylim(0, count_gdf.max().max()*1.05)
    ax[1].set_ylim(0, mean_gdf.max().max()*1.05)
    ax[1].set_yticks(np.arange(0, mean_gdf.max().max()*1.05, y1_tickspacing))
    ax[1].set_yticklabels((np.arange(0, (mean_gdf.max().max()*1.05)*100, (y1_tickspacing)*100)).astype(int))
    for a in ax:
        a.set_xlabel('Year')
        a.set_xticks([2005, 2010, 2015, 2020])
        a.set_xticks(np.arange(2005, 2024, 1), minor=True)
        a.set_aspect('auto')

    fig.legend(loc='lower center', ncol=2, bbox_to_anchor=legend_offset)
    fig.tight_layout()
    fig.savefig(sname + '.png', dpi=600)
    fig.savefig(sname + '.svg')
    if return_trends: 
        corr_dfs = {}
        for keyword in keywords:
            corr_dfs[keyword] = pg.correlation.corr(x=mean_gdf.index, y=mean_gdf[keyword], method='spearman')
        corr_dfs = pd.concat(corr_dfs)
        return corr_dfs, mean_gdf
    
def donut_plot(data, sname, color, rotate=False):
    fig, ax = plt.subplots(1)
    labels = list(data.index)
    for i in range(len(labels)): 
        mean_val = (data[labels[i]] / data.sum()) * 100
        labels[i] += '\n(' + str(round(mean_val, 2)) + '%)'
    ax.pie(data, labels=labels, colors=color, startangle=90, rotatelabels=rotate)
    my_circle=plt.Circle( (0,0), 0.6, color='white')
    p=plt.gcf()
    p.gca().add_artist(my_circle) 
    fig.savefig(sname + '.png', dpi=600, bbox_inches="tight")
    fig.savefig(sname + '.svg', bbox_inches="tight")

def stacked_plot(affiliations, keywords, sname, colors, legend_rows=1, legend_offset=(0, 0)):
    if isinstance(keywords, dict): 
        keywords = list(keywords.keys())
    keywords.append('Not mentioned')
    fig, ax = plt.subplots(figsize=(8, 5))
    xticks = []
    for i, row in affiliations.iterrows():
        running_total = 0
        stacked_plot_data = []
        for keyword in keywords:
            stacked_plot_data.append((running_total, running_total+row[keyword]))
            running_total += row[keyword]
        ax.broken_barh(stacked_plot_data, 
                        [i, 0.8],
                        facecolors=colors)
        xticks.append(row['Uni_eng'])

    ax.set_ylim(0, len(affiliations))
    ax.set_xlim(0, 1)

    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xticks([0, 0.25, 0.50, 0.75, 1])
    ax.set_xticklabels([0, 25, 50, 75, 100])
    ax.set_xlim(0, 1)
    ax.set_axisbelow(True) 
    ax.set_xlabel('Percent')
    leg = []
    for i, key in enumerate(keywords): 
        leg.append(mpatches.Patch(color=colors[i], label=key))
    ax.set_yticks(np.arange(0.4, len(affiliations), 1))
    ax.set_yticklabels(xticks)  
    plt.yticks(fontsize=8)
    ax.legend(handles=leg, ncol=int(len(leg)/legend_rows), loc='lower center', bbox_to_anchor=legend_offset)
    fig.savefig(sname + '.png', dpi=600, bbox_inches="tight")
    fig.savefig(sname + '.svg', bbox_inches="tight")

    
#%%
df = load_data()
# ANALYSIS 1
# %%
# Check each row for keywords
keywords = ['kvantitativ', 'kvalitativ']

#%%
mcolname = 'Both mentioned'
keyword_df = match_keywords(df, keywords, multiple_hit_policy='multiple_column', mcolname=mcolname)

#%%
keyword_df.rename({'kvantitativ': 'Quantitative',
                 'kvalitativ': 'Qualitative'}, axis=1, inplace=True)

#%%
cols = cm.get_cmap('Set2').colors[:3]
keywords_plot = ['Quantitative', 'Qualitative', mcolname]
stats, gdf = plot_keywords(keyword_df, df, keywords_plot, cols, './figures/kvalkvant_time', True, legend_offset=(0.5, -0.1))
print(stats)

# %%
donut_df = keyword_df[keywords_plot].copy()
donut_df['Neither mentioned'] = donut_df.apply(lambda x: 1 if x.sum() == 0 else 0, axis=1)
donut_df = donut_df.sum()
cols = cm.get_cmap('Set2').colors[:4]
donut_plot(donut_df, './figures/kvalkvant_pie', color=cols)

#%%
keywords = ['kvantitativ', 'kvalitativ']
affiliations = load_match_affiliations('data/affiliations.csv', df, keywords, 'total', multiple_hit_policy='multiple_column', mcolname='Both mentioned')
affiliations['check_sum'] = affiliations[keywords + ['Both mentioned', 'Not mentioned']].sum(axis=1)



#%%
cols = cm.get_cmap('Set2').colors[:4]
stacked_plot(affiliations, keywords + ['Both mentioned'], './figures/kvalkvant_uni', colors=cols, legend_offset=(0.5, -0.3))
#%%
#%%
# Make a plot of affiliations
aff_plot = load_match_affiliations('data/affiliations.csv', df, keywords, 'total', multiple_hit_policy='multiple_column', mcolname='Both mentioned', restrict_hist=False)




#%%
# ANALYSIS 2
keywords = json.load(open('./data/method_keywords.json', 'r'))
keyword_df = match_keywords(df, keywords, multiple_hit_policy='multiple_column', mcolname='Multiple mentioned')
keywords_nooverlap = match_keywords(df, keywords, multiple_hit_policy=None)

#%%
overlap = apply_multiple_hit_policy(keywords_nooverlap, keywords, 'return_SzymkSimp', None)

#%%
fig, ax = plt.subplots(1, figsize=(5,5))
pax = ax.imshow(overlap, vmin=0, vmax=1, cmap='inferno', rasterized=True)
ax.set_xticks(np.arange(len(keywords)))
ax.set_xticklabels(keywords, rotation=90)
ax.set_yticks(np.arange(len(keywords)))
ax.set_yticklabels(keywords)
# Add colorbar
cbar = fig.colorbar(pax)
fig.savefig('./figures/keyword_overlap.png', dpi=600)
fig.savefig('./figures/keyword_overlap.svg')
#%%
action_theses = ['Action research', 'Experiment', 'Intervention', 'Lesson study']
print(keywords_nooverlap[action_theses].sum(axis=1).mean())
#%%
cols = cm.get_cmap('tab20').colors
stats, gdf = plot_keywords(keywords_nooverlap, df, keywords, cols, './figures/keywords_time', True, legend_offset=(0.5, -0.5))
print(stats)
stats.to_csv('./stats_a2.csv')
# %%
# Drop Interview and run aagain
keyword_df_cropped = keywords_nooverlap.drop('Interview', axis=1)
keyword_df_cropped = keyword_df_cropped.drop('Observation', axis=1)
keyword_df_cropped = keyword_df_cropped.drop('Survey', axis=1)
keyword_cropped = keywords.copy()
keyword_cropped.pop('Interview')
keyword_cropped.pop('Observation')
keyword_cropped.pop('Survey')
cols = cm.get_cmap('tab20').colors
cols = cols[:3] + cols[5:]
cols = cols[:1] + cols[2:]
stats = plot_keywords(keyword_df_cropped, df, keyword_cropped, cols, './figures/keywords_time_no_survey', True, legend_offset=(0.5, -0.5), y1_tickspacing=0.02)
#%%
donut_df = keyword_df.copy()
donut_df.drop('year', axis=1, inplace=True)
donut_df['None mentioned'] = donut_df.apply(lambda x: 1 if x.sum() == 0 else 0, axis=1)
donut_df = donut_df.sum()
donut_df['Other methods'] = donut_df[donut_df/donut_df.sum()<=0.01].sum()
# drop the ones with less than 1%
donut_df = donut_df[donut_df/donut_df.sum()>0.01]
cols = cm.get_cmap('tab20').colors
donut_plot(donut_df, './figures/keywords_pie', color=cols)

#%%
affiliations = load_match_affiliations('data/affiliations.csv', df, keywords, 'total', multiple_hit_policy='multiple_column', mcolname='Multiple mentioned')

#%%
cols = cm.get_cmap('tab20').colors
uni_keywords = list(keywords.keys()) + ['Multiple mentioned']
stacked_plot(affiliations, uni_keywords, './figures/keywords_uni', colors=cols, legend_rows=4, legend_offset=(0.5, -0.6))

# %%
## Analysis 3
keywords = json.load(open('./data/keywords.json', 'r'))
keyword_df = match_keywords(df, keywords, multiple_hit_policy='heuristic_takemax_w_deprioritize', mcolname='Multiple mentioned', deprio_col='Learning and instruction')
#%%
cols = cm.get_cmap('Set1').colors
kdf = keyword_df.copy()
kdf.drop('None mentioned', axis=1, inplace=True)
stats, gdf = plot_keywords(kdf, df, keywords, cols, './figures/themes_time', True, legend_offset=(0.5, -0.25))
print(stats)
stats.to_csv('./stats_a3.csv')

#%%
donut_df = keyword_df.copy()
donut_df.drop('year', axis=1, inplace=True)
#donut_df['None mentioned'] = donut_df.apply(lambda x: 1 if x.sum() == 0 else 0, axis=1)
donut_df = donut_df.sum()
cols = cm.get_cmap('Set1').colors
donut_plot(donut_df, './figures/themes_pies', color=cols, rotate=False)

# %%
affiliations = load_match_affiliations('data/affiliations.csv', df, keywords, 'total', multiple_hit_policy='heuristic_takemax_w_deprioritize', mcolname='None mentioned', deprio_col='Learning and instruction')
affiliations = affiliations[affiliations['total']>50]
cols = cm.get_cmap('Set1').colors
stacked_plot(affiliations, keywords, './figures/themes_uni', colors=cols, legend_rows=3, legend_offset=(0.5, -0.5))

# %%
affiliations = load_match_affiliations('data/affiliations.csv', df, keywords, 'total', multiple_hit_policy='multiple_column', mcolname='Both mentioned', restrict_hist=False)
affiliations = affiliations[['Uni', 'Uni_eng', 'total']].dropna()


# %%
fig, ax = plt.subplots(1, figsize=(5, 8))
ax.barh(np.arange(0, len(affiliations)), affiliations['total'])
ax.set_yticks(np.arange(0,len(affiliations)))
ax.set_yticklabels(affiliations['Uni_eng'])
ax.plot([0, 3500], [len(affiliations)-5.5, len(affiliations)-5.5], color='gray', linestyle='--')
ax.set_xlim([0, affiliations['total'].max()*1.05])
ax.set_ylim([-0.5, len(affiliations)-0.5])
fig.savefig('./figures/n_per_uni.png', dpi=600)
fig.savefig('./figures/n_per_uni.svg')
# %%

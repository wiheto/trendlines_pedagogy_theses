#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#%%
df = pd.read_csv('./data/data.csv', index_col=[0])
# %%
fig, ax = plt.subplots(1)

ax.hist(df['Year'], bins=np.arange(1990, 2025, 1),
        color='cornflowerblue')

ax.set_ylabel('Number of thesis')
ax.set_xlabel('Year')
ax.set_yticks(np.arange(0, 4000, 1000))
ax.set_yticklabels(np.round(np.arange(0, 4000, 1000)))

ax.set_xticks(np.arange(1990, 2025, 5))
ax.set_xticklabels(np.round(np.arange(1990, 2025, 5)))

# Set background to white
fig.patch.set_facecolor('white')

fig.savefig('./figures/number_of_thesis_per_year.png', dpi=300)
fig.savefig('./figures/number_of_thesis_per_year.svg')

# %%

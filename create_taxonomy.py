# %% imports
import pandas as pd
from pprint import pprint

# %% read data
df = pd.read_csv("jobs.csv")

# %%
job = df.loc[0, ["title", "description"]].to_dict()

# %%
print(job["title"])
print(job["description"])

# %% give the title and description to the llm
# and have it extract the job title that the hired person will get
# and any job titles that are mentioned as job titles that would entail a
# person should consider themself a relevant applicant


# %%

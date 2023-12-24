# %% imports
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool, StructuredTool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.render import format_tool_to_openai_function
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.agents import AgentExecutor
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import dotenv
import os
from tqdm import tqdm
from pprint import pprint
import json
from fuzzywuzzy import process, fuzz

tqdm.pandas()

assert dotenv.load_dotenv()


# %%
openai_api_key = os.getenv("OPENAI_API_KEY")
org_id = os.getenv("OPENAI_ORG_ID")
os.environ["OPENAI_API_KEY"] = openai_api_key
openai_model = '"gpt-3.5-turbo"'
openai_model = 'gpt-3.5-turbo-1106'


llm = ChatOpenAI(model=openai_model, temperature=0.0, organization=org_id)


# %% read data
# dtype_dict = {
#     'job_id': 'uint32',
#     'title': 'string',
#     'description': 'string',
#     }

# df = pd.read_csv('job_postings.csv', dtype=dtype_dict, index_col='job_id', usecols=list(dtype_dict.keys()))
# df['full_text'] = df['title'] + "\n" + df['description']

# %% read Danish data
# dtype_dict = {

df = pd.read_csv('jobs.csv')
df['full_text'] = df['title'] + "\n" + df['description']

# %% sort df by title
# df.sort_values(by='title', inplace=True)

# %%
# job = df.loc[3757940025, "full_text"]
# pprint(job)

# %% make dataframe with taxonomy
index_cols = ['job_title']
other_cols = ['synonyms', 'description'] 
TAXONOMY = pd.DataFrame(columns=index_cols + other_cols).set_index(index_cols)
global TAXONOMY

description_of_taxonomy = """
    job_title (string): The job title that is being described
    synonyms (string): Stringed together synonyms for the job title, "##" seperated
    description (string): "Description of the job title here. Kept short and focused on keywords."
"""

# %%
def stringify_taxonomy_row(ix, row):
   return f"### job title: {ix} ###\n {row.to_dict()}\n### END ###"
   
def stringify_taxonomy_rows(df):
    return "\n".join([stringify_taxonomy_row(ix, row) for ix, row in df.iterrows()])

def top_similar_descriptions(df, new_desc, top_n=3):
    scores = process.extract(new_desc, df['description'], 
                                scorer=fuzz.token_set_ratio, limit=top_n)
    hits = []
    for score in scores:
        hit = TAXONOMY[TAXONOMY['description'] == score[0]].copy()
        hit['score'] = score[1]
        hits.append(hit)
    return pd.concat(hits)

def look_up_job_title_in_taxonomy(job_title: str) -> str:
    """
    Search the taxonomy DataFrame for a job title.
    
    :param job_title: The job title string to search for.
    :param description: stringed together keywords and key sentences (seperated with "##") that describe the job title's most common tasks, responsibilities, qualifications, tools and area of work
    
    :return: one or more rows from a pandas DataFrame, stringed together.
    """
    if TAXONOMY.empty:
       return "Taxonomy does not contain anything"

    # check if job_title is in the first level of the taxonomy's index (the one called job_title)
    try:
        return stringify_taxonomy_rows(TAXONOMY.loc[[job_title], other_cols])
    except:
        return "Job title not found in taxonomy"
    
def search_for_similar_descriptions_in_taxonomy(description: str) -> str:
    """
    Search the taxonomy DataFrame for a similar description
    """
    if TAXONOMY.empty:
       return "Taxonomy does not contain anything"
    top_matches = top_similar_descriptions(TAXONOMY, description)
    return stringify_taxonomy_rows(top_matches)
    
def add_job_title_to_taxonomy(
        job_title: str,
        synonyms: str = '',
        description: str = '',
        ) -> str:
  """
  Add a row to the taxonomy with the inputted job title in the taxonomy's "job title" column. Further, you can add synonyms and a description of the job title.
  
  :param job_title: The job title that will be put in the new rows column names "job_title"
  :param synonyms: stringed together synonyms for the job title, "##" seperated
  :param description: stringed together keywords and key sentences (seperated with "##") that describe the job title's most common tasks, responsibilities, qualifications, tools and area of work
  :return: String describing whether the job title was added or not
  """

  # add row_to_add to the dataframe taxonomy
  
  try:
    TAXONOMY.loc[job_title] = [synonyms, description]
    return "Added job title to taxonomy"
  except Exception as e:
    return "Could not add job title to taxonomy" + str(e)
  
def add_synonym_to_job_title_in_taxonomy(
        job_title: str,
        synonyms: str,
        ) -> str:
  """
  Adds a synonym to a job title in the taxonomy

  :param synonyms: stringed together synonyms for the job title, "##" seperated


  """
  try:
    TAXONOMY.loc[job_title, "synonyms"] = TAXONOMY.loc[job_title, "synonyms"] + " ## " + synonyms
    return "Added synonym"
  except Exception as e:
    return "Could not add synonym to taxonomy" + str(e)

def replace_description_of_job_title_in_taxonomy(
        job_title: str,
        updated_description: str,
        ) -> str:
  """
  Replaces a description of a job title with a new description.

  :param updated_description: stringed together keywords and key sentences (seperated with "##") that describe the job title's most common tasks, responsibilities, qualifications, tools and area of work
  :return: String describing whether the operation was a succes
  """
  try:
    TAXONOMY.loc[job_title, "description"] = updated_description
    return "Previous description replaced by updated description"
  except Exception as e:
    return "Could not edit description in taxonomy" + str(e)
  
def delete_job_title_from_taxonomy(
        job_title: str,
        ) -> str:
  """
  Deletes a job title and its associated data from the taxonomy

  :param job_title: The job title that will deleted from the taxonomy
  :return: String describing whether the operation was a succes
  """
  try:
    # delete the row from the dataframe TAXONOMY
    TAXONOMY.drop(job_title, inplace=True)
    return "Deleted job title from taxonomy"
  except Exception as e:
    return "Could not delete job title from taxonomy" + str(e)


# %% Create the custom DataFrame lookup tool
look_up_job_title_in_taxonomy_tool = Tool.from_function(
    func=look_up_job_title_in_taxonomy,
    name="look_up_job_title_in_taxonomy",
    description="Lookup a job title in the taxonomy."
)

search_for_similar_descriptions_in_taxonomy_tool = Tool.from_function(
    func=search_for_similar_descriptions_in_taxonomy,
    name="search_for_similar_descriptions_in_taxonomy",
    description="Find best matches for description int the taxonomy."
)

add_job_title_to_taxonomy_tool = StructuredTool.from_function(
    func=add_job_title_to_taxonomy,
    name="add_job_title_to_taxonomy",
    description="Add a job title and it's related information to the taxonomy"
)

add_synonym_to_job_title_in_taxonomy_tool = StructuredTool.from_function(
    func=add_synonym_to_job_title_in_taxonomy,
    name="add_synonym_to_job_title_in_taxonomy",
    description="Add a synonym to a job title in the taxonomy"
)

replace_description_of_job_title_in_taxonomy_tool = StructuredTool.from_function(
    func=replace_description_of_job_title_in_taxonomy,
    name="edit_description_of_job_title_in_taxonomy",
    description="Edit the description of a job title in the taxonomy"
)

delete_job_title_from_taxonomy_tool = Tool.from_function(
    func=delete_job_title_from_taxonomy,
    name="delete_job_title_from_taxonomy",
    description="Delete a job title from the taxonomy"
)

tools = [
    look_up_job_title_in_taxonomy_tool, 
    search_for_similar_descriptions_in_taxonomy_tool, 
    add_job_title_to_taxonomy_tool, 
    add_synonym_to_job_title_in_taxonomy_tool, 
    replace_description_of_job_title_in_taxonomy_tool,
    delete_job_title_from_taxonomy_tool
   ] 
llm_with_tools = llm.bind(functions=[format_tool_to_openai_function(t) for t in tools])

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            f"""
            You are an assistant that builts and maintains a taxonomy of job titles.

            You take job postings, find the job titles in them and add them to the txonomy if they are not there. You also check whether a job title is already in the taxonomy or whether the job title is a synonym of another job title that is already in the taxonomy.

            The taxonomy should contain job titles and descriptions of the job titles. The descriptions should be short and focused on keywords. Job titles that are synonyms should be under the same entry in the taxonomy. 
            A synonym is e.g. singular and plural forms for the same job title, e.g. "nurse" and "nurses". Or abbreviation e.g. "Assistant Store Director" and "ASD". In these cases, the job title should be the singular form and the plural or abbrevation should be listed as synonyms:
            job_title: "Nurse", synonyms: "Nurses ## RN ## Registered Nurse". If only the plural or abbreviated form is mentioned in the job posting, you should use that as the main job title.
            The taxonomy should also include a description of the job title's most common tasks, responsibilities, qualifications, tools and area of work. The description has to be short, focused on keywords and written in english.

            Here is the list of steps you should go through, when you are provided with a job posting:
            1. Identify any explicitly written job titles in the job posting. It is very important that they are written in the posting, you are not allowed to infer the job title from the description. If there are no explicitly written job titles, you should write "No explicit job title found" and then you are done.
               If the job title is written in multiple different ways (E.g. both as singular, plural and in abbreviated form), you pick the non-abbreviated singular form as the main and consider the others as synonyms. 
               So if "Assistant Store Director", "Assistant Store Directors" and "ASD" are all used in a job posting,pick "Assistant Store Director" as the main job title and add the other two as synonyms.
            2. Check if the main job title is already in the job title taxonomy. There is two ways of doing this:
                2a. First, use the tool for checking if a job title is in the taxonomy. Only do this once.
                2b. Second, use the tool for searching for similar descriptions in the taxonomy. To do this search, you have to write a description of the job title based on the job posting that follows how descriptions should be structured in the taxonomy. Only do this once, also.
            3. Evaluate your results from step 2. There are three possible outcomes:
                3a. The verbatim job title is already in the taxonomy.
                3b. The verbatim job title is not in the taxonomy but there is a description in the taxonomy that is very similar to the description in the job posting.
                3c. The verbatim job title is not in the taxonomy and there is no description in the taxonomy that is very similar to the description in the job posting.
            3. If the job title is not in the taxonomy and there is no description in the taxonomy that comes close to matching the description of the job posting, you should add the job title form the job posting to the taxonomy. Remember to include any synonyms that are written in the job posting (if there are several, string them together with ' ## ' as separator) and also to add a description of the job title. There is a tool for adding a job title to the taxonomy, including adding synonyms and a description of the job titles function.
            4. If the job title is in the taxonomy, check if the description of the job title in the taxonomy makes sense compared to the job posting you are looking at.
                4a. if the description in the taxonomy is quite close to the description in the job posting, all is good. If there is something relevant missing from the description, there is a tool for editing the description of a job title in the taxonomy.
                4b. if the description in the taxonomy is describing a fundamentally different job than the job posting is, the same verbatim job title is likely used for different jobs. This means we want to split the job title into two different job titles, e.g. splitting "Centerleder" into "Centerleder, plejecenter" and "Centerleder, jobcenter". 
                To achieve this, you need to delete the existing job title from the taxonomy and then add two new job titles to the taxonomy. There is a tool for deleting a job title from the taxonomy. You can then use the tool for adding job titles twice.
            5. If the job title is not in the taxonomy but there is a description in the taxonomy that is very similar to the description in the job posting, you have to choose whether the job title from the job posting you are currently looking at is a synonym of this job title in the taxonomy or if it is a new job title. 
               If it is a synonym, you can use the tool for adding a synonym to a job title in the taxonomy. If it is a new job title, you can use the tool for adding a job title to the taxonomy.
               You mnight also find that the taxonomy already contains "Nurses", but the job posting you are looking at contains the job title "Nurse". In this case, "Nurse" is probably a better job title than "Nurses".
               The right way to solve this is then to delete the job title "Nurses" from the taxonomy and then add the job title "Nurse" to the taxonomy, with "Nurses" as a synonym. 
               You can use the tool for deleting a job title from the taxonomy and the tool for adding a job title to the taxonomy to acomplish this.

            Here is a list of all the tools:
            {tools}
            """,
        ),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_function_messages(
            x["intermediate_steps"]
        ),
    }
    | prompt
    | llm_with_tools
    | OpenAIFunctionsAgentOutputParser()
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# %%
# agent_executor.invoke({"input": str(job)})

# %%
for ix, row in df[0:5].iterrows():
    agent_executor.invoke({"input": str(row['full_text'])})

# %%
jt_tax = {
    'job_title': 'receiving assistant',
    'synonyms': 'receiving associate ## shipping associate',
    'description': 'packing customer orders for shipping ## receiving material and equipment ## loading and unloading trucks',
}

add_job_title_to_taxonomy(**jt_tax)

# %%
TAXONOMY

# %% Check the tool for looking up a job title in the taxonomy
print(look_up_job_title_in_taxonomy(jt_tax['job_title']))

# %% Check the tool for looking up descriptions in the taxonomy
print(search_for_similar_descriptions_in_taxonomy(jt_tax['description']))


# %% Let's test
data = {
    'explicit_job_title': 'Shipping & Receiving Associate',
    'description': 'Shipping but also a deal of nursery work. And you will be working with a lot of different people.'
    # 'description': 'This job involves packing and preparing customer orders for shipping, receiving material and equipment, loading and unloading trucks, using material handling equipment, inspecting inbound materials, maintaining quality records, communicating with purchasing department, restocking delivery items, ensuring compliance with safety standards, completing paperwork, and maintaining housekeeping standards.'
    }

# %% Check that this new job title does not return anything
print(look_up_job_title_in_taxonomy(
    job_title=data['explicit_job_title'],
))

# %% Check that we can search for a description
print(search_for_similar_descriptions_in_taxonomy(
    description=data['description'],
))

# %% Test that we can add it as a synonym
add_synonym_to_job_title_in_taxonomy(
    job_title="receiving assistant",
    synonyms=data['explicit_job_title']
    )

# %% Test that we can replace a description of a job title
replace_description_of_job_title_in_taxonomy(
    job_title="receiving assistant",
    updated_description=data['description']
    )

# %% Test that we can delete a job title

delete_job_title_from_taxonomy(
    job_title="receiving assistant",
    )

# %%
TAXONOMY

# %% imports
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool
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

tqdm.pandas()

assert dotenv.load_dotenv()


# %%
openai_api_key = os.getenv("OPENAI_API_KEY")
org_id = os.getenv("OPENAI_ORG_ID")

os.environ["OPENAI_API_KEY"] = openai_api_key


# %% Set up llm (run with langchain) and openai
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0, organization=org_id)
client = OpenAI(api_key=openai_api_key, organization=org_id)

# %% read data
dtype_dict = {
    'job_id': 'uint32',
    'title': 'string',
    'description': 'string',
    }

df = pd.read_csv('job_postings.csv', dtype=dtype_dict, index_col='job_id', usecols=list(dtype_dict.keys()), nrows=50)
df['full_text'] = df['title'] + "\n" + df['description']

# %%
job = df.loc[3757940025, "full_text"]
pprint(job)

# %%
prompt = f"""
Based on the job posting below, identify the following:
1. The explicit job title being offered in this posting.
2. Only job titles that are similar or directly related to the position offered, suitable for potential applicants. Do not include job titles like supervisors, managers, or contact person's titles, such as 'pædagogisk leder', unless they are explicitly stated as relevant for applicants.
3. A description of the job title, including the most common tasks, responsibilities, qualifications, tools and area of work. But shortened to main keywords.

Job Posting:
{job}

## JOB POSTING FINISHED ##

Please provide the information in the following JSON format:
{{
  "explicit_job_title': "[Explicit job title here]",
  "related_job_titles_for_applicants": ["List of related job titles here, if they are explcitly in the job posting"],
  "description": "Description of the job title here. Keep it short and focus on keywords."
}}
"""

r = llm.invoke(prompt)
string_data = r.dict()['content']
json_data = string_data.replace("'", '"') # To ensure that the json is valid
print(string_data)

# %%
data = json.loads(json_data)

# %% make dataframe with taxonomy
index_cols = ['job_title', 'additional_qualifier']
other_cols = ['synonyms', 'relevant_job_titles', 'description'] 
complex_cols = ['vector']
taxonomy = pd.DataFrame(columns=index_cols + other_cols + complex_cols).set_index(index_cols)

description_of_taxonomy = """
    job_title (string): The job title that is being described
    additional_qualifier (string): A qualifier that is added if same job title is used in different contexts, e.g. "manager"
    synonyms (string): Stringed together synonyms for the job title, "##" seperated
    relevant_job_titles (string): Stringed together job titles that are related to the job title, "##" seperated
    description (string): stringed together keywords and key sentences (seperated with "##") that describe the job title's most common tasks, responsibilities, qualifications, tools and area of work
    vector (list of floats): A vector representation of the job title
"""





# %%
def get_embedding(text:str, model: str="text-embedding-ada-002") -> list:
    text = text.replace("\n", " ")
    return np.array([client.embeddings.create(input = [text], model=model).data[0].embedding])


def stringify_taxonomy_row(ix, row):
   return f"### job title: {ix[0]} + additional_qualifier {ix[1]} ###\n {row.to_dict()}\n### ITEM END ###"
   

def stringify_taxonomy_rows(df):
    return "\n".join([stringify_taxonomy_row(ix, row) for ix, row in df.iterrows()])


def return_top_similarities(vector_query: list) -> pd.DataFrame:
    """ 
    Just a poor man's implementation of cosine similarity. 
    """
    return taxonomy['vector'].apply(lambda vector_doc: cosine_similarity(vector_query, vector_doc)[0,0])

def query_taxonomy(
        job_title: str, 
        description: str
        ) -> str:
    """
    Search the taxonomy DataFrame for a given query.
    
    :param job_title: The job title string to search for.
    :param description: stringed together keywords and key sentences (seperated with "##") that describe the job title's most common tasks, responsibilities, qualifications, tools and area of work
    
    :return: one or more rows from a pandas DataFrame, stringed together.
    """
    if taxonomy.empty:
       return taxonomy[other_cols].to_string()

    # check if job_title is in the first level of the taxonomy's index (the one called job_title)
    if job_title in taxonomy.index.get_level_values('job_title'):
        return taxonomy.loc[[job_title], other_cols].to_string()
    
    # embed decription and do a vector search
    description_embedding = get_embedding(description)
    similarities = return_top_similarities(description_embedding)

    # return the top 5 results. Take the index of the top 5 results and use it to get the 
    # corresponding rows from the taxonomy
    top_5 = similarities.nlargest(5).index
    return stringify_taxonomy_rows(taxonomy.loc[top_5, other_cols])
    


def add_to_taxonomy(
        job_title: str,
        additional_qualifier: str = '',
        synonyms: str = '',
        relevant_job_titles: str = '',
        description: str = '',
        ) -> str:
  """
  Add a row to the taxonomy with the inputted job title in the taxonomy's "job title" column.
  The function updates the taxonomy but does not return anything
  

  :param job_title: The job title that will be put in the new rows column names "job_title"
  : return: None
  """
  

  # add row_to_add to the dataframe taxonomy
  global taxonomy
  try:
    job_title_to_add = pd.DataFrame([{
        'job_title': job_title,
        'additional_qualifier': additional_qualifier,
        'synonyms': synonyms,
        'relevant_job_titles': relevant_job_titles,
        'description': description,
        }])
    job_title_to_add.set_index(['job_title', 'additional_qualifier'], inplace=True)
    job_title_to_add['vector'] = job_title_to_add['description'].apply(get_embedding)
    taxonomy = pd.concat([taxonomy, job_title_to_add])
    return "Added job title to taxonomy"
  except IndentationError:
    return "Could not add job title to taxonomy"

# %%
# jt_tax = pd.DataFrame([{
#     'job_title': 'receiving assistant',
#     'additional_qualifier': '',
#     'synonyms': 'receiving associate ## shipping associate',
#     'relevant_job_titles': '',
#     'description': 'packing customer orders for shipping ## receiving material and equipment ## loading and unloading trucks',
# }]).set_index(['job_title', 'additional_qualifier'])
# jt_tax['vector'] = jt_tax['description'].apply(get_embedding)
# pd.concat([taxonomy, jt_tax])

# %%
jt_tax = {
    'job_title': 'receiving assistant',
    'additional_qualifier': '',
    'synonyms': 'receiving associate ## shipping associate',
    'relevant_job_titles': '',
    'description': 'packing customer orders for shipping ## receiving material and equipment ## loading and unloading trucks',
}

add_to_taxonomy(**jt_tax)
# embedding = get_embedding(jt_tax['description'])


# %% Let's test
data = {
    'explicit_job_title': 'Shipping & Receiving Associate',
    'related_job_titles_for_applicants': ['Shipping Associate','Receiving Associate'],
    'description': 'This job involves packing and preparing customer orders for shipping, receiving material and equipment, loading and unloading trucks, using material handling equipment, inspecting inbound materials, maintaining quality records, communicating with purchasing department, restocking delivery items, ensuring compliance with safety standards, completing paperwork, and maintaining housekeeping standards.'
    }

embedding = get_embedding(data['description'])

# %%
# taxonomy['vector'].apply(lambda vector_doc: cosine_similarity(embedding, vector_doc))
similarities = return_top_similarities(embedding)
top_5 = similarities.nlargest(5).index


# %%


print(stringify_taxonomy_rows(taxonomy.loc[top_5, other_cols]))


# %%

list_of_items = []
for ix, row in taxonomy.loc[top_5, other_cols].iterrows(): #.transpose().to_dict()
    job_title_item = 


print(job_title_item)

# %%
print(query_taxonomy(
    job_title=data['explicit_job_title'],
    description=data['description'],
))


# %%


# Create the custom DataFrame lookup tool
taxonomy_lookup_tool = Tool.from_function(
    func=query_taxonomy,
    name="TaxonomyLookup",
    description="Lookup information in the taxonomy. It will return matches on job title. If there are no matches, it returns top 5 most similar descriptions."
)

taxonomy_add_tool = Tool.from_function(
    func=add_to_taxonomy,
    name="TaxonomyAdd",
    description="Add a job title and it's related information to the taxonomy"
)
tools = [taxonomy_lookup_tool, taxonomy_add_tool]  # Add other tools as needed
llm_with_tools = llm.bind(functions=[format_tool_to_openai_function(t) for t in tools])

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            f"""
            You are very powerful assistant.
            Your job is to take a job title and look it up in a taxonomy. If the job title is not in the taxonomy, you should add it to the taxonomy.
            You can use the following tools:
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

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# %%
agent_executor.invoke({"input": "sygeplejerske"})

# %%
# %% imports
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
from langchain.llms import OpenAI
from langchain.tools import Tool
# from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.render import format_tool_to_openai_function
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.agents import AgentExecutor

import dotenv
import os
from tqdm import tqdm

tqdm.pandas()

assert dotenv.load_dotenv()

# %%
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
org_id = os.getenv("OPENAI_ORG_ID")

# %% Set up llm
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0, organization=org_id)

# %% read data
df = pd.read_csv("jobs.csv")

# %%
df['full_text'] = df['title'] + "\n" + df['description']

# %%
job = df.loc[1, "full_text"]

# %%
print(job)

# %%
prompt = f"""
Based on the job posting below, identify the following:
1. The explicit job title being offered in this posting.
2. Only job titles that are similar or directly related to the position offered, suitable for potential applicants. Do not include job titles like supervisors, managers, or contact person's titles, such as 'pædagogisk leder', unless they are explicitly stated as relevant for applicants.

Job Posting:
{job}

Please provide the information in the following JSON format:
{{
  'explicit_job_title': '[Explicit job title here]',
  'relevant_job_titles_for_applicants': ['List of related job titles here, if any']
}}
"""

r = llm.invoke(prompt)

# %%
print(r.dict()['content'])

# %%
taxonomy = pd.DataFrame(
    columns=[
        'job_title', 
        'synonyms', 
        'related_job_titles', 
        'additional_qualifiers',
        'description'
        ])

# taxonomy_with_entry = taxonomy.copy()
taxonomy.loc[0, 'job_title'] = 'vejleder'
taxonomy.loc[0, 'description'] = 'Det her er en vejleder på en skole'


# %%
def query_taxonomy(query: str) -> str:
    """
    Search the taxonomy DataFrame for a given query.
    
    :param query: The query string to search for.
    :return: A string describing the search results.
    """
    results = taxonomy[taxonomy['job_title'].str.contains(query, case=False, na=False)]
    if results.empty:
        return "No results found."
    else:
        return results.to_string(index=False)  # Convert DataFrame to string for output


def add_to_taxonomy(job_title: str) -> None:
  """
  Add a row to the taxonomy with the inputted job title in the taxonomy's "job title" column.
  The function updates the taxonomy but does not return anything
  

  :param job_title: The job title that will be put in the new rows column names "job_title"
  : return: None
  """
  row_to_add = pd.DataFrame([{'job_title': job_title}])

  # add row_to_add to the dataframe taxonomy
  global taxonomy
  taxonomy = pd.concat([taxonomy, row_to_add], ignore_index=True)

def add_synonym_to_taxonomy(job_title: str, synonym: str) -> None:
    return None

def 


# Create the custom DataFrame lookup tool
taxonomy_lookup_tool = Tool.from_function(
    func=query_taxonomy,
    name="TaxonomyLookup",
    description="Lookup information in the taxonomy"
)

taxonomy_add_tool = Tool.from_function(
    func=add_to_taxonomy,
    name="TaxonomyAdd",
    description="Add a job title to the taxonomy"
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
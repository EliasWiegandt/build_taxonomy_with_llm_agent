# %% imports
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool, StructuredTool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.render import format_tool_to_openai_function
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.agents import AgentExecutor
import dotenv
import os
from fuzzywuzzy import process, fuzz

assert dotenv.load_dotenv()


# %%
openai_api_key = os.getenv("OPENAI_API_KEY")
org_id = os.getenv("OPENAI_ORG_ID")
os.environ["OPENAI_API_KEY"] = openai_api_key
openai_model = 'gpt-4-1106-preview'

# %%
dtype_dict = {
    'job_id': 'uint32',
    'title': 'string',
    'description': 'string',
    }
df = pd.read_csv('job_postings.csv', dtype=dtype_dict, index_col='job_id', usecols=list(dtype_dict.keys()), nrows=50)
df['full_text'] = df['title'] + "\n" + df['description']

# %% make dataframe with taxonomy
index_cols = ['main_job_title']
other_cols = ['synonyms', 'description'] 
TAXONOMY = pd.DataFrame(columns=index_cols + other_cols).set_index(index_cols)
global TAXONOMY

description_of_taxonomy = """
    main_job_title (string): The job title that is used as the main job title in the taxonomy. Should be the singular form of the job title.
    synonyms (string): Stringed together synonyms for the job title, "##" seperated
    description (string): "Description of the job title here. Kept short and focused on keywords."
"""

# %%
def stringify_dict(d):
    return f"### START \n{'\n'.join([key + ': ' + value for key, value in d.items()])}\n### END"

def stringify_taxonomy_row(ix, row):
   return stringify_dict(row.to_dict())

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

def look_up_main_job_title_in_taxonomy(main_job_title: str) -> str:
    """
    Search the taxonomy DataFrame for a job title.
    
    :param job_title: The main job title string to search for.
    :return: one or more rows from a pandas DataFrame, stringed together.
    """
    main_job_title = main_job_title.lower()
    if TAXONOMY.empty:
       return "Taxonomy is empty, i.e. there are no job titles at all in it currently."

    try:
        return stringify_taxonomy_rows(TAXONOMY.loc[[main_job_title], other_cols])
    except:
        return "Job title not found in taxonomy"
    
def search_for_similar_descriptions_in_taxonomy(description: str) -> str:
    """
    Search the taxonomy DataFrame for a similar description

    :param description: stringed together keywords and key sentences (seperated with "##") that describe the job title's most common tasks, responsibilities, qualifications, tools and area of work
    """
    description = description.lower()
    if TAXONOMY.empty:
       return "Taxonomy is empty, i.e. there are no job titles at all in it currently."
    top_matches = top_similar_descriptions(TAXONOMY, description)
    return stringify_taxonomy_rows(top_matches)
    
def add_job_title_to_taxonomy(
        main_job_title: str,
        synonyms: str = '',
        description: str = '',
        ) -> str:
    """
    Add a row to the taxonomy with the inputted job title in the taxonomy's "main_job_title" column. Further, you can add synonyms and a description of the job title.

    :param main_job_title: The job title that will be put in the new row's column names "main_job_title"
    :param synonyms: stringed together synonyms for the job title, seperated by ','
    :param description: the job title's most common tasks, responsibilities, qualifications, tools and area of work
    :return: String describing whether the job title was succesfully added as a new job title in the taxonomy.
    """

    main_job_title = main_job_title.lower()
    synonyms = synonyms.lower()
    description = description.lower()

    try:
        TAXONOMY.loc[main_job_title] = [synonyms, description]
        return "Added job title {main_job_title} to as a new main job title in the job title taxonomy"
    except Exception as e:
       return "Could not add job title to taxonomy" + str(e)
  
def add_synonym_to_job_title_in_taxonomy(
        main_job_title: str,
        synonyms: str,
        ) -> str:
    """
    Adds a synonym to a job title in the taxonomy

    :param main_job_title: The job title the synonyms will be added to
    :param synonyms: stringed together synonyms for the job title, seperated by commas
    """

    main_job_title = main_job_title.lower()
    synonyms = synonyms.lower()
    try:
        TAXONOMY.loc[main_job_title, "synonyms"] = TAXONOMY.loc[main_job_title, "synonyms"] + ", " + synonyms
        return f"Succesfully added synonym(s) {synonyms} to {main_job_title}"
    except Exception as e:
        return "Could not add synonym to taxonomy" + str(e)

def replace_description_of_job_title_in_taxonomy(
        main_job_title: str,
        updated_description: str,
        ) -> str:
    """
    Replaces a description of a job title with a new description.

    :param main_job_title: The job title that will be put in the new row's column named "main_job_title"
    :param updated_description: stringed together keywords and key sentences (seperated with "##") that describe the job title's most common tasks, responsibilities, qualifications, tools and area of work
    :return: String describing whether the operation was a succes
    """

    main_job_title = main_job_title.lower()
    updated_description = updated_description.lower()
    try:
        TAXONOMY.loc[main_job_title, "description"] = updated_description
        return "Previous description for '{main_job_title}' replaced by updated description"
    except Exception as e:
        return "Could not edit description in taxonomy" + str(e)
  
def delete_job_title_from_taxonomy(
        main_job_title: str,
        ) -> str:
    """
    Deletes a main job title and its associated data from the taxonomy

    :param main_job_title: The job title that will deleted from the taxonomy
    :return: String describing whether the operation was a succes
    """

    main_job_title = main_job_title.lower()
    try:
        # delete the row from the dataframe TAXONOMY
        TAXONOMY.drop(main_job_title, inplace=True)
        return "Deleted job title from taxonomy"
    except Exception as e:
        return "Could not delete job title from taxonomy" + str(e)


# Set up the tools
look_up_main_job_title_in_taxonomy_tool = Tool.from_function(
    func=look_up_main_job_title_in_taxonomy,
    name="look_up_main_job_title_in_taxonomy",
    description="Lookup a job title in the taxonomy."
)

search_for_similar_descriptions_in_taxonomy_tool = Tool.from_function(
    func=search_for_similar_descriptions_in_taxonomy,
    name="search_for_similar_descriptions_in_taxonomy",
    description="Find most similar descriptions from the taxonomy."
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
    look_up_main_job_title_in_taxonomy_tool, 
    search_for_similar_descriptions_in_taxonomy_tool, 
    add_job_title_to_taxonomy_tool, 
    add_synonym_to_job_title_in_taxonomy_tool, 
    replace_description_of_job_title_in_taxonomy_tool,
    delete_job_title_from_taxonomy_tool
   ]

# %% Set up the LLM and provide it with the tools
llm = ChatOpenAI(model=openai_model, temperature=0.0, organization=org_id)
llm_with_tools = llm.bind(functions=[format_tool_to_openai_function(t) for t in tools])

system_prompt = f"""
You are an assistant that builds and maintains a taxonomy of job titles.

HIGH-LEVEL TASK
You take a job posting and 
1. Identify these from the job posting: 
    a. Note all explicitly stated (verbatim) job titles from the job posting that the future employee will hold. 
    b. Note all explicitly used synonyms of the job title in the job posting (could be plural forms or abbrevations, see further details below)
    c. Make a short description of the tasks, responsibilities, qualifications, tools and area of work described in the job posting.
2. Compare the job title(s) you found in the job posting to the job titles in the taxonomy (both verbatim and by comparing descriptions). It is very  important you use your tools for this.
3. Choose whether the job title from the job posting represents:
    a. A new job title, that you should add to the taxonomy, along with any synonyms you found and the description you made.
    b. A synonym of a job title already in the taxonomy, implying you should update the synonyms for this job title in the taxonomy.
    c. A job title already in the taxonomy, implying you should check if the description of the job title in the taxonomy is accurate and update it if necessary.


DEFINITIONS
Each row in the taxonomy should be a "main job title", its synonyms and a description of tasks, responsibilities, qualifications, tools and area of work described commonly related to the job title.
A "main job title" is the job title that we use to refer to the entire row in the taxonomy. We prefer non-abbreviated singular forms as main job titles.
A synonym is e.g. plural forms of the main job title, e.g. "nurses" for the main_job_title "nurse". Or abbreviations e.g. "ASD" for "Assistant Store Director". 

EXAMPLES
These are examples of work flows and how you use the different tools. Describe you thoughts at each step.

EXAMPLE 1
You are given this (shortned) job posting: "To pædagoger på 28-30 timer til KKFO'en ved Dyvekeskolen [...]"
1. You extract the following information:
    a. Explicitly stated job title: "pædagoger" (note, although the job title is plural, if the singular form is not mentioned anywhere in the job posting, you should use the plural form as the main job title in the taxonomy)
    b. Synonyms: NONE
    b. Description: YOU MAKE THIS BASED ON THE POSTING
2. You use the tool 'look_up_main_job_title_in_taxonomy' for checking if a job title is in the taxonomy and find that "pædagoger" is not in the taxonomy.
    You then use the tool 'search_for_similar_descriptions_in_taxonomy_tool' for searching for similar descriptions. It returns three results (by default). None are similar to the description of the job posting you made.
3. You choose that "pædagoger" is a new job title that should be added to the taxonomy. You use the tool for adding a job title to the taxonomy to add "Pædagoger" as a new job title to the taxonomy, along with the synonyms "Pædagoger" and a combined description.

EXAMPLE 2
You are given this (shortned) job posting: "Adelphi is seeking a Nurse Practitioner [...]"
1. You extract the following information:
    a. Explicitly stated job title: "Nurse Practitioner"
    b. Synonyms: NONE
    b. Description: YOU MAKE THIS BASED ON THE POSTING
2. You use the tool for checking if a job title is in the taxonomy and find that "Nurse Practitioner" is not in the taxonomy. 
   You then use the tool for searching for similar descriptions. It returns three results (by default). One has the main job title "nurse" and a description that is similar to the description you made.
3. You choose that "Nurse Practitioner" is a synonym of "nurse" and use the tool for adding a synonym to a main job title in the taxonomy to add "Nurse Practitioner" as a synonym to "nurse".

EXAMPLE 3
You are given this (shortned) job posting: "Assistant Store Director (ASD) for [...]"
1. You extract the following information:
    a. Explicitly stated job title: "Assistant Store Director"
    b. Synonyms: "ASD"
    b. Description: YOU MAKE THIS BASED ON THE POSTING
2. You use the tool for checking if a job title is in the taxonomy and find that "Assistant Store Director" is not in the taxonomy. 
   You then use the tool for searching for similar descriptions. It returns three results (by default). None are similar to the description of the job posting you made.
3. You choose that "Assistant Store Director" is a new job title that should be added to the taxonomy. You use the tool for adding a job title to the taxonomy to add "Assistant Store Director" as a new job title to the taxonomy, along with the synonyms "ASD" and the description you made.

EXAMPLE 4
You are given this (shortned) job posting: "Engageret og faglig pædagog til basisteam i Haraldsgården [...]"
1. You extract the following information:
    a. Explicitly stated job title: "pædagog"
    b. Synonyms: NONE
    b. Description: YOU MAKE THIS BASED ON THE POSTING
2. You use the tool for checking if a job title is in the taxonomy and find that "pædagog" is not in the taxonomy. 
   You then use the tool for searching for similar descriptions. It returns three results (by default). One of them "pædagoger" is the plural form of the job title you found in the job posting.
3. You decide that "pædagog", as the singular form, is a better choice for the main job title in the taxonomy than "pædagoger". You use the tool for deleting a job title in the taxonomy to delete "pædagoger". You then add "pædagog" as a main job title in the taxonomy, along with the synonym "pædagoger" and a combined description.


EXAMPLE 5
You are given this (shortned) job posting: "Engageret og faglig pædagog til basisteam i Haraldsgården [...]"
1. You extract the following information:
    a. Explicitly stated job title: "pædagog"
    b. Synonyms: NONE
    b. Description: YOU MAKE THIS BASED ON THE POSTING
2. You use the tool for checking if a job title is in the taxonomy and find that "pædagog" is already in the taxonomy. You check the provided description and find that it is accurate. You then conclude the job title is already in the taxonomy and no further action is needed.

So again:
- Find the job title(s) in the job posting
- Check if the job title(s) is in the taxonomy
- If not, add it to the taxonomy. If it is, or something similar is, decide if the current entries need updating or if the job title is already in the taxonomy and no further action is needed.
"""


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            f"""
            {''.join(system_prompt)}

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
for ix, row in df[-10:].iterrows():
    agent_executor.invoke({"input": str(row['full_text'])})

# %%
TAXONOMY.loc["seasonal beauty adcisor", 'description']
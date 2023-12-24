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
from tqdm import tqdm
from pprint import pprint
from fuzzywuzzy import process, fuzz
import json

tqdm.pandas()

assert dotenv.load_dotenv()


# %%
openai_api_key = os.getenv("OPENAI_API_KEY")
org_id = os.getenv("OPENAI_ORG_ID")
os.environ["OPENAI_API_KEY"] = openai_api_key
openai_model = "gpt-3.5-turbo"
# openai_model = "gpt-3.5-turbo-1106"
llm = ChatOpenAI(model=openai_model, temperature=0.0, organization=org_id)

# %% read Danish data
df = pd.read_csv('jobs.csv')
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
job = df.loc[0, 'full_text']
print(job)

# %% STEP 1: getting the information from the job posting

def get_information_from_job_posting(job: str) -> dict:
    """
    Extract information from a job posting.
    
    :param job: The job posting string to extract information from.
    :return: dictionary with the extracted information.
    """
    prompt = f"""
    Based on the job posting below, identify the following:
    1. The verbatim "main job title" being offered in this posting.
    2. Any synonyms of "main job title" that are explicitly mentioned in the posting. Separate synonyms with "##". This could be plural or singular forms of the job title, or abbreviations. But only if they are eplixitly mentioned in the job posting.
    3. A short summary of the tasks, responsibilities, qualifications, tools and area of work described in the job posting. Keep it short and focused on keywords. Separate keywords and key sentences with "##".

    Job Posting:
    {job}

    ## JOB POSTING FINISHED ##

    Please provide the information in the following JSON format:
    {{
    "explicit_main_job_title': "[Explicit job title here]",
    "explicit_synonyms": "[Synonyms here, if they are explicitly mentioned in the job posting". Separate by ' ## ']",
    "description": "list of the mains tasks and qualifications. Specified as keywords or key setences, seperated by "##""
    }}
    """

    r = llm.invoke(prompt)
    return json.loads(r.dict()['content'].replace("'", '"'))

extracted_from_posting = get_information_from_job_posting(job)

# %%
pprint(extracted_from_posting)

# %% STEP 2: checking if the job title or something close to it is in the taxonomy
def look_up_main_job_title_in_taxonomy(main_job_title: str) -> pd.DataFrame:
    """
    Search the taxonomy DataFrame for a job title.
    
    :param main_job_title: The main job title string to search for.
    :return: pandas DataFrame.
    """
    main_job_title = main_job_title.lower()
    return TAXONOMY[TAXONOMY.index == main_job_title]

main_job_title_matches = look_up_main_job_title_in_taxonomy(extracted_from_posting['explicit_main_job_title'])  

def get_most_similar_descriptions(description_from_job_posting, top_n=3):
    scores = process.extract(description_from_job_posting, TAXONOMY['description'], 
                                scorer=fuzz.token_set_ratio, limit=top_n)
    hits = []
    for score in scores:
        hit = TAXONOMY[TAXONOMY['description'] == score[0]].copy()
        hit['score'] = score[1]
        hits.append(hit)
    if hits:
        return pd.concat(hits)
    else:
        return TAXONOMY[0:0].copy() # return TAXONOMY but with all rows removed

most_similar_descriptions = get_most_similar_descriptions(extracted_from_posting['description'])


# %% STEP 3: deciding what to do with the information from the job posting
if not main_job_title_matches.empty:
    # have LLM evaluate if this is acceptable

elif not most_similar_descriptions.empty:
    # have LLM evaluate if this is acceptable

else:
    # add the job title to the taxonomy

# %% STEP 4: adding the job title to the taxonomy


# %%
# print(stringify_dict(extracted_from_posting))

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


system_prompt = (
    "You are an assistant tasked with building and maintaining a taxonomy of job titles. "
    "Job title are the names of positions within a company that are used in job postings, such as 'Software Engineer' or 'Nurse.' "
    "Your role involves processing job postings to identify job titles, adding them to the "
    "taxonomy, and ensuring synonyms an decriptions are correctly attached to each job title." 
    "Further, a short description of the tasks, responsibilities, qualifications, tools and area of work of each job title is also needed in the taxonomy. "
    "Here are your specific responsibilities and guidelines:\n\n"
    "1. **Identifying Job Titles**: Extract job titles from job postings. Here I mean the explicitly written verbatim job title from the job posting. Do not infer plural or singular forms of job titles from the text in the job posting."
    "If no explicit job title is found, state 'No explicit job title found.'\n\n"
    "2. **Handling Singular and Plural Forms**: Singular and plural forms of a job title should be treated as synonyms. "
    "For instance, 'Nurse' and 'Nurses' are synonyms."
    "If a job posting mentions both forms, add the plural form as a synonym under the singular form."
    "If only the plural form is mentioned, add the plural form as the main job title to the taxonomy.\n\n"
    "3. **Handling Abbreviations**: Treat abbreviations as synonyms. For example, 'Assistant Store Director,' 'Assistant Store Directors,' "
    "and 'ASD' should be considered synonyms. Use the full, non-abbreviated singular form as the main job title, listing the plural form and abbreviations as synonyms.\n\n"
    "4. **Adding to Taxonomy**:\n"
    "   - **Verbatim Job Title in Taxonomy**: If the exact job title is already in the taxonomy, no action is needed unless the "
    "description needs updating.\n"
    "   - **Similar Description in Taxonomy**: If a similar description exists but under a different job title, determine if it's a "
    "synonym or a distinct job title. Add synonyms accordingly.\n"
    "   - **No Match in Taxonomy**: If there’s no matching job title or description, add the new job title with any synonyms and a "
    "keyword-focused description.\n\n"
    "5. **Description and Synonyms**: Job title descriptions should be concise, focusing on tasks, responsibilities, qualifications, tools and area of work of the job title."
    "Synonyms should be listed using ',' as a separator.\n\n"
    "6. **Special Case - Replacing Plural with Singular**: If the taxonomy contains a plural form as the main job title (e.g., 'Nurses'), "
    "and a singular form (e.g., 'Nurse') is found in a job posting, replace the plural form with the singular in the taxonomy, listing the "
    "plural as a synonym. You will need to first delete the plural job title from the taxonomy and then add the singular form as a new job title in the taxonmy, with the plural form of the job title as a synonym.\n\n"
    "7. **Tool Usage**: Utilize the provided tools effectively for each step, including looking up job titles, finding similar descriptions, "
    "adding new titles, updating synonyms or descriptions, and deleting and replacing job titles as necessary.\n\n"
    "Your task is to process job postings methodically, ensuring the taxonomy remains accurate, clear, and useful. Always prioritize clarity "
    "and precision in categorizing job titles and their descriptions."
)

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
Each row in the taxonomy should be a 'main job title', its synonyms and a description of tasks, responsibilities, qualifications, tools and area of work described commonly related to the job title.
A 'main job title' is the job title that we use to refer to the entire row in the taxonomy. We prefer non-abbreviated singular forms as main job titles.
A synonym is e.g. plural forms of the main job title, e.g. 'nurses' for the main_job_title 'nurse'. Or abbreviations e.g. 'ASD' for 'Assistant Store Director'. 

EXAMPLES
These are examples of work flows and how you use the different tools. Describe you thoughts at each step.

EXAMPLE 1
You are given this (shortned) job posting: 'To pædagoger på 28-30 timer til KKFO'en ved Dyvekeskolen [...]'
1. You extract the following information:
    a. Explicitly stated job title: 'pædagoger' (note, although the job title is plural, if the singular form is not mentioned anywhere in the job posting, you should use the plural form as the main job title in the taxonomy)
    b. Synonyms: NONE
    b. Description: YOU MAKE THIS BASED ON THE POSTING
2. You use the tool 'look_up_main_job_title_in_taxonomy' for checking if a job title is in the taxonomy and find that 'pædagoger' is not in the taxonomy.
    You then use the tool 'search_for_similar_descriptions_in_taxonomy_tool' for searching for similar descriptions. It returns three results (by default). None are similar to the description of the job posting you made.
3. You choose that 'pædagoger' is a new job title that should be added to the taxonomy. You use the tool for adding a job title to the taxonomy to add 'Pædagoger' as a new job title to the taxonomy, along with the synonyms 'Pædagoger' and a combined description.

EXAMPLE 2
You are given this (shortned) job posting: 'Adelphi is seeking a Nurse Practitioner [...]'
1. You extract the following information:
    a. Explicitly stated job title: 'Nurse Practitioner'
    b. Synonyms: NONE
    b. Description: YOU MAKE THIS BASED ON THE POSTING
2. You use the tool for checking if a job title is in the taxonomy and find that 'Nurse Practitioner' is not in the taxonomy. 
   You then use the tool for searching for similar descriptions. It returns three results (by default). One has the main job title 'nurse' and a description that is similar to the description you made.
3. You choose that 'Nurse Practitioner' is a synonym of 'nurse' and use the tool for adding a synonym to a main job title in the taxonomy to add 'Nurse Practitioner' as a synonym to 'nurse'.

EXAMPLE 3
You are given this (shortned) job posting: 'Assistant Store Director (ASD) for [...]'
1. You extract the following information:
    a. Explicitly stated job title: 'Assistant Store Director'
    b. Synonyms: 'ASD'
    b. Description: YOU MAKE THIS BASED ON THE POSTING
2. You use the tool for checking if a job title is in the taxonomy and find that 'Assistant Store Director' is not in the taxonomy. 
   You then use the tool for searching for similar descriptions. It returns three results (by default). None are similar to the description of the job posting you made.
3. You choose that 'Assistant Store Director' is a new job title that should be added to the taxonomy. You use the tool for adding a job title to the taxonomy to add 'Assistant Store Director' as a new job title to the taxonomy, along with the synonyms 'ASD' and the description you made.

EXAMPLE 4
You are given this (shortned) job posting: 'Engageret og faglig pædagog til basisteam i Haraldsgården [...]'
1. You extract the following information:
    a. Explicitly stated job title: 'pædagog'
    b. Synonyms: NONE
    b. Description: YOU MAKE THIS BASED ON THE POSTING
2. You use the tool for checking if a job title is in the taxonomy and find that 'pædagog' is not in the taxonomy. 
   You then use the tool for searching for similar descriptions. It returns three results (by default). One of them 'pædagoger' is the plural form of the job title you found in the job posting.
3. You decide that 'pædagog', as the singular form, is a better choice for the main job title in the taxonomy than 'pædagoger'. You use the tool for deleting a job title in the taxonomy to delete 'pædagoger'. You then add 'pædagog' as a main job title in the taxonomy, along with the synonym 'pædagoger' and a combined description.


EXAMPLE 5
You are given this (shortned) job posting: 'Engageret og faglig pædagog til basisteam i Haraldsgården [...]'
1. You extract the following information:
    a. Explicitly stated job title: 'pædagog'
    b. Synonyms: NONE
    b. Description: YOU MAKE THIS BASED ON THE POSTING
2. You use the tool for checking if a job title is in the taxonomy and find that 'pædagog' is already in the taxonomy. You check the provided description and find that it is accurate. You then conclude the job title is already in the taxonomy and no further action is needed.

EXAMPLE 6
You are given this (shortned) job posting: 'To pædagoger på 28-30 timer til KKFO'en ved Dyvekeskolen [...]. Den ene pædagog skal varetage [...]'
1. You extract the following information:
    a. Explicitly stated job title: 'pædagog' 
    b. Synonyms: 'pædagoger'
    b. Description: YOU MAKE THIS BASED ON THE POSTING
2. You use the tool 'look_up_main_job_title_in_taxonomy' to check if 'pædagog' is in the taxonomy. You can see that it is and the description for this job title is accurate. But there are no synonyms for it.
3. You choose that 'pædagoger' is a synonym for 'pædagog' and should be added. You use the tool for adding a synonym to a main job title in the taxonomy to add 'pædagoger' as a synonym to 'pædagog'.


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
full_text = """
To pædagoger på 28-30 timer til KKFO'en ved Dyvekeskolen
![](https://cdn.ofir.dk/Images/integration/kobenhavnskommunehrmanager/description/da20e8bd-8216-4653-8f57-1b89ea2b2db7.jpeg)

**Vi er en fritidsinstitution ved Dyvekeskolen med børn i alderen 5,5-10 år, i
alt har vi ca. 350 børn, som er lige flyttet i et nyt hus. Vi mangler to
aktive kolleger.**

Vi er en aktiv og dynamisk personalegruppe, der har fuld fart på i vores
arbejde. Vi arbejder i aktivitetsperioder.

**Vi kan tilbyde dig**

  * en ambitiøs og udviklingsorienteret personalegruppe med stort engagement og høj faglighed
  * en dejlig børnegruppe 
  * en velorganiseret afdeling, der har fokus på at arbejde med udvikling og sparring 
  * dygtige og spændende kolleger på alle niveauer 

**Vi har brug for**

  * en der har erfaring indenfor planlægning og daglig koordinering af arbejdsopgaver 
  * en der kan inspirere, motivere og gå forrest 
  * kan skabe resultater, der gør en forskel for børnene
  * en aktiv person, der sætter aktiviteter i gang, lige som vi gør
  * en fagligt velfunderet kollega, der ser (og forstår) børnene og deres forskelligheder/behov
  * en kollega med erfaring, der kan være med til at afholde forældresamtaler
  * en person der kan tilbyde en aktivitet
  * en troværdig kollega, der tager ansvar
  * en person, der kan se muligheder i vores institution – den er stor
  * en udadvendt person, der gør det, du siger, som siger din mening til vores møder
  * at du er initiativrig og sætter aktiviteter i gang med børnene
  * at du kan se de mange forskellige opgaver og påtage dig dem
  * at vi (og børnene) kan regne med dig

Din opgave som pædagog vil også være at have kontakt med skolen og de andre
respektive samarbejdspartnere. En del timer skal lægges i skolen i perioden
august-maj. Din opgave er også at varetage alle opgaverne i forbindelse med
vores børnegruppe, såsom forældrekontakt og indkøringen af børnene når de
starter.

Vi har et godt samarbejde på tværs i hele huset og dette udvikler vi (os med)
løbende. Mener du, at du kan tilbyde os lige dét, vi har brug for, så søg
stillingen.

**Ansættelsesvilkår**  
Løn- og ansættelsesvilkår fastsættes i henhold til kvalifikationer og gældende
overenskomst mellem kommunen og den relevante faglige organisation. I
forbindelse med ansættelsen indhenter vi børne- og straffeattest samt
referencer. Stillingerne er på 28-30 timer om ugen, og ønskes besat snarest
muligt. Stillingerne indebærer en del timer på skolen.

**Yderligere oplysninger**  
Vil du høre mere om stillingerne kan du kontakte pædagogisk leder Snezana
Letic på 2497 1216.

Arbejdstiden ligger mellem 06.30-17.00.

**Søg via linket senest onsdag den 10. januar 2024**  
Ansættelsessamtaler er i januar

"""


agent_executor.invoke({"input": full_text})









# %%
agent_executor.invoke({"input": df.loc[4, 'full_text']})

# %%
TAXONOMY

# %%
agent_executor.invoke({"input": df.loc[1, 'full_text']})

# %%
agent_executor.invoke({"input": df.loc[0, 'full_text']})



# %%

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
print(look_up_main_job_title_in_taxonomy(jt_tax['job_title']))

# %% Check the tool for looking up descriptions in the taxonomy
print(search_for_similar_descriptions_in_taxonomy(jt_tax['description']))


# %% Let's test
data = {
    'explicit_job_title': 'Shipping & Receiving Associate',
    'description': 'Shipping but also a deal of nursery work. And you will be working with a lot of different people.'
    # 'description': 'This job involves packing and preparing customer orders for shipping, receiving material and equipment, loading and unloading trucks, using material handling equipment, inspecting inbound materials, maintaining quality records, communicating with purchasing department, restocking delivery items, ensuring compliance with safety standards, completing paperwork, and maintaining housekeeping standards.'
    }

# %% Check that this new job title does not return anything
print(look_up_main_job_title_in_taxonomy(
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

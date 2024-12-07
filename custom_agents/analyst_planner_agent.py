from typing_extensions import TypedDict
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langgraph.graph import END, StateGraph
import sys
from transformers import AutoTokenizer
#__import__('pysqlite3')
#sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import sqlite3
import time
import os
#from langchain_groq import ChatGroq
from custom_agents.prompt_formatter import PromptFormatter
from custom_agents.base_agent import BaseAgent
#from custom_agents.sql_investment_db_agent import sql_investment_db_agent
#import chromadb
import pandas as pd
from rapidfuzz import process, fuzz



class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: user question
        generation: LLM generation
        context: results from sql db so far
    """
    question : str
    generation : str
    context : str
    num_queries: int 
    num_revisions: int
    analysis_plan: str
    query: str
    generation_log: str
    query_historic: str
    next_action: str
    observations: str
    debug_info: str
    pars: str
    df_results: pd.DataFrame

# In[6]:
class analyst_planner_agent(BaseAgent):
    def __init__(self, llm, tokenizer, planning_llm, long_ctx_llm, log_level='INFO', log_file=None, logging_enabled=True):
        # Call the parent class constructor with all the necessary parameters
        super().__init__(llm, tokenizer, planning_llm, long_ctx_llm, log_level=log_level, log_file=log_file, logging_enabled=logging_enabled)
        

        current_time = time.strftime("%H:%M:%S")
        

        self.analyze_doc_chain = self._initialize_analyze_doc_chain()
        self.extract_pars_chain = self._initialize_extract_pars_chain()
        self.get_category_chain = self._initialize_get_category_chain()

        self.workflow = StateGraph(GraphState)
        self.workflow.add_node("generate_answer", self.generate_answer)
        self.workflow.add_node("execute_plan", self.execute_plan)
        self.workflow.add_node("analyze_doc", self.analyze_doc)
        #self.workflow.add_node("ask_investment_db_expert", self.ask_investment_db_expert)
        self.workflow.add_node("init_agent", self.init_agent)
        #self.workflow.add_node("extract_pars",self.extract_pars)
    
        
        self.workflow.set_entry_point("init_agent")
        
        self.workflow.add_edge("generate_answer", END)
        #self.workflow.add_edge("init_agent","extract_pars")
        #self.workflow.add_edge("extract_pars","analyze_doc")
        self.workflow.add_edge("init_agent","analyze_doc") 
        self.workflow.add_edge("analyze_doc","execute_plan")
        #self.workflow.add_edge("ask_investment_db_expert",END)
        self.workflow.add_edge("execute_plan","generate_answer")

        self.local_agent = self.workflow.compile()

    def _initialize_extract_pars_chain(self):
        extract_pars_formatter = PromptFormatter("Llama3")
        extract_pars_formatter.init_message("")
        extract_pars_formatter.add_message("""You are responsible for extracting parameters out a user question about investment and finance.
The user question is: {question}

PAY ATTENTION to investors or target companies. Please observe and reason first to understand if the company or companies mentioned are investors or target companies. If the company mentioned is doing the investments then is an investor, if the company receives the investments it is a target company. 


# Answer with a JSON dictionary with the following entries (key: possible values):
(PAY SPECIAL ATTENTION TO INVESTOR or TARGET ORGANIZATION NAMES. If the user asks for common deals from two investors, they will be investors.)
selected_investor_names: List with the names of detected investors (ie ["Sequoia Capital","Y Combinator"]). If no investor is specified or detected in the user question the list must be ["ALL"].
selected_target_organization_names: List with the names of detected target organizations (ie ["AirBnb","Cabify"]). If no target organizations names are specified or detected in the user question the list must be ["ALL"].

selected_tags: A list with detected industrial sectors or areas of interest (ie ["Saas","Health Care"]) or [""] if none is specified. This is category of interest of the investor.        
include_all_tags: Does the user wants to do an exclusive search (AND) about the selected_tags? or the user wants to do an inclusive search (OR) about selected_tags? 
selected_countries (["ALL"] by default): A list with the detected countries (ie ["USA","Argentina"]), or a list with one value  = ["ALL"] if no country was detected.
selected_world_region: The name of the detected world region or continent specified by the user (ie africa, middle east, etc) or "ALL" if none is specified
selected_states: a list of two letters US states (ie ["CA","NY"])  or a list with one value  = ["ALL"] if no state was detected
selected_cities: a list with the names of the selected cities (ie ["Berkeley","Buenos Aires"]) or a list with one value  = ["ALL"] if no city was detected
selected_deal_types: a list with the types of series selected (ie ["pre_seed","series_a"]) (valid strings are pre_seed, seed, angel, series_a, series_b...) or a list with one value  = ["ALL"] if no deal type was detected
date_from: initial date for the research ie( "2023-01-01") or "1980-01-01" if none is specified
date_to: end of date range for the query (ie "2023-12-31") or use "2100-12-31"  if none is specified
select_only_lead_investments: "Yes" if the user wants only deals as lead investor, "No" if the user wants only deals as not lead investor, or "ALL" if not specified or to retrieve all the deals
select_only_alone_investments: "Yes" if the user wants only deals with the investor as the only investor in the deal or "No" if not specified (to include all investor counts)

PAY ATTENTION TO DEFAULT VALUES!! specially to selected_countries! if no countries are specified the list must be ["ALL"]!

Answer with a JSON dictionary as specified, don"t do comments, nor greets, nor header or footers.
When generating JSON, use double quotes (") for all strings and array values, instead of simple quotes. This includes JSON keys and all string-type values.
Remember, ALWAYS use double quotes in the JSON, never simple quotes.
""", "system")
        extract_pars_formatter.close_message("assistant")
        extract_pars_prompt = PromptTemplate(
            template=extract_pars_formatter.prompt,
            input_variables=["question"],
        )
        
        return extract_pars_prompt | (lambda x: self.add_tokens_from_prompt(x)) | self.planning_llm | (lambda x: self.add_tokens_from_prompt(x)) | JsonOutputParser()
#################

    def _initialize_get_category_chain(self):
        get_category_formatter = PromptFormatter("Llama3")
        get_category_formatter.init_message("")
        get_category_formatter.add_message("""Your task is to choose the correct industry category from a list given an aproximated input from user.
        
        User category: {orig_category}
        Valid category list: {valid_category_list}
        
        Your choice?
answer in json format with the key chosen_category only with the exact selected value 

""", "system")
        get_category_formatter.close_message("assistant")
        get_category_prompt = PromptTemplate(
            template=get_category_formatter.prompt,
            input_variables=["orig_category","valid_category_list"],
        )
        
        return get_category_prompt | (lambda x: self.add_tokens_from_prompt(x)) | self.planning_llm | (lambda x: self.add_tokens_from_prompt(x)) | JsonOutputParser()


#################
    def _initialize_analyze_doc_chain(self):
        current_time = time.strftime("%H:%M:%S")
        
 
        prompt_planner = """You are a planning agent capable of performing complex operations using basic operations on structured data. You have access to the following data schema, entities, and operations:

### Data Schema and Entities
Album(AlbumId, Title, ArtistId)
  Primary Key: AlbumId
  Foreign Keys:
    - ArtistId -> Artist(ArtistId)

Artist(ArtistId, Name)
  Primary Key: ArtistId

Customer(CustomerId, FirstName, LastName, Company, Address, City, State, Country, PostalCode, Phone, Fax, Email, SupportRepId)
  Primary Key: CustomerId
  Foreign Keys:
    - SupportRepId -> Employee(EmployeeId)

Employee(EmployeeId, LastName, FirstName, Title, ReportsTo, BirthDate, HireDate, Address, City, State, Country, PostalCode, Phone, Fax, Email)
  Primary Key: EmployeeId
  Foreign Keys:
    - ReportsTo -> Employee(EmployeeId)

Genre(GenreId, Name)
  Primary Key: GenreId

Invoice(InvoiceId, CustomerId, InvoiceDate, BillingAddress, BillingCity, BillingState, BillingCountry, BillingPostalCode, Total)
  Primary Key: InvoiceId
  Foreign Keys:
    - CustomerId -> Customer(CustomerId)

InvoiceLine(InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity)
  Primary Key: InvoiceLineId
  Foreign Keys:
    - TrackId -> Track(TrackId)
    - InvoiceId -> Invoice(InvoiceId)

MediaType(MediaTypeId, Name)
  Primary Key: MediaTypeId

Playlist(PlaylistId, Name)
  Primary Key: PlaylistId

PlaylistTrack(PlaylistId, TrackId)
  Primary Key: PlaylistId
  Foreign Keys:
    - TrackId -> Track(TrackId)
    - PlaylistId -> Playlist(PlaylistId)

Track(TrackId, Name, AlbumId, MediaTypeId, GenreId, Composer, Milliseconds, Bytes, UnitPrice)
  Primary Key: TrackId
  Foreign Keys:
    - MediaTypeId -> MediaType(MediaTypeId)
    - GenreId -> Genre(GenreId)
    - AlbumId -> Album(AlbumId)


### Parameters
(No parameters defined at this stage.)

### Reports
(No reports defined at this stage.)

### Reports to get data from source (db) into dataframes
- **Custom Query**:
  - **REPORT_CUSTOM_SQL**: Executes a custom SQL (sqlite3 syntax. Put tables in FROM clause and use WHERE clause to join tables. Dont' use JOIN clause or INNER OR OUTER.) query on the data schema to answer the user question. Parameters: sql_query. Returns: Result of the executed SQL query.

### Operations THAT CAN BE ONLY USED ON dataframes, you need first to get data from db. Use these operations ONLY the data retrieved needed more transformation. If the original query resolves the user question, then no more transformations are needed.
(none by now)

### Task: Generate an action plan (in JSON format) to answer the user question.

The action plan should include:
1. The operations to perform, if applicable.
2. Where to store the intermediate and final results.

### Examples:
Example 1: "Top 10 selling artists"

[
  [OPEN CURLY BRACKET]"step": 1, "action": "REPORT_CUSTOM_SQL", "parameters": [OPEN CURLY BRACKET]"sql_query": "SELECT T1.Name, STRFTIME('%Y', T5.InvoiceDate) AS year, SUM(T4.Quantity) AS total_units FROM Artist T1, Album T2, Track T3, InvoiceLine T4, Invoice T5 WHERE T1.ArtistId = T2.ArtistId AND T2.AlbumId = T3.AlbumId AND T3.TrackId = T4.TrackId AND T4.InvoiceId = T5.InvoiceId GROUP BY T1.Name, STRFTIME('%Y', T5.InvoiceDate) ORDER BY total_units DESC"[CLOSE CURLY BRACKET], "save_as": "df_1"[CLOSE CURLY BRACKET]
]

## LIMIT YOURSELF TO THE GIVEN REPORTS AND OPERATIONS, DO NOT CREATE NEW ONES.
!! ATTENTION DO NOT CREATE NEW FUNCTIONS OR OPERATIONS, USE THE EXISTING ONES !!
        """

        analyze_doc_formatter = PromptFormatter("Llama3")
        analyze_doc_formatter.init_message("")

        analyze_doc_formatter.add_message(prompt_planner, "system")
        analyze_doc_formatter.add_message("""User Question: {question}

        Expert parameters, you can use the following parameters get the correct values for columns especially for categories:
        {expert_pars}
        """, "user")
        analyze_doc_formatter.close_message("assistant")
        analyze_doc_prompt = PromptTemplate(
            template=analyze_doc_formatter.prompt,
            input_variables=["question", "expert_pars"],
        )

        return analyze_doc_prompt | (lambda x: self.add_tokens_from_prompt(x))| self.llm | (lambda x: self.add_tokens_from_prompt(x)) | JsonOutputParser()
 

    def analyze_doc(self,state):
        current_time = time.strftime("%H:%M:%S")
        print("#debug begin analyze doc  ",current_time)
        
        question = state["question"]
        pars = str(state["pars"])
        self.log(f"#Analyze doc planner: {question}/{pars} ", level='DEBUG')
        #print("###93 question ", str(question))
        try:
            analyze_doc_choice = self.analyze_doc_chain.invoke({"question": question,"expert_pars":pars})
        except Exception as e:  
            print("###94 error ", str(e))
        #print("###95 post analyze ", str(analyze_doc_choice))
        self.log(f"#Analyze doc planner choice: {analyze_doc_choice} ", level='DEBUG')
        current_time = time.strftime("%H:%M:%S")
        print("#debug end analyze doc  ",current_time)
        return {"analysis_plan": analyze_doc_choice}
    
    def extract_pars(self,state):
        #print("#debug extract pars")
        current_time = time.strftime("%H:%M:%S")
        print("#debug begin extract pars ",current_time)

        question = state["question"]
        self.log(f"#Before extract pars {question}", level='DEBUG')
        extract_pars_results = self.extract_pars_chain.invoke({"question": question})
        self.log(f"#After extract pars {extract_pars_results}", level='DEBUG')
        pars_final = self.post_process_pars(extract_pars_results)
        self.log(f"#Post process pars {pars_final}", level='DEBUG')
        #print("#debug extract pars fin")
        current_time = time.strftime("%H:%M:%S")
        print("#debug extract pars fin ",current_time)

        return {"pars": pars_final }


    def post_process_pars(self, pars_results):
        print("#87 pp par ", pars_results)
        
        updated_pars = pars_results.copy()

        for key, value in pars_results.items():
            #print(f"key: {key}, value: {value}")

            if key == "selected_tags":
                updated_tags = ['ALL']
                print("#90 selected_tags ", value)
                if value != ['ALL'] and value != [] and value != ['']:
                    updated_tags = []
                    for tag in value:
                        db_category = self.get_category(tag)
                        updated_tags.append(db_category) 
                print("#90 updated tags ", updated_tags)
                updated_pars[key] = updated_tags
            
            elif key == "selected_cities":
                updated_cities = ['ALL']
                if value != ["ALL"] and value != []:
                    updated_cities = []
                    for city in value:
                        city_string = self.get_city(city)
                        updated_cities.append(city_string)
                updated_pars["selected_cities"] = updated_cities
                    
            elif key == "selected_deal_types":
                updated_deal_types = ['ALL']
                if value != ["ALL"] and value != []:
                    updated_deal_types = []
                    for deal_type in value:
                        deal_type_string = self.get_deal_type(deal_type)
                        updated_deal_types.append(deal_type_string)
                updated_pars["selected_deal_types"] = updated_deal_types


            elif key ==  "selected_investor_names":
                updated_investor_names = []
                updated_investor_uuids = []
                if value != ['ALL']:
                    for investor in value:
                        selected_investor_uuid,selected_investor_name = self.get_investor_uuid(investor)
                        updated_investor_names.append(selected_investor_name)
                        updated_investor_uuids.append(selected_investor_uuid)
                else:
                    updated_investor_uuids = ['ALL']
                    updated_investor_names = value
                updated_pars["selected_investor_names"] = updated_investor_names
                updated_pars["selected_investor_uuids"] = updated_investor_uuids
            
            elif key == "selected_countries":
                
                updated_countries = ['ALL']

                if value != ['ALL'] and value != []:
                    updated_countries = []
                    for country in value:
                        country_list = self.get_country(country)
                        updated_countries.append(country_list)
                
                updated_pars["selected_countries"] = updated_countries

            elif key == "selected_world_region":
                if value != "ALL":
                    country_list = self.get_country_list(value)
                    country_list = country_list.split(',')
           
                    
                    updated_pars["selected_countries"] = country_list

            elif key == "selected_target_organization_names":
                updated_organization_names = []
                updated_organization_uuids = []
                if value != ['ALL']:
                    for organization in value:
                        selected_organization_uuid,selected_organization_name = self.get_organization_uuid(organization)
                        updated_organization_names.append(selected_organization_name)
                        updated_organization_uuids.append(selected_organization_uuid)
                else:
                    updated_organization_uuids = ['ALL']
                    updated_organization_names = value
                updated_pars["selected_target_organization_names"] = updated_organization_names
                updated_pars["selected_target_organization_uuids"] = updated_organization_uuids



        print("#87 POST ", pars_results)
        return updated_pars


    def execute_plan(self, state):
        current_time = time.strftime("%H:%M:%S")
        print("#debug begin exe plan db expert ",current_time)

        selected_plan = state["analysis_plan"]
        conn = sqlite3.connect("./Chinook_Sqlite.sqlite")
        cur = conn.cursor()
        df_results = {}
        #print("#78 plan ",str(selected_plan))
        
        #cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        #tables = cur.fetchall()

        # Generar la descripción de cada tabla
        #schema_description = ""
        #for table in tables:
        #    table_name = table[0]
        #    schema_description += f"Table: {table_name}\n"
        #    cur.execute(f"PRAGMA table_info({table_name});")
        #    columns = cur.fetchall()
        #    for column in columns:
        #        schema_description += f"  - {column[1]} ({column[2]})\n"
        #    schema_description += "\n"
        
        #print("#777 schema ", schema_description)


        try:
            for step in selected_plan:
                action = step['action']
                parameters = step['parameters']
                save_as = step['save_as']
                complete_query=""
                current_time = time.strftime("%H:%M:%S")
                print("#debug init action   ",current_time)

                #print("#778 action, parameters, save_as ",action, parameters, save_as)
                self.log(f"#Execute plan, action, parameters, save_as: {action}/{parameters}/{save_as} ", level='DEBUG')
                #print("##77 len df_results ", len(df_results))
                #print("#779") 
                if action.startswith('REPORT'):
                    sql_query = "SELECT 0"
                    #print("#779.a")
                    if action == 'REPORT_CUSTOM_SQL':
                        #print("#779.b get complete query ")
                        sql_query = parameters['sql_query']
                    
                    #print("#780")        

                    #complete_query = self.get_complete_report_query_string(action, parameters)
                    #print("#79 complete query ", sql_query)
                    cur.execute(sql_query)
                    #print("#80 execute  ")
                    results = cur.fetchall()
                    columns = [desc[0] for desc in cur.description]
                    df = pd.DataFrame(results, columns=columns)
                    df_results[save_as] = df

                elif action.startswith('OP'):
                    if action == 'OP_SELECT_COLUMNS':
                        df_a = df_results[parameters['df_a']]
                        #print("#84 ", df_a.columns)
                        columns = parameters['list_of_columns']
                        df = df_a[columns]
                        df_results[save_as] = df

                    elif action == 'OP_INTERSECT':
                        df_a = df_results[parameters['df_a']]
                        df_b = df_results[parameters['df_b']]
                        print("#75 ", len(df_a), len(df_b))
                        df_a = df_a.applymap(str)
                        df_b = df_b.applymap(str)
                        df = pd.merge(df_a, df_b, how='inner')
                        df_results[save_as] = df
                        print("#76 ", len(df))

                    elif action == 'OP_DISTINCT':
                        df_a = df_results[parameters['df_a']]
                        df = df_a.drop_duplicates().reset_index(drop=True)
                        df_results[save_as] = df
                    elif action == 'OP_EXCEPTION':
                        reason = parameters['reason']
                        data = {'reason': [reason]}
                        df = pd.DataFrame(data)
                        df_results[save_as] = df


                    elif action == 'OP_LIMIT':
                        df_a = df_results[parameters['df_a']]
                        limit = parameters['limit']
                        df = df_a.head(limit)
                        df_results[save_as] = df

                    elif action == 'OP_ORDER':
                        df_a = df_results[parameters['df_a']]
                        columns = parameters['columns']
                        #print("#99 df_a cols ",df_a.columns)
                        ascending = parameters.get('ascending', "True")
                        ascending_bool = (ascending == 'True') 
                        df = df_a.sort_values(by=columns, ascending=ascending_bool)
                        df_results[save_as] = df

                    elif action == 'OP_GROUP_BY':
                        df_a = df_results[parameters['df_a']]
                        group_by_columns = parameters['group_by_columns']
                        
                        # Convertir la lista de funciones de agregación a un formato adecuado
                        agg_functions = {item['column']: item['function'] for item in parameters['agg_functions']}
                        
                        # Aplicar el groupby y la agregación
                        df = df_a.groupby(group_by_columns).agg(agg_functions).reset_index()
                        
                        # Renombrar las columnas agregadas
                        new_column_names = {item['column']: f"{item['function']}_{item['column']}" for item in parameters['agg_functions']}
                        df = df.rename(columns=new_column_names)
                        
                        # Guardar el resultado en df_results
                        df_results[save_as] = df

                    elif action == 'OP_FILTER':
                        df_a = df_results[parameters['df_a']]
                        condition = parameters['condition']
                        df = df_a.query(condition)
                        df_results[save_as] = df

                    elif action == 'OP_JOIN':
                        df_a = df_results[parameters['df_a']]
                        df_b = df_results[parameters['df_b']]
                        on = parameters['on']
                        how = parameters.get('how', 'inner')
                        
                        
                        df = pd.merge(df_a, df_b, on=on, how=how, suffixes=('', '_y'))
                        
                        # Procesar las columnas resultantes
                        columns_to_drop = []
                        columns_to_rename = {}
                        
                        for col in df.columns:
                            if col.endswith('_y'):
                                base_col = col[:-2]
                                if base_col in df.columns:
                                    columns_to_drop.append(col)
                                else:
                                    columns_to_rename[col] = base_col
                        
                        # Aplicar los cambios
                        df.drop(columns=columns_to_drop, inplace=True)
                        df.rename(columns=columns_to_rename, inplace=True)
                        
                        df_results[save_as] = df

                    elif action == 'OP_CREATE_DF_VALUE':
                        value = parameters['value']
                        column_name = parameters.get('column_name', 'value')
                        
                        df = pd.DataFrame({column_name: [value]})
                        df_results[save_as] = df

                    elif action == 'OP_COUNT_AGG':
                        df_a = df_results[parameters['df_a']]
                        print("#98 ",len(df_a))
                        count_col = parameters['count_column']
                        new_col_name = f'count_{count_col}'
                        df = pd.DataFrame({new_col_name: [df_a[count_col].count()]})
                        df_results[save_as] = df
                        print("#99 ",len(df))

                    elif action == 'OP_SUM_AGG':
                        df_a = df_results[parameters['df_a']]
                        sum_col = parameters['column']
                        new_col_name = f'sum_{sum_col}'
                        df = pd.DataFrame({new_col_name: [df_a[sum_col].sum()]})
                        df_results[save_as] = df

                    elif action == 'OP_AVERAGE_AGG':
                        df_a = df_results[parameters['df_a']]
                        avg_col = parameters['column']
                        new_col_name = f'avg_{avg_col}'
                        df = pd.DataFrame({new_col_name: [df_a[avg_col].mean()]})
                        df_results[save_as] = df

                    elif action == 'OP_MIN_AGG':
                        df_a = df_results[parameters['df_a']]
                        min_col = parameters['column']
                        new_col_name = f'min_{min_col}'
                        df = pd.DataFrame({new_col_name: [df_a[min_col].min()]})
                        df_results[save_as] = df

                    elif action == 'OP_MAX_AGG':
                        df_a = df_results[parameters['df_a']]
                        max_col = parameters['column']
                        new_col_name = f'max_{max_col}'
                        df = pd.DataFrame({new_col_name: [df_a[max_col].max()]})
                        df_results[save_as] = df


                    elif action == 'OP_RENAME_COLUMNS':
                        df_a = df_results[parameters['df_a']]
                        columns = parameters['columns']
                        df = df_a.rename(columns=columns)
                        df_results[save_as] = df

                    elif action == 'OP_APPLY':
                        df_a = df_results[parameters['df_a']]
                        function = parameters['function']
                        axis = parameters.get('axis', 0)
                        df = df_a.apply(function, axis=axis)
                        df_results[save_as] = df

                    elif action == 'OP_DIVIDE_DFS':
                        df_a = df_results[parameters['df_a']]
                        df_b = df_results[parameters['df_b']]
                        column_a = parameters.get('column_a', df_a.columns[0])
                        column_b = parameters.get('column_b', df_b.columns[0])
                        
                        result = df_a[column_a].iloc[0] / df_b[column_b].iloc[0]
                        df_result = pd.DataFrame({f'result': [result]})
                        
                        df_results[save_as] = df_result

                    elif action == 'OP_MULTIPLY_DFS':
                        df_a = df_results[parameters['df_a']]
                        df_b = df_results[parameters['df_b']]
                        column_a = parameters.get('column_a', df_a.columns[0])
                        column_b = parameters.get('column_b', df_b.columns[0])
                        
                        result = df_a[column_a].iloc[0] * df_b[column_b].iloc[0]
                        df_result = pd.DataFrame({f'result': [result]})
                        
                        df_results[save_as] = df_result

                    elif action == 'OP_SUBSTRACT_DFS':
                        df_a = df_results[parameters['df_a']]
                        df_b = df_results[parameters['df_b']]
                        column_a = parameters.get('column_a', df_a.columns[0])
                        column_b = parameters.get('column_b', df_b.columns[0])
                        
                        result = df_a[column_a].iloc[0] - df_b[column_b].iloc[0]
                        df_result = pd.DataFrame({f'result': [result]})
                        
                        df_results[save_as] = df_result

                    elif action == 'OP_ADD_DFS':
                        df_a = df_results[parameters['df_a']]
                        df_b = df_results[parameters['df_b']]
                        column_a = parameters.get('column_a', df_a.columns[0])
                        column_b = parameters.get('column_b', df_b.columns[0])
                        
                        result = df_a[column_a].iloc[0] + df_b[column_b].iloc[0]
                        df_result = pd.DataFrame({f'result': [result]})
                        
                        df_results[save_as] = df_result
                current_time = time.strftime("%H:%M:%S")
                print("#debug end action   ",current_time)

        except Exception as e:
            conn.close()
            print("#SQL CB AG -", str(e), "****SQL***\n", complete_query)
            final_results = "It was not possible to retrieve the data."
            return {"next_action": "END", "generation": final_results}

        conn.close()
        # Assuming the last step's result is the final result
        final_step = selected_plan[-1]
        print("#07 ", final_step['save_as'])
        
        final_df = df_results[final_step['save_as']]
        #print("#08 ", len(final_df))
        #print("#09 ", final_df.head())
        current_time = time.strftime("%H:%M:%S")
        print("#debug end plan   ",current_time)
        final_results = self.generate_text_from_dataframe(final_df)
        current_time = time.strftime("%H:%M:%S")
        print("#debug end x plan after gen from db   ",current_time)

        return {"next_action": "END", "generation": final_results, "df_results": final_df}

    def generate_text_from_dataframe(self, df):
        top_results = 15
        limit_chars = 400

        if df.empty:
            return "Expert says there is no data available about that question."
            
        text_results = "* Answer from expert:\n"

        # Limit to the first k results if there are more than k rows
        if len(df) > top_results:
            df = df.head(top_results)
            text_results += "(results limited to first {} rows. Here you have a limited list, please use with caution because it's incomplete, inform the user about that and don't use the following information to do answer follow up questions, instead re write your question)\n".format(top_results)
        
        # Add column names
        column_names_text = " | ".join(df.columns)
        text_results += f"* Columns of the report: {column_names_text}\nData from the report:\n"

        # Create a string with results and column names
        for _, row in df.iterrows():
            row_text = " | ".join(str(val) for val in row)
            if len(row_text) > limit_chars:
                row_text = row_text[:limit_chars] + " (results trimmed to {} characters)".format(limit_chars)
            text_results += f"{row_text}\n"

        return text_results

    def verify_query(self,state):
        next_action = state["next_action"]
        if next_action == 'END':
            return END
        else:
            return next_action
    
    def generate_answer(self,state):
        return 
    

    
    def verify_metric_selection(self,state):
        print("#debug verify metric selection")
        selected_metric = state["analysis_plan"]
        if selected_metric == 'CUSTOM_METRIC':
            return "ask_investment_db_expert"
        else:
            return "query_metrics_db"



    def init_agent(self,state):
        
        current_time = time.strftime("%H:%M:%S")
        print("#debug begin init db ",current_time)
        self.log("#10 analyst init", level='INFO')
        self.reset_token_counter()
        return {"num_queries": 0,"query_historic":"","context":"","next_action":"","debug_info":"","pars":"", "df_results": pd.DataFrame() }
        current_time = time.strftime("%H:%M:%S")
        print("#debug end init db ",current_time)
 

    def ask_question(self, par_state):
        # Invoke the agent once and store the result
        result = self.local_agent.invoke(par_state)
        print("#007 db expert ask question par state ", str(par_state))
        # Extract the needed fields from the result
        answer = result['generation']
        df_res = result['df_results']
        
        return answer, df_res

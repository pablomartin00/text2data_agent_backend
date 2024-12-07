from typing_extensions import TypedDict
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langgraph.graph import END, StateGraph
import sys
from transformers import AutoTokenizer
import os 
import subprocess
import urllib.parse
import os
#from langchain_groq import ChatGroq
from custom_agents.prompt_formatter import PromptFormatter
from custom_agents.analyst_planner_agent import analyst_planner_agent
from custom_agents.web_planner_agent import web_planner_agent
from custom_agents.base_agent import BaseAgent
#from custom_agents.graphrag_notes_agent import graphrag_notes_agent
from custom_agents.meeting_agent import meeting_agent
import pandas as pd


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: user question
        generation: LLM generation
        context: results from semantic db so far
    """
    question : str
    generation : str
    context : str
    num_queries: int 
    num_revisions: int
    analysis_choice: str
    query: str
    generation_log: str
    query_historic: str
    next_action: str
    observations: str
    messages: list
    information: str
    internal_message: str
    df_last_result: pd.DataFrame
    

# In[6]:
class chat_planner_agent(BaseAgent):
    def __init__(self, llm, tokenizer, planning_llm, long_ctx_llm, log_level='INFO', log_file=None, logging_enabled=True):
        # Call the parent class constructor with all the necessary parameters
        super().__init__(llm, tokenizer, planning_llm, long_ctx_llm, log_level=log_level, log_file=log_file, logging_enabled=logging_enabled)
        
        self.generate_answer_chain = self._initialize_generate_answer_chain()
        self.analyze_user_question_chain = self._initialize_analyze_user_question_chain()
        self.generate_web_question_chain = self._initialize_generate_web_question_chain()
        self.analyst_planner_expert = analyst_planner_agent(self.llm, self.tokenizer , self.planning_llm, self.long_ctx_llm, log_level=log_level, log_file='./agentlogs/analyst_planner.txt', logging_enabled=True )
        self.web_planner_expert = web_planner_agent(self.llm, self.tokenizer , self.planning_llm, self.long_ctx_llm, log_level=log_level, log_file='./agentlogs/web_planner.txt', logging_enabled=True)
        #self.graphrag_notes_expert = graphrag_notes_agent(self.llm , self.tokenizer, log_level=log_level, log_file='./agentlogs/rag_agent.txt', logging_enabled=True)
        self.meeting_expert = meeting_agent(self.llm, self.tokenizer , self.planning_llm, self.long_ctx_llm , log_level=log_level, log_file='./agentlogs/meeting_assistant.txt', logging_enabled=True)
        
        self.workflow = StateGraph(GraphState)

        self.workflow.add_node("init_agent", self.init_agent)
        self.workflow.add_node("analyze_user_question", self.analyze_user_question)
        self.workflow.add_node("answer_user", self.answer_user)
        self.workflow.add_node("execute_plan", self.execute_plan)

        self.workflow.set_entry_point("init_agent")
        
        self.workflow.add_edge("answer_user", END)
        self.workflow.add_edge("init_agent","analyze_user_question")
        self.workflow.add_edge("analyze_user_question","execute_plan")
        #self.workflow.add_conditional_edges("analyze_user_question",self.analysis_router)
        self.workflow.add_edge("execute_plan","answer_user")
        #self.workflow.add_edge("ask_web_expert","answer_user")
        #self.workflow.add_edge("reject_question","answer_user")
        
        #self.df_last_result = pd.DataFrame()
        ####
        
        self.local_agent = self.workflow.compile()

        self.meeting_data = self.get_meeting_data()
    
    def get_meeting_data(self):
        # Create test data
        data = {
            'meeting_id': [1, 2, 3],
            'target_company_name': ['Company A', 'Company B', 'Company C'],
            'target_company_uuid': ['uuid-1234', 'uuid-5678', 'uuid-91011'],
            'subject' : ['Presentation of CRM tool','Evaluate investment opportunities','Corporate introduction'],
            'participants': [
                [
                    {"participant_name": "Alice Smith", "participant_email": "alice.smith@example.com", "participant_company_name": "Company X", "participant_company_uuid": "uuid-1111"},
                    {"participant_name": "Bob Johnson", "participant_email": "bob.johnson@example.com", "participant_company_name": "Company Y", "participant_company_uuid": "uuid-2222"}
                ],
                [
                    {"participant_name": "Carol White", "participant_email": "carol.white@example.com", "participant_company_name": "Company Z", "participant_company_uuid": "uuid-3333"},
                    {"participant_name": "David Brown", "participant_email": "david.brown@example.com", "participant_company_name": "Company W", "participant_company_uuid": "uuid-4444"}
                ],
                [
                    {"participant_name": "Eve Black", "participant_email": "eve.black@example.com", "participant_company_name": "Company V", "participant_company_uuid": "uuid-5555"},
                    {"participant_name": "Frank Green", "participant_email": "frank.green@example.com", "participant_company_name": "Company U", "participant_company_uuid": "uuid-6666"}
                ]
            ]
        }

        # Create the DataFrame with the test data
        meeting_data = pd.DataFrame(data)

        return meeting_data

    def execute_plan(self, state):
        selected_plan = state["analysis_choice"]
        df_last_result = state["df_last_result"]
        #print("#008 analysis choice selected plan in execute plan (chat): ", str(selected_plan))
        try:
            dict_results = {}
            for step in selected_plan:
                action = step['action']
                parameters = step['parameters']
                save_as = step['save_as']
                complete_query=""
                
                self.log(f"#Execute plan, action, parameters, save_as: {action}/{parameters}/{save_as} ", level='DEBUG')

                if action == "ask_db_expert":
                    query = parameters.get('query', "")
                    context_datastore = parameters.get('context_data',"")
                    context_data = ""
                    
                    if context_datastore != "":
                        context_data = dict_results[context_datastore]
                
                    #print("#99 ask db expert query ",query)
                    result,df_last_result = self.ask_db_expert(query)
                    print("#100 result from db ",query, result, len(df_last_result ))
                    dict_results[save_as] = df_last_result
                elif action == "quick_answer":
                    query = parameters.get('query', "")
                    result = query

                    dict_results[save_as] = result
                elif action == "ask_web_expert":
                    query = parameters.get('query', "")
                    context_datastore = parameters.get('context_data',"")
                    context_data = ""
                    #print("#04")
                    if context_datastore != "":
                        print("#04.a")
                        context_data = dict_results[context_datastore]
                        print("#04.b")
                        query += f"\n extra data: {context_data}"
                    
                   # encoded_query_string = urllib.parse.quote(query)
                    #print("#05 ecnoded ", query)
                    #encoded_query_string = "information about Y Combinator"
                    result = self.ask_web_expert(query)
                    dict_results[save_as] = result
                elif action == "define_global_query":
                    global_query = parameters.get('query', "")
                     
                elif action == "ask_notes_expert":
                    query = parameters.get('query', "")
                    context_datastore = parameters.get('context_data',"")
                    context_data = ""
                    print("#07")
                    if context_datastore != "":
                        print("#07.a")
                        context_data = dict_results[context_datastore]
                        #print("#07.b")
                        #query += f"\n extra data: {context_data}"
                    
                   # encoded_query_string = urllib.parse.quote(query)
                    print("#05 ecnoded ", query)
                    #encoded_query_string = "information about Y Combinator"
                    result = self.ask_notes_expert(query)
                    dict_results[save_as] = result
                elif action == 'more_rows':
                    row_from = int(parameters.get('row_from', 0))
                    row_to = int(parameters.get('row_to', row_from + 10))
                    print("#21 len df last result ",len(df_last_result))
                    results = self.get_more_rows(df_last_result, row_from, row_to)
                    dict_results[save_as] = results
                elif action == 'prepare_my_meeting_assistant':
                    #'participant_names_list', 'company_names_list','participant_emails_list' and 'company_uuids_list'
                    meeting_id = parameters.get('meeting_id', "0")
                    results = self.ask_meeting_assistant(meeting_id)
                    print("#68 meeting assistant results ",results)
                    dict_results[save_as] = results
        except Exception as e:
            print("#error chat planner execute plan ", str(e))
            final_results = "It was not possible to elaborate an answer"
            return {"information": final_results}
        print("#11 ")
        # Assuming the last step's result is the final result
        final_step = selected_plan[-1]
        
        final_results = dict_results[final_step['save_as']]
        print("#11 ",type(final_results), final_results)
        return {"information": final_results,"query": global_query,"df_last_result": df_last_result}

    def _initialize_generate_web_question_chain(self):
        generate_answer_formatter = PromptFormatter("Llama3")
        generate_answer_formatter.init_message("")
        generate_answer_formatter.add_message("""You are an expert in generating web requests for a web search.
        Here you have a question and information in context but they may be messy an desordered.
        Refactor the user question into a complete web search, return only a string with the web search, no commentaries or explanations.
        """, "system")
        
        generate_answer_formatter.add_raw("user question and data: {question}\n")
        generate_answer_formatter.close_message("assistant")

        generate_answer_prompt = PromptTemplate(
            template=generate_answer_formatter.prompt,
            input_variables=["question"],
        )
        return generate_answer_prompt| (lambda x: self.add_tokens_from_prompt(x)) | self.llm | (lambda x: self.add_tokens_from_prompt(x)) | StrOutputParser()


    def _initialize_generate_answer_chain(self):
        generate_answer_formatter = PromptFormatter("Llama3")
        generate_answer_formatter.init_message("")
        generate_answer_formatter.add_message("""You are an AI assistant chat for an app that helps user to answer questions on custom private music sales data. 
            Try to continue the dialog between the user giving fluidity to the conversation.
            To answer the user take into account the conversation context and the following context data provided by our experts to a related question:

            Expert results: {information}

            Answer the user question with the provided data in context. 
            If there no  results just tell the user you don't know.
            Provide only information related to the user question in a friendly and professional way.
            Use and respect the report format if you receive and answer from the meeting preparation assistant.
            For other database results, a tabular format may be better to present the results.
            Use report format if you receive and answer from the meeting preparation assistant, ATTENTION TO THAT.
        """, "system")
        
        generate_answer_formatter.add_raw("{messages}\n")
        generate_answer_formatter.close_message("assistant")

        generate_answer_prompt = PromptTemplate(
            template=generate_answer_formatter.prompt,
            input_variables=["information","messages","query"],
        )
        return generate_answer_prompt| (lambda x: self.add_tokens_from_prompt(x)) | self.llm | (lambda x: self.add_tokens_from_prompt(x)) | StrOutputParser()

    def _initialize_analyze_user_question_chain(self):
        analyze_user_question_formatter = PromptFormatter("Llama3")
        analyze_user_question_formatter.init_message("")
        analyze_user_question_formatter.add_message("""You are an AI assistant chat for an app that helps user to answer questions on custom private music sales data.
            Given the following conversation between the user and the assistant you must determine an action plan.
            ### Task: Generate an action plan (in json format) to answer the user question.
 
            The action plan should include steps:
            1.define_global_query to understand the main task you identified (MANDATORY AS FIRST STEP!)
            2.the rest of the steps to solve the user question
            
            Each step includes:
            2. The tool to be used.
            2. The parameters needed for the tool (context_data is the label for a local store that you can used )
            3. Where to store the intermediate and final results (this tag will be used latter to retrieve the data in next steps)
            
            The user's question may require retrieving more information from their internal structured database, from a web search, or from their internal notes, or, if possible, answer them directly.

            You have the following tools:

            ******* 'define_global_query': This is the first step of the plan, explaining the global question to resolve.

            ******* 'ask_db_expert': Ask our structured database expert for information required to satisfy the user's question. 
You will also write the question in natural english language for our database expert to search for information.
Sometimes you will need to ask the expert almost the same thing that the user asks (e.g., "give me data on artist sales trends").
In some cases, the user may ask a follow-up question (e.g., "Why is that?") so you will need to understand the question and ask our expert a full question with context, adding all necessary information to retrieve data.
The expert does not perform follow-ups, so don’t include previous results in the expert question, as it cannot use that data.
For example, if the user references a previous answer and wants more information, you will need to ask our expert by creating a complete question since our expert does not remember the conversation.
Take into account the data schema and basic entities about structured data:
                ### Data schema and basic entities for you to have an idea:
                Album(AlbumId, Title, ArtistId)
                Artist(ArtistId, Name)
                Customer(CustomerId, FirstName, LastName, Company, Address, City, State, Country, PostalCode, Phone, Fax, Email, SupportRepId)
                Employee(EmployeeId, LastName, FirstName, Title, ReportsTo, BirthDate, HireDate, Address, City, State, Country, PostalCode, Phone, Fax, Email)
                Genre(GenreId, Name)
                Invoice(InvoiceId, CustomerId, InvoiceDate, BillingAddress, BillingCity, BillingState, BillingCountry, BillingPostalCode, Total)
                InvoiceLine(InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity)
                MediaType(MediaTypeId, Name)
                Playlist(PlaylistId, Name)
                PlaylistTrack(PlaylistId, TrackId)
                Track(TrackId, Name, AlbumId, MediaTypeId, GenreId, Composer, Milliseconds, Bytes, UnitPrice)
                                
            ******* 'quick_answer' if you think you have enough information to answer. Also this can be the case if the user asks for a follow up question about the current topic and you can answer with current data but always is a priority to ask the expert, do not skip the experts unless is too obvious. 
            If the user is asking something that needs search, first ask the expert. You can use this options to reject a questionif it’s off-topic or unrelated to music sales data. Use query key to provide your feedback.
            IF the user asks something illegal you must answer with quick_answer and use the key query to explain why you reject the question, always in JSON!!! otherwise you will induce an error.

            ******* 'ask_web_expert': Ask our web expert to do a research the internet for additional or current information. Use this when database information is insufficient or potentially outdated. Formulate a clear, specific query for web research. 

            ******* 'ask_notes_expert': Ask our notes expert to do a research within the internal user notes, this is vital source of internal information, if the user wants an internal search, use this option. 
            Formulate a clear, specific query for our notes expert, he will do research and elaborate an answer. 

            ******* 'more_rows': if the user asks for more rows from a preceding query, you can ask directly the dataframe that contains the full results from the last database question answered by db expert.
            It takes as parameters: 'row_from' and 'row_to' to indicate which rows to get. Maximum 20 rows at a time. Instead of doing the query again, you can use this option for performance.

            ******* 'prepare_my_meeting_assistant': if the user asks to prepare a meeting with meeting id use this assistant.
            It takes as parameters 'meeting_id' which is the identification that you must pass to the assistant 

### Here are some examples:

example 1: "Latest news about Spotify?", would require a web search and answer.
[
  [OPEN BRACKET]"step": 1, "action": "define_global_query", "parameters": [OPEN BRACKET]"query": "Latest news about target organization Spotify","context_data":""[CLOSING BRACKET], "save_as": "response_1"[CLOSING BRACKET],
  [OPEN BRACKET]"step": 2, "action": "ask_web_expert", "parameters": [OPEN BRACKET]"query": "Latest news about target organization Spotify","context_data":""[CLOSING BRACKET], "save_as": "response_2"[CLOSING BRACKET],
]

example 2: "In which countries is the artist A's music most popular in 2023?", would require asking our db expert and answering the user.
[
  [OPEN BRACKET]"step": 1, "action": "define_global_query", "parameters": [OPEN BRACKET]"query": "In which countries is artist A's music most popular in 2023?","context_data":""[CLOSING BRACKET], "save_as": "response_1"[CLOSING BRACKET],
  [OPEN BRACKET]"step": 2, "action": "ask_db_expert", "parameters": [OPEN BRACKET]"query": "In which countries is artist A's music most popular in 2023?","context_data":""[CLOSING BRACKET], "save_as": "response_2"[CLOSING BRACKET],
]

example 3: "Latest news on the genres popular in each country where artist A's music was most popular in 2023?", would require asking the db expert, then performing a web search, and finally answering the user.
[
  [OPEN BRACKET]"step": 1, "action": "define_global_query", "parameters": [OPEN BRACKET]"query": "Latest news on the genres popular in each country where artist A's music was most popular in 2023?","context_data":""[CLOSING BRACKET], "save_as": "response_1"[CLOSING BRACKET],
  [OPEN BRACKET]"step": 2, "action": "ask_db_expert", "parameters": [OPEN BRACKET]"query": "In which countries is artist A's music most popular in 2023?","context_data":""[CLOSING BRACKET], "save_as": "response_2"[CLOSING BRACKET],
  [OPEN BRACKET]"step": 3, "action": "ask_web_expert", "parameters": [OPEN BRACKET]"query": "Latest music genre news in context countries","context_data":"response_1"[CLOSING BRACKET], "save_as": "response_3"[CLOSING BRACKET],
]

example 4: (after an initial list of tracks) "show me more tracks".
[
  [OPEN BRACKET]"step": 1, "action": "define_global_query", "parameters": [OPEN BRACKET]"query": "get rows from stored data from 15 to 30","context_data":""[CLOSING BRACKET], "save_as": "response_1"[CLOSING BRACKET],
  [OPEN BRACKET]"step": 2, "action": "more_rows", "parameters": [OPEN BRACKET]"row_from": 15,"row_to": 30, "context_data":""[CLOSING BRACKET], "save_as": "response_2"[CLOSING BRACKET],
]

### END OF EXAMPLES

 
No premable or explanation. No comments, no introductions, no explanations, only a JSON as explained! 
            ALWAYS ANSWER WITH JSON!!

        """, "system")
        

        analyze_user_question_formatter.add_raw("{messages}\n")
        analyze_user_question_formatter.close_message("assistant")
        analyze_user_question_prompt = PromptTemplate(
            template=analyze_user_question_formatter.prompt,
            input_variables=["messages"],
        )
        
        return analyze_user_question_prompt | (lambda x: self.add_tokens_from_prompt(x))| self.llm | (lambda x: self.add_tokens_from_prompt(x))| JsonOutputParser()





    def analyze_user_question(self, state):
        """
        analyze doc

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        messages = state['messages']
        message_formatter = PromptFormatter("Llama3")
        for message in messages:
            message_formatter.add_message(message['content'],message['role'])

        self.log(f"#Chat planner messages: {message_formatter.prompt}", level='DEBUG')
        analysis_choice = self.analyze_user_question_chain.invoke({"messages": message_formatter.prompt})
        #chat_planner_choice = analysis_choice["choice"]
        #chat_planner_query = analysis_choice["query"]

        self.log(f"#Chat planner choice: {analysis_choice}", level='DEBUG')
        #return {"analysis_choice": chat_planner_choice,"query": chat_planner_query}
        return {"analysis_choice": analysis_choice}


    # In[90]:

    def get_more_rows(self, df, row_from, row_to):
        """
        Fetch a subset of rows from a DataFrame within the specified range.
        
        :param df: DataFrame from which rows are to be fetched
        :param row_from: Starting index for row selection (inclusive)
        :param row_to: Ending index for row selection (exclusive)
        :return: DataFrame containing the specified rows
        """
        try:
            # Ensure row_from and row_to are within bounds
            row_from = max(row_from, 0)
            row_to = min(row_to, len(df))
            print("#31 len df orig ",len(df) )
            # Fetch the rows
            result = df.iloc[row_from:row_to]
            print("#32 len df result ",len(result) )
            text_result = self.generate_text_from_dataframe(result) 
            print("#33 text result ", text_result)
            return text_result
        except Exception as e:
            print(f"An error occurred while fetching rows: {e}")
            return pd.DataFrame()  # Return empty DataFrame on error

    def generate_text_from_dataframe(self, df):
        top_results = 100
        limit_chars = 400

        if df.empty:
            return "Expert says there is no data available about that question."
            
        text_results = "* Answer from expert:\n"

        # Limit to the first k results if there are more than k rows
        if len(df) > top_results:
            df = df.head(top_results)
            text_results += "(Here you have a limited list, please use with caution because it may be incomplete, inform the user about that )\n".format(top_results)
        
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

    def answer_user(self,state):
        """
        Generate answer

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        
        messages = state['messages']
        information = state['information']
        query = state['query']
        print("#001 answere user", query)
        message_formatter = PromptFormatter("Llama3")
        for message in messages:
            message_formatter.add_message(message['content'],message['role'])
        # Answer Generation
        #print("#Chat agent - answer user - information ", information)
        print("#002 query answer information ", information)
        generation = self.generate_answer_chain.invoke({"information": information, "messages": message_formatter.prompt,"query":query})
        print("#003 query answer ")
        self.log(f"#Assistant output: {generation}", level='DEBUG')
        return {"generation": generation}


    def init_agent(self,state):
        #yield {"internal_message: ": "#12 init chat agent"}
        #if self.st_session:
            #self.st_session.write("#####Init chat agent")
            #print("#13 init chat agent st write")
        try:
            df_last_result = state["df_last_result"]
        except Exception as e:
            self.log(f"#201 init agent chat error **", str(e))
            df_last_result = pd.DataFrame()

        self.log(f"#200 init agent chat **", level='DEBUG')    
        print("#200 init agent chat **")
        return {"num_queries": 0,"context":"","next_action":"","information":"","query":"","df_last_result":df_last_result}



    def ask_db_expert(self,question):

        #question = state['query']
        #print("#00 before ask question")
        self.log("#Ask expert ", level='DEBUG')
        
        

        information,df_db_result = self.analyst_planner_expert.ask_question({"question":question})
        self.log(f"#Expert output: {information}", level='DEBUG')

        self.add_token_amount(self.analyst_planner_expert.tokenCounter)

        return  information,df_db_result


    def get_meeting_info_by_id(self, meeting_id):
        # Find the row corresponding to the meeting_id
        meeting_row = self.meeting_data[self.meeting_data['meeting_id'] == meeting_id]
        if not meeting_row.empty:
            # Convert the row to a dictionary (taking the first found row in case of multiple matches)
            meeting_info = meeting_row.iloc[0].to_dict()
            return meeting_info
        else:
            print(f"Meeting with id {meeting_id} not found.")
            return None



    def ask_meeting_assistant(self,meeting_id):
        self.log("#Ask meeting assistant ", level='DEBUG')
        print("#34")
        meeting_info = self.get_meeting_info_by_id(meeting_id)
        print("#35")
        information = ""
        if meeting_info:
            information = self.meeting_expert.ask_question({"meeting_info":meeting_info})
        print("#36")
        self.log(f"#Expert output: {information}", level='DEBUG')

        self.add_token_amount(self.meeting_expert.tokenCounter)

        return  information

    def ask_web_expert(self, question):

        #question = state['query']
        #print("#00 before ask question")
        self.log("#Ask web expert ", level='DEBUG')
        #print("#07 ", question)
        question_corrected = self.generate_web_question_chain.invoke({"question": question})
        #print("#08 ", question_corrected)
        information = self.web_planner_expert.ask_question({"question":question_corrected})
        #print("#09 ", information)
        #output = "dow jones has increased 10% in one day due to new US regulation in july 2024"
        self.log(f"#Web Expert output: {information}", level='DEBUG')

        return information




    def ask_notes_expert(self, question):

        #question = state['query']
        #print("#00 before ask question")
        self.log("#Ask web expert ", level='DEBUG')
        #print("#07 ", question)
        #question_corrected = self.generate_web_question_chain.invoke({"question": question})
        #print("#08 ", question)
        #information = self.graphrag_notes_expert.ask_question({"question":question})
        #information = self.web_planner_expert.ask_question({"question":question_corrected})
        information = "Notes not available yet"
        #print("#09 ", information)
        #output = "dow jones has increased 10% in one day due to new US regulation in july 2024"
        self.log(f"#Web Expert output: {information}", level='DEBUG')

        return information

    def analysis_router(self,state):
        analysis_choice = state["analysis_choice"]
        
        return analysis_choice

    def reject_question(self,state):
        information = "The user question is not about investments, finance or related topics. reject politely."
        return {"information": information}

    def check_retrieval(self,state):
        next_action = state["next_action"]


        return next_action
    
    def ask_question(self, par_state):
        #self.st_session = par_st
        #yield "Initiating task..."

        #answer = self.local_agent.invoke(par_state)
        #for step in answer:
        #    print("#36 ",step)

        #print("#77 total tokens ", self.tokenCounter)
        chat_agent_instance = self.workflow.compile()
        
        agent_response = chat_agent_instance.invoke(par_state)
        
        answer = agent_response['generation']
        df_last_result = agent_response['df_last_result']
        
        #answer = self.local_agent.invoke(par_state)['generation']
        return answer,df_last_result
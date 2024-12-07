from typing_extensions import TypedDict
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langgraph.graph import END, StateGraph
import sys
from transformers import AutoTokenizer
#__import__('pysqlite3')
#sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
#import sqlite3
import time
import os
#from langchain_groq import ChatGroq
from custom_agents.prompt_formatter import PromptFormatter
from custom_agents.base_agent import BaseAgent
import pandas as pd
#from tavily import TavilyClient


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
    meeting_info: dict
    

# In[6]:
class meeting_agent(BaseAgent):
    def __init__(self, llm, tokenizer, planning_llm, long_ctx_llm, log_level='INFO', log_file=None, logging_enabled=True):
        # Call the parent class constructor with all the necessary parameters
        super().__init__(llm, tokenizer, planning_llm, long_ctx_llm, log_level=log_level, log_file=log_file, logging_enabled=logging_enabled)
        

        
        current_time = time.strftime("%H:%M:%S")
        



        self.workflow = StateGraph(GraphState)
        self.workflow.add_node("generate_answer", self.generate_answer)

        self.workflow.add_node("init_agent", self.init_agent)

        self.workflow.add_node("meeting_search", self.meeting_search)
        

        
        self.workflow.set_entry_point("init_agent")
        
        self.workflow.add_edge("generate_answer", END)
        self.workflow.add_edge("init_agent","meeting_search")
        self.workflow.add_edge("meeting_search","generate_answer")

        self.local_agent = self.workflow.compile()

        self.generate_answer_chain = self._initialize_generate_answer_chain()
        
    def _initialize_generate_answer_chain(self):
        generate_answer_formatter = PromptFormatter("Llama3")
        generate_answer_formatter.init_message("")
        generate_answer_formatter.add_message("""You are an AI assistant with the objective of preparing a brief for a meeting. 
            You will receive raw data from the target company and the participants (name and company).
            The target company may be the same or not of the participant's companies.
            
            Meeting raw data: {context}
            
            The brief should be like this:

            Meeting with: target_company_name
            Subject: subject of meeting from raw data
            Company notes: a short description and history synthesized from the raw data
            
            Participants: (in tabular format, columns: "person name",  "person's company", "participant notes", "company notes" )
            A table with four columns:
            person 1 name | person's 1 company | a short person description and history synthesized from the raw data | short description of its company only if it's different from the target company to avoid redundancy
            person 2 name | person's 2 company | a short person description and history synthesized from the raw data | short description of its company only if it's different from the target company to avoid redundancy 
            ...

        """, "system")
        generate_answer_formatter.close_message("assistant")

        generate_answer_prompt = PromptTemplate(
            template=generate_answer_formatter.prompt,
            input_variables=["context"]
        )
        return generate_answer_prompt| (lambda x: self.add_tokens_from_prompt(x)) | self.llm | (lambda x: self.add_tokens_from_prompt(x)) | StrOutputParser()

    def meeting_search(self,state):
        meeting_context = state['meeting_info']
        report = ""
        self.log(f"#02 meeting search :  ", level='DEBUG')
        print("#88 meeting search context ", meeting_context)

        # Mockup script to process target_company_name
        target_company_name = meeting_context.get('target_company_name', 'Unknown Company')
        meeting_subject = meeting_context.get('subject', 'Unknown meeting subject')
        self.log(f"Processing target company: {target_company_name}", level='DEBUG')
       
        company_info = self.get_company_profile_data(target_company_name)

        # Mockup script to process participants and their attributes
        participants = meeting_context.get('participants', [])
        participant_info = []
        for participant in participants:
            participant_name = participant.get('participant_name', 'Unknown Participant')
            participant_company = participant.get('participant_company_name', 'Unknown Company')
            participant_email = participant.get('participant_email', 'Unknown Email')
            self.log(f"Processing participant: {participant_name} from {participant_company}", level='DEBUG')
            # Example processing: Create a summary for each participant
            company_profile_data = self.get_company_profile_data(participant_company)
            participant_profile_data = self.get_participant_profile_data(participant_email)
            participant_summary = f"Participant: {participant_name} from {participant_company} (company info: {company_profile_data}) (Email: {participant_email}) (Profile: {participant_profile_data})"
            participant_info.append(participant_summary)

        # Combine the processed information into the report
        report = f"""Meeting preparation assistant report: Meeting with {target_company_name}. 
        Meeting subject: {meeting_subject}
        Company info: {company_info} 
        Participants: {'; '.join(participant_info)}.
        """

        self.log(f"#04 meeting result: {report} ", level='DEBUG')
        return {"context": report}

    def get_company_profile_data(self, target_company_name):
        
        if target_company_name == 'Unknown Company':
            return ""

        company_profile_data = 'Risky company'

        return company_profile_data


    def get_participant_profile_data(self, participant_name):
        
        if participant_name == 'Unknown Participant':
            return ""
            
        participant_profile_data = 'Part of board of directors'

        return participant_profile_data

    def verify_query(self,state):
        next_action = state["next_action"]
        if next_action == 'END':
            return END
        else:
            return next_action

    def generate_answer(self,state):
        """
        Generate answer

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        context = state["context"]
        print("#09 web generate answer", context)
        # Answer Generation
        generation = self.generate_answer_chain.invoke({"context": context})
        #generation = context
        print("#091 generate answer post ", generation)
        return {"generation": generation}

    def init_agent(self,state):
        current_time = time.strftime("%H:%M:%S")
        self.log("#web expert init", level='INFO')
        self.reset_token_counter()
        #here i should get the dict with the meeting parameters and build internal dict
        return {"num_queries": 0,"query_historic":"","context":"","next_action":"","debug_info":"","pars":""}


    
    def ask_question(self, par_state):
        answer = self.local_agent.invoke(par_state)['generation']
        return answer        


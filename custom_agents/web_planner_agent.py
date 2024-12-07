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
from tavily import TavilyClient


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

# In[6]:
class web_planner_agent(BaseAgent):
    def __init__(self, llm, tokenizer, planning_llm, long_ctx_llm, log_level='INFO', log_file=None, logging_enabled=True):
        # Call the parent class constructor with all the necessary parameters
        super().__init__(llm, tokenizer, planning_llm, long_ctx_llm, log_level=log_level, log_file=log_file, logging_enabled=logging_enabled)
        

        
        current_time = time.strftime("%H:%M:%S")
        



        self.workflow = StateGraph(GraphState)
        self.workflow.add_node("generate_answer", self.generate_answer)

        self.workflow.add_node("init_agent", self.init_agent)

        self.workflow.add_node("web_search", self.web_search)
        

        
        self.workflow.set_entry_point("init_agent")
        
        self.workflow.add_edge("generate_answer", END)
        self.workflow.add_edge("init_agent","web_search")
        self.workflow.add_edge("web_search","generate_answer")

        self.local_agent = self.workflow.compile()

    def web_search(self,state):
        tavily_client = TavilyClient(api_key="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")
        web_question = str(state["question"]).strip().replace('"', '')
        pars = str(state["pars"])
        self.log(f"#02 web search : {web_question}/{pars} ", level='DEBUG')
        #try:
        question_test = ""
        question_test += web_question
        #question_test = "latest news about cac40"
        self.log(f"#03 web search : {question_test} ", level='DEBUG')
        context_web = tavily_client.get_search_context(query=question_test)
        #except Exception as e:  
        #    print("###94 error ", str(e))
        self.log(f"#04 web search result: {context_web} ", level='DEBUG')
        return {"context": context_web}

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
        """
        Generate answer

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        question = state["question"]
        context = state["context"]
        print("#09 web generate answer", context)
        # Answer Generation
        #generation = self.generate_answer_chain.invoke({"context": context, "question": question})
        generation = context
        return {"generation": generation}

    def init_agent(self,state):
        current_time = time.strftime("%H:%M:%S")
        self.log("#web expert init", level='INFO')
        self.reset_token_counter()
        return {"num_queries": 0,"query_historic":"","context":"","next_action":"","debug_info":"","pars":""}


    
    def ask_question(self, par_state):
        answer = self.local_agent.invoke(par_state)['generation']
        return answer        


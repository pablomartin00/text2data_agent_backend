import logging
import os
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
import uuid  # Importación de uuid

class BaseAgent:
    def __init__(self, llm, tokenizer, planning_llm, long_ctx_llm, log_level='INFO', log_file=None, logging_enabled=True):
        self.tokenCounter = 0
        self.llm = llm
        self.planning_llm = planning_llm
        self.long_ctx_llm = long_ctx_llm 
        self.tokenizer = tokenizer
        self.logger = logging.getLogger(f"{__name__}.{id(self)}")
        self.logger.propagate = False
        self.logging_enabled = logging_enabled
        
        
        #self.st_session = None

        if self.logging_enabled:
            self._set_log_level(log_level)
            if log_file:
                self.set_log_file(log_file)
            else:
                console_handler = logging.StreamHandler()
                console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
                self.logger.addHandler(console_handler)

    def _set_log_level(self, level):
        level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }
        log_level = level_map.get(level.upper(), logging.INFO)
        self.logger.setLevel(log_level)
        for handler in self.logger.handlers:
            handler.setLevel(log_level)

    def set_log_file(self, file_path):
        if not self.logging_enabled:
            return

        if file_path:
            directory, filename = os.path.split(file_path)
            base_name, extension = os.path.splitext(filename)
            current_date = datetime.now().strftime("%Y%m%d")
            unique_id = uuid.uuid4().hex  # Generación de un identificador único
            new_filename = f"{base_name}-{current_date}-{unique_id}{extension}"  # Modificación del nombre del archivo de log
            new_file_path = os.path.join(directory, new_filename)

            os.makedirs(os.path.dirname(new_file_path), exist_ok=True)

            file_handler = TimedRotatingFileHandler(new_file_path, when='midnight', interval=1, backupCount=7)
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            file_handler.setLevel(self.logger.level)  # Set the handler's level to match the logger's level

            self._clear_handlers()
            self.logger.addHandler(file_handler)

            return new_file_path

    def _clear_handlers(self):
        """Remove all handlers from the logger."""
        while self.logger.handlers:
            handler = self.logger.handlers[0]
            handler.close()
            self.logger.removeHandler(handler)

    def log(self, message, level='INFO'):
        if not self.logging_enabled:
            return
        log_level = getattr(logging, level.upper(), logging.INFO)
        if self.logger.isEnabledFor(log_level):
            self.logger.log(log_level, message)
            for handler in self.logger.handlers:
                handler.flush() 

    def enable_logging(self):
        self.logging_enabled = True

    def disable_logging(self):
        self.logging_enabled = False

    def add_tokens_from_prompt(self, prompt):
        """Add the specified number of tokens to the counter."""
        prompt_type = str(type(prompt))
        
        if prompt_type == "<class 'langchain_core.messages.ai.AIMessage'>":
            tokenAmount = len(self.tokenizer.tokenize(str(prompt)))
            #print("#44 ", str(prompt))
        elif prompt_type == "<class 'langchain_core.prompt_values.StringPromptValue'>":
            tokenAmount = len(self.tokenizer.tokenize(prompt.to_string()))
            #print("#44b ", prompt.to_string())
        else:
            tokenAmount = -1
        
        self.tokenCounter += tokenAmount
        return prompt

    def add_token_amount(self, tokenAmount):
        """Add the specified number of tokens to the counter."""
        self.tokenCounter += tokenAmount
        return self.tokenCounter

    def reset_token_counter(self):
        """Reset the token counter to zero."""
        self.tokenCounter = 0

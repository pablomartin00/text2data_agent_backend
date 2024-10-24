class PromptFormatter:
    def __init__(self, model):
        self.prompt = ""
        self.model = model

    def add_message(self, message, speaker):
        if self.model == "Llama3":
            self.prompt += f"""<|start_header_id|>{speaker}<|end_header_id|>\n{message}\n<|eot_id|>\n"""
    
    def init_message(self, message):
        if self.model == "Llama3":
            self.prompt = f"<|begin_of_text|>\n" 


    def close_message(self, speaker):
        self.prompt += f"<|start_header_id|>{speaker}<|end_header_id|>\n"

    def add_raw(self, raw_text):
        self.prompt += raw_text
            
            


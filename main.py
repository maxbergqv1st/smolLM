from transformers import pipeline
from pprint import pprint

class SmolLM:
  def __init__(self, model_name="HuggingFaceTB/SmolLM-135M-Instruct"):
    print("Loading {model_name} inte memory (this may take a while)...")
    self.pipe = pipeline("text-generation", model_name)
    print("Model loaded successfully!")

  def invoke(self, prompt: str):
    messages = [{ "role": "user", "content": prompt}]
    output = self.pipe(messages, max_new_tokens=150)
    return output[0]['generated_text'][-1]['content']


class PromptTemplate:
  def __init__():
    pass
  
  def format():
    pass

llm = SmolLM()


# pprint(llm.invoke("what is the capital of france"))

# generate = pipeline("text-generation", "HuggingFaceTB/SmolLM-135M-Instruct")

#messages = [
#  { "role": "system", "content": "You're a concise and witty AI tutor"},
#  { "role": "user", "content": "Explain what a token is in NLP using cooking metaphores"}
#]


#"Give me a quick 2-step recipe for a {dish} using only {}"


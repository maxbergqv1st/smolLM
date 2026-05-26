from typing import Any, Callable
from transformers import pipeline
from pprint import pprint

from pydantic import BaseModel, ConfigDict, SerializeAsAny

class Runnable(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def invoke(self, data: Any) -> Any:
      raise NotImplementedError("Iam not implemented! :o")
    
    def __or__(self, other: Any) -> Any:
      if isinstance(other, Runnable):
        return RunnableSequence(
          first=self,
          second=other
        )
      if callable(other):
        return RunnableSequence(
          first=self,
          second=RunnableLamda(func=other)
        )
      return NotImplemented

    def __ror__(self, other: Any) -> Any:
      if callable(other):
        return RunnableSequence(
          first=RunnableLamda(func=other),
          second=self,
        )


class RunnableLamda(Runnable):
  func: Callable[[Any], Any]

  def invoke(self, data: Any) -> Callable[[Any], Any]:
    return self.func(data)

class RunnableSequence(Runnable):
  first: Runnable
  second: Runnable

  def invoke(self, data: Any) -> Any:
    return self.second.invoke(data)




class SmolLM:
  def __init__(self, model_name="HuggingFaceTB/SmolLM-135M-Instruct"):
    print(f"Loading {model_name} into memory (this may take a while)...")
    self.pipe = pipeline("text-generation", model_name)
    print("Model loaded successfully!")

  def invoke(self, prompt: str):
    messages = [{ "role": "user", "content": prompt}]
    output = self.pipe(messages, max_new_tokens=150)
    return output[0]['generated_text'][-1]['content']


class PromptTemplate:
  def __init__(self, template_str: str):
    self.template_str = template_str

  def format(self, **kwargs):
    return self.template_str.format(**kwargs)
    

  def __or__(self, other):
    if isinstance(other, SmolLM):
      return LLMChain(
        prompt_template=self,
        llm=other
      )
    raise TypeError("Its not an instance of SmolLM")

class LLMChain:
  def __init__(
      self, 
      prompt_template: PromptTemplate, 
      llm: SmolLM):
    self.prompt_template = prompt_template
    self.llm = llm

  def invoke(self, **kwargs):
    formatted_prompt = self.prompt_template.format(**kwargs)
    return self.llm.invoke(formatted_prompt)

llm = SmolLM()

recipe_template = PromptTemplate(
  template_str="Give me a quick 2-step recipe for a {dish} using only {ingredient_count} ingredients."
)

recipe_chain = recipe_template | llm

result = recipe_chain.invoke(dish="Spagetti", ingredient_count="Five")
pprint(result)

# pprint(llm.invoke("what is the capital of france"))

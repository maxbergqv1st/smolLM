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

  def invoke(self, data: Any) -> Any:
    return self.func(data)

class RunnableSequence(Runnable):
  first: Runnable
  second: Runnable

  def invoke(self, data: Any) -> Any:
    intermediate = self.first.invoke(data)
    return self.second.invoke(intermediate)


class TicketInput(BaseModel):
  customer_id: int
  message: str


class ProcessedTicket(BaseModel):
  customer_id: int
  sentiment: str
  urgency: str
  summary: str


class SentimentAnalyser(Runnable):
  name: str = "ticket_parser"
  model_version: str = "2.1-stable"

  def invoke(self, ticket: TicketInput) -> dict:
    msg_lower = ticket.message.lower()

    sentiment = "negative" if "broken" in msg_lower or "angry" in msg_lower else "neutral"
    urgency = "high" if "broken" in msg_lower or "urgent" in msg_lower else "low"

    return {
      "customer_id": ticket.customer_id,
      "sentiment": sentiment,
      "urgency": urgency,
      "summary": ticket.message[:40] + "..."
    }


class TicketParser(Runnable):
  def invoke(self, raw_dict: dict) -> ProcessedTicket:
    return ProcessedTicket(**raw_dict)


class RouteTicket(Runnable):
  def invoke(self, ticket: ProcessedTicket) -> dict:
    destination = "engineering_team" if ticket.urgency == "high" else "general_support"
    return {
      "status": "routed",
      "assigned_to": destination,
      "ticket_details": ticket.model_dump()
    }


ticket_pipeline = SentimentAnalyser() | TicketParser() | RouteTicket()

incoming_ticket = TicketInput(
  customer_id=1337,
  message="the payment portal is bokren urgent fix is needed asap!"
)
"""return  'urgency': 'high'}}"""


"""incoming_ticket = TicketInput(
  customer_id=33,
  message="ni suger"
)"""

"""return  'urgency': 'low'}}"""

final_output = ticket_pipeline.invoke(incoming_ticket)

print("--- PIPELINE EXECUTED SUCCESSFULLY ---")
pprint(final_output)

print("--- INSIGHT ---")
pprint(ticket_pipeline.model_dump(exclude_none=True))


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

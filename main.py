from transformers import pipeline
from pprint import pprint

# ============================================================================
# SYSTEMÖVERSIKT: Flödet är följande:
# 1. SmolLM - Laddar en språkmodell från HuggingFace
# 2. PromptTemplate - Definierar en malli med platshållare ({dish}, {ingredient_count})
# 3. LLMChain - Kopplar ihop mallen med modellen
# 4. Slutanvändare - Anropar chain.invoke() som:
#    a. Formaterar mallen med användarens värden
#    b. Skickar den formaterade prompten till modellen
#    c. Returnerar det genererade svaret
# ============================================================================

class SmolLM:
  """
  Kapslar in en HuggingFace-språkmodell för textgenerering.
  
  Flöde:
  - __init__: Laddar modellen från HuggingFace (detta är dyrt och tar tid)
  - invoke(prompt): Tar en textprompt, skickar den till modellen, returnerar svaret
  """
  def __init__(self, model_name="HuggingFaceTB/SmolLM-135M-Instruct"):
    print(f"Loading {model_name} into memory (this may take a while)...")
    # Ladda modellen via HuggingFace pipeline
    self.pipe = pipeline("text-generation", model_name)
    print("Model loaded successfully!")

  def invoke(self, prompt: str):
    """
    Skickar en prompt till modellen och får tillbaka ett genererat svar.
    
    Flöde:
    1. Formatet prompten i rätt format för modellen (messages-lista)
    2. Skicka till modellen och be den generera upp till 150 nya tokens
    3. Extrahera och returnera endast innehållet (inte metadata)
    """
    messages = [{ "role": "user", "content": prompt}]
    # Modellen genererar svar baserat på prompten
    output = self.pipe(messages, max_new_tokens=150)
    # Extrahera det faktiska innehållet från modellens output
    return output[0]['generated_text'][-1]['content']


class PromptTemplate:
  """
  Skapar en återanvändbar prompt-mall med variabler.
  
  Exempel:
    template = PromptTemplate("Give me a recipe for {dish}")
    formatted = template.format(dish="Pizza")
    # Resultat: "Give me a recipe for Pizza"
  """
  def __init__(self, template_str: str):
    # Spara mallsträngen (innehåller {platshållare})
    self.template_str = template_str

  def format(self, **kwargs):
    """
    Ersätter alla {platshållare} med faktiska värden från kwargs.
    Flöde: template_str -> ersätt placeholders -> returnera fylld sträng
    """
    return self.template_str.format(**kwargs)
    

  def __or__(self, other):
    """
    Överladdar | operatorn för att koppla ihop PromptTemplate med SmolLM.
    
    Flöde:
    1. Kontrollera att 'other' är en SmolLM-instans
    2. Om ja: skapa en LLMChain som kombinerar denna mall med modellen
    3. Om nej: kasta ett TypeError
    
    Exempel:
      chain = prompt_template | llm  # Använder | operatorn här
    """
    if isinstance(other, SmolLM):
      return LLMChain(
        prompt_template=self,
        llm=other
      )
    raise TypeError("Its not an instance of SmolLM")

class LLMChain:
  """
  Kopplar samman en PromptTemplate med en SmolLM-modell.
  
  Detta är kärnan i systemet - det är här allt flödar tillsammans.
  """
  def __init__(
      self, 
      prompt_template: PromptTemplate, 
      llm: SmolLM):
    # Spara referenserna till mallen och modellen
    self.prompt_template = prompt_template
    self.llm = llm

  def invoke(self, **kwargs):
    """
    Kör hela flödet från mall till slutsvar.
    
    FLÖDE:
    1. Formatera mallen med användarens värden (kwargs)
       Exempel: {dish} blir "Spaghetti", {ingredient_count} blir "Five"
    2. Skicka den färdiga prompten till SmolLM-modellen
    3. Returnera modellens genererade svar
    
    Exempel:
      result = chain.invoke(dish="Spaghetti", ingredient_count="Five")
      # Steg 1: "Give me a quick 2-step recipe for a Spaghetti using only Five ingredients."
      # Steg 2: Skicka till modellen -> få svar
      # Steg 3: Returnera svaret
    """
    # Steg 1: Formatera mallen med de värden som användaren gav
    formatted_prompt = self.prompt_template.format(**kwargs)
    # Steg 2 & 3: Skicka prompten till modellen och få tillbaka svaret
    return self.llm.invoke(formatted_prompt)

# ============================================================================
# ANVÄNDNING - Här ser du flödet i praktiken
# ============================================================================

# Steg 1: Ladda språkmodellen (detta tar några sekunder)
llm = SmolLM()

# Steg 2: Definiera en prompt-mall med placeholders
recipe_template = PromptTemplate(
  template_str="Give me a quick 2-step recipe for a {dish} using only {ingredient_count} ingredients."
)

# Steg 3: Koppla ihop mallen med modellen använding | operatorn
# Detta skapar en LLMChain som är redo att användas
recipe_chain = recipe_template | llm

# Steg 4: Anropa chainens invoke() med faktiska värden
# FLÖDE SOM HÄNDER HÄR:
# - invoke() anropas med dish="Spagetti", ingredient_count="Five"
# - Mallen formateras: "Give me a quick 2-step recipe for a Spagetti using only Five ingredients."
# - Den formaterade texten skickas till SmolLM-modellen
# - Modellen genererar ett svar
# - Svaret returneras och skrivs ut
result = recipe_chain.invoke(dish="Spagetti", ingredient_count="Five")
pprint(result)

# Alternativt: Du kan också anropa modellen direkt utan mall
# pprint(llm.invoke("what is the capital of france"))

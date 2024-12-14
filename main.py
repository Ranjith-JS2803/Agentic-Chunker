import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from pprint import pprint
import json
import google.generativeai as genAI

genAI.configure(api_key = "AIzaSyCRffrcJDwkdqhrChqJiDiadzn0DSVYXQ0")
llm_model = genAI.GenerativeModel("gemini-1.5-flash")

# First generate the Propositions
print("----- Propositions -----")

prompt = """Decompose the "Content" into clear and simple propositions, ensuring they are interpretable out of
context.
1. Split compound sentence into simple sentences. Maintain the original phrasing from the input
whenever possible.
2. For any named entity that is accompanied by additional descriptive information, separate this
information into its own distinct proposition.
3. Decontextualize the proposition by adding necessary modifier to nouns or entire sentences
and replacing pronouns (e.g., "it", "he", "she", "they", "this", "that") with the full name of the
entities they refer to.
4. Present the results as a list of strings., formatted in JSON.

Example:

Input: Title: ¯Eostre. Section: Theories and interpretations, Connection to Easter Hares. Content:
The earliest evidence for the Easter Hare (Osterhase) was recorded in south-west Germany in
1678 by the professor of medicine Georg Franck von Franckenau, but it remained unknown in
other parts of Germany until the 18th century. Scholar Richard Sermon writes that "hares were
frequently seen in gardens in spring, and thus may have served as a convenient explanation for the
origin of the colored eggs hidden there for children. Alternatively, there is a European tradition
that hares laid eggs, since a hare’s scratch or form and a lapwing’s nest look very similar, and
both occur on grassland and are first seen in the spring. In the nineteenth century the influence
of Easter cards, toys, and books was to make the Easter Hare/Rabbit popular throughout Europe.
German immigrants then exported the custom to Britain and America where it evolved into the
Easter Bunny."
Output: [ "The earliest evidence for the Easter Hare was recorded in south-west Germany in
1678 by Georg Franck von Franckenau.", "Georg Franck von Franckenau was a professor of
medicine.", "The evidence for the Easter Hare remained unknown in other parts of Germany until
the 18th century.", "Richard Sermon was a scholar.", "Richard Sermon writes a hypothesis about
the possible explanation for the connection between hares and the tradition during Easter", "Hares
were frequently seen in gardens in spring.", "Hares may have served as a convenient explanation
for the origin of the colored eggs hidden in gardens for children.", "There is a European tradition
that hares laid eggs.", "A hare’s scratch or form and a lapwing’s nest look very similar.", "Both
hares and lapwing’s nests occur on grassland and are first seen in the spring.", "In the nineteenth
century the influence of Easter cards, toys, and books was to make the Easter Hare/Rabbit popular
throughout Europe.", "German immigrants exported the custom of the Easter Hare/Rabbit to
Britain and America.", "The custom of the Easter Hare/Rabbit evolved into the Easter Bunny in
Britain and America."]

Decompose the following:
{input}"""

def get_propositions(text):
    response = llm_model.generate_content(prompt.replace("{input}", text)).text
    return json.loads(response[8:-5])

with open("./Sample Files/text_data.txt", "r") as file:
    text = file.read()
    
paragraphs = text.split("\n\n")
text_propositions = []
for i, para in enumerate(paragraphs):
    propositions = get_propositions(para)
    text_propositions.extend(propositions)
    print(f"Done with {i+1}-Paragraph")

print(f"You have {len(text_propositions)} propositions\n")
pprint(text_propositions[:5])
print("\n")


print("----- Agentic Chunking ------")

from AgenticChunker import AgenticChunker
ac = AgenticChunker()
ac.add_propositions(text_propositions[:4]) # using only 5 propositions for demonstration
ac.pretty_print_chunks()

import uuid
import google.generativeai as genAI
import time
genAI.configure(api_key = "your_actual_api_key_here")

class AgenticChunker:
    def __init__(self):
        self.chunks = {} # chunk information
        self.agent = genAI.GenerativeModel("gemini-1.5-flash") # LLM model
        self.chunk_id_length = 5 # For truncating the Chunk ID

    def add_propositions(self, propositions):
        for proposition in propositions:
            self.add_proposition(proposition)
            time.sleep(1.5) # This pause is required for controlling the overhead through API calls
    
    def add_proposition(self, proposition): # Method to add a proposition to a chunk
        print(f"Adding : {proposition}")

        if len(self.chunks) == 0: # When there is no chunk present
            print("No Chunks, Creating!!")
            self.create_new_chunk(proposition)
            return
        
        relevant_chunk_id = self.find_relevant_chunk(proposition) # To get a relevant chunk with respective to proposition
        
        if relevant_chunk_id != None: # if a relevant chunk is found
            print(f"Chunk Found!! ({self.chunks[relevant_chunk_id]['chunk_id']}), adding to : {self.chunks[relevant_chunk_id]['title']}")
            self.add_proposition_to_chunk(relevant_chunk_id, proposition) # Add the proposition to existing chunk, update title and summary
            return
        else:
            print("Chunk Not Found")
            self.create_new_chunk(proposition) # Create a new chunk

    def add_proposition_to_chunk(self, chunk_id, proposition):
        self.chunks[chunk_id]["propositions"].append(proposition)

        self.chunks[chunk_id]["summary"] = self.update_chunk_summary(self.chunks[chunk_id])
        self.chunks[chunk_id]["summary"] = self.update_chunk_title(self.chunks[chunk_id])
    
    def update_chunk_summary(self, chunk): # Method to update with new propositions and assign summary to a chunk
        prompt = f"""
        You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic
        A new proposition was just added to one of your chunks, you should generate a very brief 1-sentence summary which will inform viewers what a chunk group is about.

        A good summary will say what the chunk is about, and give any clarifying instructions on what to add to the chunk.

        You will be given a group of propositions which are in the chunk and the chunks current summary.

        Your summaries should anticipate generalization. If you get a proposition about apples, generalize it to food.
        Or month, generalize it to "date and times".

        Example:
        Input: Proposition: Greg likes to eat pizza
        Output: This chunk contains information about the types of food Greg likes to eat.

        Only respond with the chunk new summary, nothing else.

        Chunk's propositions:\n{chunk['propositions']}\n\nCurrent chunk summary:\n{chunk['summary']}
        """

        return self.agent.generate_content(prompt).text # returns updated summary

    def update_chunk_title(self, chunk): # Method to update with new propositions and assign title to a chunk
        prompt = f"""
        You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic
        A new proposition was just added to one of your chunks, you should generate a very brief updated chunk title which will inform viewers what a chunk group is about.

        A good title will say what the chunk is about.

        You will be given a group of propositions which are in the chunk, chunk summary and the chunk title.

        Your title should anticipate generalization. If you get a proposition about apples, generalize it to food.
        Or month, generalize it to "date and times".

        Example:
        Input: Summary: This chunk is about dates and times that the author talks about
        Output: Date & Times

        Only respond with the new chunk title, nothing else.

        Chunk's propositions:\n{chunk['propositions']}\n\nChunk summary:\n{chunk['summary']}\n\nCurrent chunk title:\n{chunk['title']}
        """

        return self.agent.generate_content(prompt).text # returns updated title

    def get_new_chunk_summary(self, proposition): # Method to create and assign summary to a chunk
        prompt = f"""
        You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic
        You should generate a very brief 1-sentence summary which will inform viewers what a chunk group is about.

        A good summary will say what the chunk is about, and give any clarifying instructions on what to add to the chunk.

        You will be given a proposition which will go into a new chunk. This new chunk needs a summary.

        Your summaries should anticipate generalization. If you get a proposition about apples, generalize it to food.
        Or month, generalize it to "date and times".

        Example:
        Input: Proposition: Greg likes to eat pizza
        Output: This chunk contains information about the types of food Greg likes to eat.

        Only respond with the new chunk summary, nothing else.

        Determine the summary of the new chunk that this proposition will go into:
        {proposition}
        """
        return self.agent.generate_content(prompt).text # generates a title

    def get_new_chunk_title(self, summary): # Method to create and assign title to a chunk
        prompt = f"""
        You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic
        You should generate a very brief few word chunk title which will inform viewers what a chunk group is about.

        A good chunk title is brief but encompasses what the chunk is about

        You will be given a summary of a chunk which needs a title

        Your titles should anticipate generalization. If you get a proposition about apples, generalize it to food.
        Or month, generalize it to "date and times".

        Example:
        Input: Summary: This chunk is about dates and times that the author talks about
        Output: Date & Times

        Only respond with the new chunk title, nothing else.

        Determine the title of the chunk that this summary belongs to:
        {summary}
        """
        return self.agent.generate_content(prompt).text # generates a summary

    def create_new_chunk(self,proposition): # Method to create a new chunk, add summary and title with respective to the proposition
        new_chunk_id = str(uuid.uuid4())[:self.chunk_id_length] # chunk's unique id
        new_chunk_summary = self.get_new_chunk_summary(proposition) # chunk's summary
        new_chunk_title = self.get_new_chunk_title(new_chunk_summary) # chunk's title

        self.chunks[new_chunk_id] = {
            'chunk_id' : new_chunk_id,
            'propositions': [proposition],
            'title' : new_chunk_title,
            'summary': new_chunk_summary,
            'chunk_index' : len(self.chunks)
        }

        print(f"Created new chunk ({new_chunk_id}) : {new_chunk_title}")

    def get_chunk_outline(self): # Method that returns all the chunks' detail
        curr_chunk_outline = ""

        for chunk_id, chunk in self.chunks.items():
            chunk_outline = f"Chunk ID: {chunk_id}\nChunk Name: {chunk['title']}\nChunk Summary: {chunk['summary']}\n\n"
            curr_chunk_outline += chunk_outline
        
        return curr_chunk_outline
    
    def find_relevant_chunk(self, proposition): # Method to search for a chunk present in the self.chunks that matches this proposition
        curr_chunk_outline = self.get_chunk_outline()

        prompt = f"""
        Determine whether or not the "Proposition" should belong to any of the existing chunks.

        A proposition should belong to a chunk of their meaning, direction, or intention are similar.
        The goal is to group similar propositions and chunks.

        If you think a proposition should be joined with a chunk, return the chunk id.
        If you do not think an item should be joined with an existing chunk, just return "No chunks"

        Example:
        Input:
            - Proposition: "Greg really likes hamburgers"
            - Current Chunks:
                - Chunk ID: 2n4l3d
                - Chunk Name: Places in San Francisco
                - Chunk Summary: Overview of the things to do with San Francisco Places

                - Chunk ID: 93833k
                - Chunk Name: Food Greg likes
                - Chunk Summary: Lists of the food and dishes that Greg likes
        Output: 93833k
        
        Current Chunks:\n--Start of current chunks--\n{curr_chunk_outline}\n--End of current chunks--
        Determine if the following statement should belong to one of the chunks outlined:
        {proposition}
        """
        curr_chunk_id = self.agent.generate_content(prompt).text.strip()
        
        if len(curr_chunk_id) != self.chunk_id_length:
            return None
        return curr_chunk_id

    def pretty_print_chunks(self): # Method to display the chunks
        print("\n----- Chunks Created -----\n")
        for _, chunk in self.chunks.items():
            print(f"Chunk ID    : {chunk['chunk_id']}")
            print(f"Title       : {chunk['title'].strip()}")
            print(f"Summary     : {chunk['summary'].strip()}")
            print("Propositions:")
            for prop in chunk['propositions']:
                print(f"    -{prop}")
            print("\n\n")

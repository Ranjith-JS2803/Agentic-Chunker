## Agentic Chunker

Agentic Chunking involves taking a text and organizing its propositions into grouped "chunks." Each chunk is a collection of related propositions that are interconnected, allowing for more efficient processing and retrieval within a RAG system.


### How does a human go about chunking a text?

1. You take a pen and paper, and you start at the top of the text, treating the first part as the starting point for a new chunk.

2. As you move down the text, you evaluate if a new sentence or piece should be a part of the previous chunk, if not, then create a new chunk.

3. You repeat this process, methodically working through the text chunk by chunk until you've covered the entire text.

**Who is the "agent" here?** You're correct - it's the human!!

### Understand Files

- **`main.py`**: The main file responsible for generating propositions from a given text and executing the agentic chunking process.
- **`AgenticChunker.py`**: Contains the class and methods that implement the agentic chunking functionality.

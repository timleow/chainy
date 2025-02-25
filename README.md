


![Chainy Logo](public/logo_light.png)

Welcome to Chainy, a cool personal chatbot built using Retrieval-Augmented Generation (RAG)! Chainy is here to answer all your non-personal questions about me with a mix of retrieval and generative magic.

## Features

- **Retrieval-Augmented Answers**: Chainy can answer most questions regarding my endeavours in school, software engineering, or jazz, and it mainly takes information from PDF documents about me that are [here](https://github.com/timleow/chainy/tree/main/pdfs).
- **Integration with Personal Portfolio**: Chainy lives in a chat pop-up in my [portfolio page](https://timleow.netlify.app/). (You can find its own link there too)

## Tech Stack

Chainy is built using the following technologies:

- [Groq API](https://groq.com) - LLM Inferences
-  [🦜🔗 LangChain](https://langchain.com) - RAG Logic
- [FAISS](https://ai.meta.com/tools/faiss/) - Vector Store
- [Chainlit](https://chainlit.io) - FrontEnd

## Future Enhancements

- **Agentic RAG**: Chainy shouldn't need to retrieve from the vector store for every single query. For queries that are irrelevant to its knowledge base, it should simply tell you it doesn't know. I plan to separate the vector stores for the various topics it can answer about me and have another LLM instance in the loop to route the retrieval based on the query, or simply indicate that a retrieval isn't necessary at all.



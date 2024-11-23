from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable, RunnablePassthrough, RunnableConfig

from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser

from langchain.callbacks.base import BaseCallbackHandler

import chainlit as cl
from utils import process_pdfs, PDF_STORAGE_PATH, WELCOME_MESSAGE

doc_search = process_pdfs(PDF_STORAGE_PATH)

@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Chainy's introduction!",
            message="Hey, what can you do?",
            icon="/public/intro.svg",
            ),
        cl.Starter(
            label="Tim's introduction",
            message="Tell me about Tim!",
            icon="/public/tim.svg",
            ),
        cl.Starter(
            label="Tim as a Software Engineer",
            message="Tell me about Tim as a software engineer.",
            icon="/public/com.svg",
            ),

        cl.Starter(
            label="Tim's jazz endeavours",
            message="What has Tim achieved in jazz?",
            icon="/public/sax.svg",
            ),
        ]

@cl.on_chat_start
async def on_chat_start():
    model = ChatGroq(temperature=0,model_name="llama3-8b-8192")
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                '''You're Chainy, an AI assistant to answer questions about Timothy Leow, a software engineer who also loves jazz.
                Answer only based the context provided below, and do not provide any information that is not in the context.
                If the question is not answerable based on the context, simply say "I'm sorry, but I can't answer that.".
                Do not speak like you are referring to a context, just answer the question directly.
                \n\n
                {context}''',
            ),
            ("human", "{question}"),
        ]
    )
    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    retriever = doc_search.as_retriever(search_kwargs={"k": 20})

    runnable = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    cl.user_session.set("runnable", runnable)


@cl.on_message
async def on_message(message: cl.Message):
    # SPECIAL CASE for starter message in chainlit.md
    if message.content == "Hey, what can you do?":
        await cl.Message(content=WELCOME_MESSAGE).send()
        return

    runnable = cl.user_session.get("runnable")  # type: Runnable
    msg = cl.Message(content="")

    class PostMessageHandler(BaseCallbackHandler):
        """
        Callback handler for handling the retriever and LLM processes.
        Used to post the sources of the retrieved documents as a Chainlit element.
        """

        def __init__(self, msg: cl.Message):
            BaseCallbackHandler.__init__(self)
            self.msg = msg
            self.sources = set()  # To store unique pairs

        def on_retriever_end(self, documents, *, run_id, parent_run_id, **kwargs):
            for d in documents:
                source = d.metadata['source'].split('/')[-1]
                source_page_pair = (source, d.metadata['page'])
                self.sources.add(source_page_pair)  # Add unique pairs to the set

        def on_llm_end(self, response, *, run_id, parent_run_id, **kwargs):
            if len(self.sources):
                sources_text = "\n".join([f"[{source}](https://github.com/timleow/chainy/blob/main/pdfs/{source}), page {page+1}" for source, page in self.sources])
                self.msg.elements.append(
                    cl.Text(name="Sources", content=sources_text, display="inline")
                )

    async for chunk in runnable.astream(
        message.content,
        config=RunnableConfig(callbacks=[
            cl.LangchainCallbackHandler(),
            PostMessageHandler(msg)
        ]),
    ):
        await msg.stream_token(chunk)

    await msg.send()
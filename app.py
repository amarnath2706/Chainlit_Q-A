import os
from typing import List

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import (ConversationalRetrievalChain,)
from langchain.chat_models import ChatOpenAI

from langchain.docstore.document import Document
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from dotenv import load_dotenv
import chainlit as cl

print('All ok')

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

#initialize the text_splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

@cl.on_chat_start
async def on_chat_start(): #async - concurrency (at the same time we can run the multiprocessing or thread)
    #collecting files
    files = None

    while files == None:
        files = await cl.AskFileMessage(
            content="Please upload a text file to begin!",
            accept = ["text/plain"],
            max_size_mb=20,
            timeout=180,
        ).send()

    file = files[0]

    #Once we got the file - open the file - perform text splitting

    msg =  cl.Message(content=f"Processing {file.name} ...", disable_feedback=True)
    await msg.send()

    #open the files
    with open(file.path, "r", encoding='utf-8') as f:   
        text = f.read()

    #split the text into chunks 
    texts = text_splitter.split_text(text)

    #create a metadata
    metadatas = [{"source":f"{i}-pl"}for i in range(len(texts))]

    #In RAG the first process is data ingestion then we have to convert into enbeddings
    #initiate the embeddings
    embeddings = OpenAIEmbeddings()
    #create the embeddings and store inside the chromadb
    docsearch = await cl.make_async(Chroma.from_texts)(
        texts,embeddings, metadatas = metadatas
    )

    #Sustain the chat history
    message_history=ChatMessageHistory()
    memory=ConversationBufferMemory(
        memory_key='chat_history',
        output_key='answer',
        chat_memory=message_history,
        return_messages=True
    )

    #Chain
    chain=ConversationalRetrievalChain.from_llm(
        ChatOpenAI(model_name='gpt-3.5-turbo',temperature=0,streaming=True),
        chain_type='stuff',
        retriever=docsearch.as_retriever(),
        memory=memory,
        return_source_documents=True
    )

    #Start the app by asking questions based on the documents
    msg.content = f"Processing {file.name} done. You can now ask questions!"
    await msg.update()  

    cl.user_session.set("chain",chain)

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler()

    res = await chain.acall(message.content,callbacks=[cb])
    answer=res['answer']
    source_documents = res['source_documents']

    #final snswer
    
    text_elements = []  # type: List[cl.Text]

    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            # Create the text element referenced in the message
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name)
            )
        source_names = [text_el.name for text_el in text_elements]

        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"

    await cl.Message(content=answer, elements=text_elements).send()    
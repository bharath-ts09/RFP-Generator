import json
import os
import sys
import boto3
import streamlit as st

# To generate embeddings

from langchain.embeddings import BedrockEmbeddings
from langchain_community.chat_models import BedrockChat

# Data ingestion

import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader

#Vector Embedding and Vector store

from langchain_chroma import Chroma
from langchain_core.documents import Document

#LLM 

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

#Bedrock Clients Titan Text Embeddings V2
bedrock=boto3.client(service_name="bedrock-runtime", region_name = "us-east-1")
bedrock_embeddings=BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", client=bedrock)

# Data ingestion

def data_ingest():
    urls= ['https://www.ibef.org/industry/manufacturing-sector-india']
    loader=UnstructuredURLLoader(urls=urls)
    data=loader.load()

    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)

    docs=text_splitter.split_documents(data)
    return docs

# Vector Embedding and vector store

def get_vector_store(docs):
    vectorstore = Chroma.from_documents(docs,bedrock_embeddings,persist_directory="chroma_index")
    #vectorstore.persist()


def get_sonnet_llm():
    llm=BedrockChat(model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",client=bedrock,model_kwargs={'max_tokens':1000})
    return llm

prompt_template = """
Objective : To create Request for Proposal (RFP) document
Human: You are a senior purchasing manager, you have been tasked to create RFP to solicit proposals from vendors
for products and services that your company intend to purchase. The RFP is a critical document that outlines 
companies requirements, expectations and evaluation criteria. Generate an effective RFP by addressingthe following 
key points:

Introduction and Background: Provide brief introduction about your company, its mission and purpose of RFP
Scope : Clearly define the scope of work and spcific products and service you require from company
Vendor qualifications: Specify the qualifications and experience you expect from vendors
Terms and conditions : Outline any specific terms, conditions and requirements that vendor must agree including
pricing, schedules, intellectual property
<context>
{context}
</context>

Question: {question}


Assistant:"""

PROMPT = PromptTemplate(template=prompt_template,input_variables=["context","question"])

def get_response_llm(llm,vectorstore,query):
    qa=RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}

    )
    answer=qa({"query":query})
    return answer['result']

def main():
    st.set_page_config("Generate RFP")
    st.header("Take customer query to generate RFP")

    user_question = st.text_input("Which buyer do you want to generate RFP")

    with st.sidebar:
        st.title("Menu:")

        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                docs = data_ingest()
                get_vector_store(docs)
                st.success("Done")

    if st.button("Generate RFP"):
        with st.spinner("Processing..."):
            #Chroma_index = Chroma.load("chroma_index")
            Chroma_index = Chroma(persist_directory="chroma_index", embedding_function=bedrock_embeddings)
            llm=get_sonnet_llm()

            st.write(get_response_llm(llm,Chroma_index,user_question))
            st.success("Done")


if __name__ == "__main__":
    main()
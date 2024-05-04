#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import openai
import langchain
import tiktoken
from dotenv import load_dotenv, find_dotenv
import os
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document


# In[ ]:


#load_dotenv(find_dotenv(), override=True)
api_key="sk-proj-IYP3uNpGeC8Xq9I5ein6T3BlbkFJCT0bfEE2zKFxE9qCTlAU"


# In[ ]:


def crear_resumen(txt):
    llm = ChatOpenAI(temperature=0.5, model_name='gpt-4', openai_api_key=api_key)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=20000, chunk_overlap=500)
    chunks=text_splitter.create_documents([texto])
    
    map_prompt='''
    Escriba un resumen conciso y breve del siguiente texto:
    Que el resumen esté en español.
    TEXTO: {text}
    '''
    map_prompt_template = PromptTemplate(
        input_variables=['text'],
        template=map_prompt
    )
    
    prompt_combinado='''
    Escriba un resumen conciso del siguiente texto que abarque los puntos claves.
    Añade un título al resumen.
    Comience su resumen con un PÁRRAFO DE INTRODUCCIÓN que ofrezca una visión general del tema SEGUIDO
    por los PUNTOS tipo "BULLET Point" si es posible Y termine el resumen con una CONCLUSIÓN.
    Texto: {text}
    '''
    prompt_template_combinado= PromptTemplate(
        input_variables=['text'],
        template=prompt_combinado
    )
    
    cadena_resumen = load_summarize_chain(
    llm=llm,
    chain_type='map_reduce',
    map_prompt = map_prompt_template,
    combine_prompt=prompt_template_combinado,
    verbose=False
    )
    
    return cadena_resumen.run(chunks)

# Título App en Web
st.set_page_config(page_title='App Resumen Texto')
st.title('App Resumen Texto')

# Introducción Texto
texto = st.text_area('Introduce texto a resumir:', '', height=200)

# Aceptar el texto introducido por el usuario para el resumen
result=[]
with st.form('summarize_form', clear_on_submit=True):
    submitted = st.form_submit_button('Submit')
    if submitted:
        with st.spinner('Resumiendo...'):
            resumen = crear_resumen(texto)
            result.append(resumen)

if len(result):
    st.info(resumen)
            
    


#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importación de librerias necesarias

import streamlit as st
import langchain
import tiktoken
from dotenv import load_dotenv, find_dotenv
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.docstore.document import Document


# In[3]:


# Definimos una función para la realización del resumen del texto que tengamos.
# Utilizaré el modelo gpt-4 para ChatOpenAi de la libreríra langchain.

def crear_resumen_mapreduce(txt, temp, modelo):

    llm = ChatOpenAI(temperature=temp, model_name=modelo, openai_api_key=openai_api_key)
    

    # Para evitar problemas con textos grandes. Dividiremos el texto en "chunks" con unos 20000 caracteres
    # cada uno y que se solapen unos 500 caracteres para mantener el contexto.
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=20000, chunk_overlap=500)
    chunks=text_splitter.create_documents([txt])
    
    # Como vamos a utilizar el métod map_reduce, definimos un prompt_template para las primeras particiones del texto
    # al cual se les hará un resumen; y otro template para resumir la combinación de los resumenes anteriores.
    
    # Le indicamos cómo queremos que haga el resumen:
    map_prompt='''
    Escriba un resumen conciso y breve del siguiente texto:  
    Que el resumen esté en español.
    TEXTO: {text}
    '''
    
    map_prompt_template = PromptTemplate(
        input_variables=['text'],
        template=map_prompt
    )
    
    # En este prompt le damos más detalles de cómo queremos que dé formato el resumen final, con una título, apartados
    # con "bullet points" y una conclusión.
    
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
    
    # En este momento realizamos la cadena resumen, para ello utilizamos el LLM configurado anteriormente con ChatOpenAI,
    # con el método map_reduce, donde primero se generan resumenes de los fragmentos con la plantilla 'map_prompt_template',
    # y leugo se hace un resumen final a partir de estos con la plantilla 'prompot_template_combinado'
    
    cadena_resumen = load_summarize_chain(
        llm=llm,
        chain_type='map_reduce',
        map_prompt = map_prompt_template,
        combine_prompt=prompt_template_combinado,
        verbose=False
    )
    
    return cadena_resumen.run(chunks)

def crear_resumen_refine(txt, temp, modelo):

    llm = ChatOpenAI(temperature=temp, model_name=modelo, openai_api_key=openai_api_key)
    

    # Para evitar problemas con textos grandes. Dividiremos el texto en "chunks" con unos 20000 caracteres
    # cada uno y que se solapen unos 500 caracteres para mantener el contexto.
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=20000, chunk_overlap=500)
    chunks=text_splitter.create_documents([txt])
    
    # Como vamos a utilizar el métod map_reduce, definimos un prompt_template para las primeras particiones del texto
    # al cual se les hará un resumen; y otro template para resumir la combinación de los resumenes anteriores.
    
    # Le indicamos cómo queremos que haga el resumen:
    prompt_template='''
    Escriba un resumen conciso y breve del siguiente texto:  
    Que el resumen esté en español.
    TEXTO: {text}
    '''
    
    prompt_inicial = PromptTemplate(
        input_variables=['text'],
        template=prompt_template
    )
    
    # En este prompt le damos más detalles de cómo queremos que dé formato el resumen final, con una título, apartados
    # con "bullet points" y una conclusión.
    
    refine_template='''
    Escriba un resumen final.
    He proporcinado un resumen existente hasta cierto punto: {existing_answer}.
    Perfecciona el resumen existente con algo más de contexto a continuación.
    -------------
    {text}
    -------------
    Comience el resumen final con una "Introdiucción" que nos de una visión general del tema seguido por los puntos
    más relevantes ("Bullet Points"). Termina el resumen con una conclusión.
    '''
    refine_prompt= PromptTemplate(
        input_variables=['existing_answer', 'text'],
        template=refine_template
    )
    
    # En este momento realizamos la cadena resumen, para ello utilizamos el LLM configurado anteriormente con ChatOpenAI,
    # con el método map_reduce, donde primero se generan resumenes de los fragmentos con la plantilla 'map_prompt_template',
    # y leugo se hace un resumen final a partir de estos con la plantilla 'prompot_template_combinado'
    
    cadena_resumen = load_summarize_chain(
        llm=llm,
        chain_type='refine',
        question_prompt=prompt_inicial,
        refine_prompt=refine_prompt,
        return_intermediate_steps=False
    )
    
    return cadena_resumen.run(chunks)





#####    STREAMLIT  ########

# Vamos a definir la apariencia de nuestra app en streamlit:

# Título App en Web
st.set_page_config(page_title='App Resumen Texto')
st.title('App Resumen Texto')

# Creamos un botón de usuario avanzado:

#if st.button('Opciones Avanzadas'):
#    # Muestra un control deslizante
#    valor_temperatura = st.slider('Selecciona Creatividad (0 Nula, 2 Creativo)', 0.0, 2.0, 0.1)
#   st.write(f'Has seleccionado: {valor_temperatura}')
#    seleccion = st.selectbox('Seleccione método:', ['map_reduce', 'refine'])
    
#else:
#    valor_temperatura = 0.5
#    seleccion = 'map_reduce'

valor_temperatura = st.slider('Selecciona Creatividad (0 Nula, 1 Creativo)', 0.0, 1.0, 0.5)
seleccion = st.selectbox('Seleccione método:', ['map_reduce', 'refine'])
modelo = st.radio('Elige un modelo:', options=('gpt-4', 'gpt-3.5-turbo'))


        
# Introducción Texto en un área creada a tal efecto
texto = st.text_area('Introduce texto a resumir:', '', height=200)

# Aceptar el texto introducido por el usuario para el resumen
result=[]

# Creamos un formulario en Streamlit que borraremos después de enviarlo

with st.form('summarize_form', clear_on_submit=True): 
# Pedimos la clave API de openAI, cuando se haya ingresado un texto
    openai_api_key = st.text_input('OpenAI API Key', type = 'password', disabled=not texto) 
    submitted = st.form_submit_button('Submit') # Creamos un boton 'Submit' para que el usuario enví el formulario
    if submitted and openai_api_key.startswith('sk-'): # Comprobamos si se ha dado a 'Submit' y si la clave es correcta.
        with st.spinner('Resumiendo...'): # Se genera un mensaje de 'Resumiendo...' mientras se realiza la tarea
            
            
            
            if seleccion == 'map_reduce':
                resumen = crear_resumen_mapreduce(texto, valor_temperatura, modelo) # Llamamos a la función para realizar el resumen
                result.append(resumen) # Añadimos el resumen a una lista
            if seleccion == 'refine':
                resumen = crear_resumen_refine(texto, valor_temperatura, modelo)
                result.append(resumen)
            
if len(result):
    st.info(resumen) # si la longitud de la lista es mayor que 0, se muestra el resumen en la interfaz de usuario
            
    


# In[ ]:





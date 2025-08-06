import chromadb

client_dbg = chromadb.PersistentClient(path="chroma_db")
col_dbg    = client_dbg.get_collection("student_profile_mock",
                                       embedding_function=None)
print("stored vectors:", col_dbg.count())     # prints 1 immediately
del col_dbg, client_dbg 

import os
import openai
from dotenv import load_dotenv, find_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import pandas as pd


API_KEY="**************"
API_BASE="******************"
llm_model = model 

print("creating emebbder", flush=True)
embedder = OpenAIEmbeddings(
    model           = "text-embedding-3-small",
    openai_api_key  = API_KEY,
    openai_api_base = API_BASE
)
print("vectorstore", flush=True)

vectorstore = Chroma(
    persist_directory = "chroma_db",
    collection_name   = "student_profile_mock",
    embedding_function= embedder,
)
print("vect finsihed")
print("Collection size =", vectorstore._collection.count())   # should print > 0 immediately

initial_question = input("Ask me a math question: ")
docs = vectorstore.similarity_search(initial_question, k=1)
print("Retrieved", len(docs), "doc(s)")
context = "\n\n".join(d.page_content for d in docs)


system_prompt = f"""You are a helpful math tutor. 
      Before responding to a student question, look at the information I'm providing you about the student, 
      use that information to guide your responses. For example, if the student has gaps listed 
      focus first on explaining basic terms, then confirm understanding, and then move on to abstract rules.
    ------------------
    The data:
    {context}
    ------------------
    """

print( "about to call openai", flush=True)

openai_client = openai.OpenAI(api_key=API_KEY, base_url=API_BASE)

student_response_prompt = 'pretend you are student with poor understanding of math responding to the tutor\'s explanation. \
    The following is the message from a tutor for you to respond to. Ask for more help or to simplify explanation: ' 

def tutor(initial_question, model=llm_model):
    counter = 0
    tutor_responses=[]
    student_questions =[initial_question]
    while counter < 3:
        if counter==0:
            completion = openai_client.chat.completions.create(
                model=model,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": initial_question}
                    ]) # ask first question
            # store tutor response
            tutor_responses.append(completion.choices[0].message.content)
            
        else:
            # generate next student question
            student_q = student(tutor_responses[-1], student_response_prompt)
            student_questions.append(student_q)
            completion = openai_client.chat.completions.create(
                model=model,
                temperature=0.0,
                messages=[{"role": "user", "content": student_q}])
            # generate tutor response to next question
            tutor_responses.append(completion.choices[0].message.content)
                       
        counter += 1
    df = pd.DataFrame({'question': student_questions, 'responses': tutor_responses})
    print(df.to_string(index=False))   
    return df
    

def student(response, student_response_prompt,  model=llm_model, system=None):
    responses=[]
    completion = openai_client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[{"role": "user", "content": student_response_prompt + response}])
    response = completion.choices[0].message.content
    responses.append(response)
    return responses[-1]

tutor(initial_question)

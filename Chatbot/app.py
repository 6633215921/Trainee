SYSTEM_TEMPLATE = """You are an expert at analyzing contexts to provide accurate answers. Use the provided context and chat history to answer questions correctly. Do not mention the chat history or question or provided context in your response. If you don't know the answer, say so without making up an answer."""
FULL_PROMPT_TEMPLATE = """
Chat history:
{history}

Context:
{context}

Question:
{question}

You are an expert at analyzing contexts to provide accurate answers. 
Use the provided context and chat history to answer questions correctly. 
Do not mention the chat history, question, or provided context in your response. 
If you cannot find a definitive answer based on the information given, respond politely with a gentle phrase like "ขออภัยนะคะ/ครับ ดิฉัน/ผมไม่มีข้อมูลเพียงพอที่จะตอบคำถามนี้ได้อย่างถูกต้องค่ะ/ครับ" 
Always use polite endings in Thai such as "ค่ะ" or "ครับ" as appropriate. Do not elaborate or speculate. Answer concisely and politely as if the information is your own knowledge.
RESPONSE IN THAI ONLY."""
PROMPT_CLASSIFICATION = """
You are an intelligent assistant tasked with classifying incoming questions into three categories: SSO (Social Security Office), SharePoint (Microsoft), or Other (neither SSO nor SharePoint). 
Please follow these instructions carefully:

Classification Criteria:
    SSO: Questions related to Social Security Office, such as social security benefits, policies, procedures, applications, and related services. (Class 1)
    SharePoint: Questions related to Microsoft SharePoint, such as document management, collaboration tools, site management, integration with other Microsoft services, and related functionalities. (Class 2)
    Other: Questions that do not pertain to either SSO or SharePoint. (Class 3)
    
Response Format:
    Respond with the classification number only (1, 2, or 3) based on the question content.
    
Examples:
    Example 1:
    Question: How do I apply for social security benefits?
    Response: 1
    
    Example 2:
    Question: How can I create a new site in SharePoint?
    Response: 2
    
    Example 3:
    Question: What are the latest features of Microsoft Office 365?
    Response: 3
    
    Example 4:
    Question: วิธีการสมัครประกันสังคมคืออะไร?
    Response: 1
    
    Example 5:
    Question: ฉันจะเพิ่มผู้ใช้ใหม่ใน SharePoint ได้อย่างไร?
    Response: 2

Question:{question}"""


from langchain_openai import OpenAIEmbeddings
import os
import time
import PyPDF2
from langchain_community.llms import Ollama
import tiktoken
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.document import Document
from qdrant_client import QdrantClient , models
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Qdrant
from qdrant_client.models import Distance, VectorParams
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client.models import PointStruct
from database import MySQLDatabase as db
from datetime import datetime
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
import numpy as np

# load_dotenv()
client = QdrantClient(url=os.getenv('QDRANT_URL'))
input_price = {"llama3": 0,"gpt-3.5-turbo": 1.5, "gpt-4o": 5, "gpt-4o-mini": 0.15,"claude-3-sonnet-20240229": 3, "claude-3-haiku-20240307": 0.25, "gemini-1.0-pro": 0.5, "gemini-1.5-pro": 3.5, "gemini-1.5-flash": 0.35}
output_price = {"llama3": 0,"gpt-3.5-turbo": 2, "gpt-4o": 15, "gpt-4o-mini": 0.6, "claude-3-sonnet-20240229": 15, "claude-3-haiku-20240307": 1.25, "gemini-1.0-pro": 1.5, "gemini-1.5-pro": 10.50, "gemini-1.5-flash": 1.05}

def tiktoken_encodings(example_string: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    token_integers = encoding.encode(example_string)
    num_tokens = len(token_integers)
    return num_tokens

def ensure_string(input_value):
    if not isinstance(input_value, str):
        return str(input_value)
    return input_value

def cal_token_cost(number_token,model_name,type_input_output):
    if type_input_output == "input":
        cal = input_price.get(model_name)
    elif type_input_output == "output":
        cal = output_price.get(model_name)
    return (number_token * cal)/1e06


def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500 , chunk_overlap=50)
    return text_splitter.split_documents(documents)

def add_vectorstore(documents : list[Document],collection_name : str):
    if not client.collection_exists(collection_name=collection_name):
        client.recreate_collection(collection_name=collection_name,vectors_config=VectorParams(size=3072, distance=Distance.DOT),)
    points = []
    response = client.scroll(collection_name=collection_name,limit=2**30)
    try:max_id = response[0][len(response[0])-1].id
    except:max_id = -1
    EMBEDDINGS_MODEL = OpenAIEmbeddings(model='text-embedding-3-large', api_key=os.getenv('EMBEDDINGS_MODEL_API'), dimensions=3072)
    for i, document in enumerate(documents):
        embedding = EMBEDDINGS_MODEL.embed_query(document.page_content)
        point = PointStruct(id= max_id+i+1,vector=embedding,payload={"page_content": document.page_content,"metadata": document.metadata})
        points.append(point)
    client.upsert(collection_name=collection_name,wait=True,points=points,)
    print("add documents complete", len(documents))
    return

def search_list(qdrant_instance: Qdrant, query_text: str, limit: int, collection_name: str):
    EMBEDDINGS_MODEL = OpenAIEmbeddings(model='text-embedding-3-large', api_key=os.getenv('EMBEDDINGS_MODEL_API'), dimensions=3072)
    query_vector = EMBEDDINGS_MODEL.embed_query(query_text)
    search_results = qdrant_instance.search(collection_name=collection_name, query_vector=query_vector, with_payload=True, limit=limit)
    context = [ {'context':result.payload['page_content'], 'source':result.payload['metadata']['source']} for result in search_results if result.score > 0.0]
    return context

def list_to_context(list_context:list[str]):
    context = ""
    for member in list_context:
        context += f"{member}\n\n"
    return context

def pdf_to_document(pdf_file, pdf_name):    
    text = ""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num] 
        text += page.extract_text()
    if text:
        return [Document(page_content = text.replace("\n","").replace(" ",""), metadata={"source": pdf_name})]
    return []

def stream_llama3_chat(prompt:str):
    llm = Ollama(model="llama3")
    options = {"device": "cuda"}
    for chunks in llm.stream(prompt, **options):
        yield chunks
def stream_gemini_chat(prompt:str,model_name:str):
    llm = ChatGoogleGenerativeAI(model=model_name)
    stream = llm.stream([prompt])
    return stream
def stream_claude_chat(prompt:str,model_name:str):
    llm = ChatAnthropic(model=model_name,api_key=os.getenv('ChatAnthropic_API'))
    stream = llm.stream([prompt])
    return stream
def stream_gpt_chat(prompt:str,model_name:str):
    llm = ChatOpenAI(model=model_name, api_key=os.getenv('ChatOpenAI_API'))
    stream = llm.stream([prompt])
    return stream
    
def invoke_llama3_classification(query:str):
    llm = Ollama(model="llama3",temperature=0,num_predict=1)
    result = llm.invoke(query)
    return result[0]
def invoke_gemini_classification(query:str,model_name:str):
    llm = ChatGoogleGenerativeAI(model=model_name,temperature=0,max_output_tokens=1)
    result = llm.invoke(query)
    return result.content[0]
def invoke_claude_classification(query:str,model_name:str):
    llm = ChatAnthropic(model=model_name,temperature=0,api_key=os.getenv('ChatAnthropic_API'))
    result = llm.invoke(query)
    return result.content[0]
def invoke_gpt_classification(query:str,model_name:str):
    llm = ChatOpenAI(model=model_name, api_key=os.getenv('ChatOpenAI_API'), temperature=0.0,max_tokens=1)
    response = llm.invoke([query])
    return response.content[0]
def render(document_list: list):
    retriever_message = st.expander(f"Click to see retriever's document 📄")
    with retriever_message:
        st.markdown("#### Retrieval results")
        number = 1
        while len(document_list)>0:
            button_columns = st.columns([0.2, 0.2, 0.2, 0.2, 0.2], gap="small")
            for index, document in enumerate(np.array(document_list)[:5]):
                with button_columns[index], st.popover(f"Source {number}"):
                    st.markdown(f"""
                                #### Source: {document["source"]}

                                """)
                    st.markdown(document["context"])
                number += 1
            if len(document_list) <= 5:
                document_list = []
            else:
                document_list = document_list[5:]

st.set_page_config(page_title="Chat with PDFs",page_icon=":robot_face:")
st.header("Chat with PDFs:robot_face:")

with st.sidebar:
    st.header("Your document")
    pdf_docs = st.file_uploader("Upload your PDF",accept_multiple_files=True)
    colA, colB = st.columns(2)
    colC, colD = st.columns(2)
    uploaded = colA.button("Upload File")
    if 'List_of_pdf' not in st.session_state :
        st.session_state['List_of_pdf'] = []
    if pdf_docs and uploaded :
        with st.spinner("Add vectorstore"):
            documents = []
            for one_pdf in pdf_docs:
                if one_pdf.name not in st.session_state["List_of_pdf"]:
                    documents.extend(pdf_to_document(one_pdf, one_pdf.name))
            if documents:
                chunks = split_documents(documents)
                chunks_by_pdf = {}
                for chunk in chunks:
                    source = chunk.metadata.get("source")
                    if source not in chunks_by_pdf:
                        chunks_by_pdf[source] = []
                    chunks_by_pdf[source].append(chunk)
                for source, chunks_list in chunks_by_pdf.items():
                    add_vectorstore(chunks_list,st.session_state["collection_name"])
    
    colB.link_button("Go to Database", "http://localhost:6334/dashboard")         
    if 'MEMORY_FOR_DISPLAY' not in st.session_state:
        st.session_state['MEMORY_FOR_DISPLAY'] = []
    if colC.button("Clear Memory"):
        st.session_state['MEMORY_FOR_DISPLAY'] = []
    if colD.button("Clear Database"):
        client.recreate_collection(collection_name=st.session_state["collection_name"],vectors_config=models.VectorParams(size=3072, distance=models.Distance.COSINE),)        
    
    if 'collection_name' not in st.session_state :
        st.session_state['collection_name'] = ""
        
    if 'class_last' not in st.session_state :
        st.session_state['class_last'] = [0,0]
    if 'class_sum' not in st.session_state :
        st.session_state['class_sum'] = [0,0]    
    if 'llm_last' not in st.session_state :
        st.session_state['llm_last'] = [0,0]
    if 'llm_sum' not in st.session_state :
        st.session_state['llm_sum'] = [0,0]    
    
    st.header("Setting")
    with st.popover("Open setting"):
        st.session_state["model_classification"] = st.selectbox("Choose model classification", ["claude-3-sonnet-20240229","llama3","gpt-3.5-turbo","gpt-4o", "gpt-4o-mini","claude-3-haiku-20240307","gemini-1.0-pro", "gemini-1.5-pro","gemini-1.5-flash"])
        st.session_state["model_result"] = st.selectbox("Choose model result", ["gemini-1.5-flash","llama3","gpt-3.5-turbo","gpt-4o", "gpt-4o-mini","claude-3-sonnet-20240229","claude-3-haiku-20240307","gemini-1.0-pro", "gemini-1.5-pro"])
        st.session_state["collection_name"] = st.selectbox("Choose your database to upload file", ["SSO","Sharepoint"])
        try :
            st.write(f"List PDF of {st.session_state['collection_name']} : ")
            response = client.scroll(collection_name=st.session_state["collection_name"],limit=2**30)
            st.session_state["List_of_pdf"] = {item.payload['metadata']['source'] for item in response[0]}
            for idx, source in enumerate(sorted(st.session_state["List_of_pdf"]) , start=1 ) :
                st.write(f"{idx}.{source}")
        except:
            response = []
            st.session_state["List_of_pdf"] = []
        if st.button("Restart", type="primary" ):
            st.rerun()

for msg in st.session_state['MEMORY_FOR_DISPLAY']:
    if msg['role'] == 'user':
        message = st.chat_message("user")
        message.write(msg['content'])
    elif msg['role'] == 'assistant':
        message = st.chat_message("assistant")
        message.write(msg['content'])
    elif msg['role'] == 'report':
        st.caption(msg['content'])
    elif msg['role'] == 'retrieval':
        render(msg['content'])
    

question = st.chat_input("Ask PDFs")
if question:
    start = time.time()
    model_classification = st.session_state["model_classification"]
    model_result = st.session_state["model_result"]
    with st.chat_message("user"):
        st.markdown(question)
        
    # check 
    prompt_template = PromptTemplate.from_template(PROMPT_CLASSIFICATION)
    class_prompt = prompt_template.format(question=question)
    
    if model_classification == "llama3":
        check = invoke_llama3_classification(class_prompt)  
    elif model_classification in ["gpt-3.5-turbo","gpt-4o", "gpt-4o-mini"]:
        check = invoke_gpt_classification(class_prompt,model_classification) 
    elif model_classification in ["claude-3-sonnet-20240229","claude-3-haiku-20240307"] :
        check = invoke_claude_classification(class_prompt,model_classification) 
    elif model_classification in ["gemini-1.0-pro", "gemini-1.5-pro","gemini-1.5-flash"] :
        check = invoke_gemini_classification(class_prompt,model_classification) 
        
    # context
    try:
        find_bool = int(check)
    except: 
        find_bool = 3
    if find_bool == 1 :#SSO
        list_context = search_list(client,question,7,"SSO")
        context = list_to_context([item['context'] for item in list_context])
    elif find_bool == 2 :#Sharepoint
        list_context = search_list(client,question,7,"Sharepoint")
        context = list_to_context([item['context'] for item in list_context])
    else : #Other
        context = ""
    
    # upload db first step
    db.reset()
    db.question = question
    db.result = check
    db.model_name = model_classification
    db.input_prompt = class_prompt
    db.input_token = tiktoken_encodings(class_prompt)
    db.output_token = tiktoken_encodings(check)
    if find_bool == 1 :#SSO
        db.database_name = "SSO"
    elif find_bool == 2 :#Sharepoint
        db.database_name = "Sharepoint"
    else : #Other
        db.database_name = "None_find"
    input_cost = cal_token_cost(tiktoken_encodings(class_prompt),model_classification,"input")
    output_cost = cal_token_cost(tiktoken_encodings(check),model_classification,"output")
    db.cost = input_cost+output_cost
    db.time_stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    db.time_taken_to_answer = (time.time()-start)
    db.step = "classification"
    db.insert_into_database()
    
    start = time.time()
    # system
    system = os.getenv('SYSTEM_TEMPLATE')
    
    # history
    memorys = st.session_state["MEMORY_FOR_DISPLAY"]
    memorys = [memory for memory in memorys if memory['role'] != 'report']
    max_memorys = 4
    if len(memorys) > max_memorys:
        memorys = memorys[-max_memorys:] 
    history = ''.join([f"{memory['role']}: {memory['content']}\n" for memory in memorys])
    
    # create input_prompt    
    # create system_history_context_queation
    # FULL_PROMPT_TEMPLATE = os.getenv('FULL_PROMPT_TEMPLATE')
    prompt_template = PromptTemplate.from_template(FULL_PROMPT_TEMPLATE)
    input_prompt = prompt_template.format(history=history,context=context,question=question)
    
    with st.chat_message("assistant"):  
        if model_result == "llama3":
            message = st.write_stream(stream_llama3_chat(input_prompt))
        elif model_result in ["gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini"]:
            message = st.write_stream(stream_gpt_chat(input_prompt, model_result))
        elif model_result in ["claude-3-sonnet-20240229", "claude-3-haiku-20240307"]:
            message = st.write_stream(stream_claude_chat(input_prompt, model_result))
        elif model_result in ["gemini-1.0-pro", "gemini-1.5-pro", "gemini-1.5-flash"]:
            message = st.write_stream(stream_gemini_chat(input_prompt, model_result))
                        
    A = cal_token_cost(tiktoken_encodings(class_prompt),model_classification,"input")
    B = cal_token_cost(tiktoken_encodings(check),model_classification,"output")
    C = cal_token_cost(tiktoken_encodings(input_prompt),model_result,"input")
    D = cal_token_cost(tiktoken_encodings(message),model_result,"output")
    total_cost = A+B+C+D

    check = ensure_string(check)
    message = ensure_string(message)

    st.session_state["class_last"][0] = tiktoken_encodings(class_prompt)
    st.session_state["class_last"][1] = tiktoken_encodings(check)
    st.session_state["llm_last"][0] = tiktoken_encodings(input_prompt)
    st.session_state["llm_last"][1] = tiktoken_encodings(message)
    
    caption = f':blue[Report]: :gray[*Took {(time.time() - start):.2f}s. INPUT {st.session_state["class_last"][0]+st.session_state["llm_last"][0]} tkns. OUTPUT {st.session_state["class_last"][1]+st.session_state["llm_last"][1]} tkns. Cost {round(total_cost*37, 4)} THB*]'
    st.caption(caption)
    if find_bool in [1,2]:
        retrieval_report = render(list_context)
        
        st.session_state['MEMORY_FOR_DISPLAY'].extend([
            {'role': 'user', 'content': question},
            {'role': 'assistant', 'content': message},
            {'role': 'report', 'content': caption},
            {'role': 'retrieval', 'content': list_context}
        ])
    else:
        st.session_state['MEMORY_FOR_DISPLAY'].extend([
            {'role': 'user', 'content': question},
            {'role': 'assistant', 'content': message},
            {'role': 'report', 'content': caption}
        ])
    
    st.session_state['class_sum'] = [sum(x) for x in zip(st.session_state['class_sum'], st.session_state['class_last'])]
    st.session_state['llm_sum'] = [sum(x) for x in zip(st.session_state['llm_sum'], st.session_state['llm_last'])]
    
    # upload db second step
    db.reset()
    db.question = question
    db.result = message
    db.model_name = model_result
    db.input_prompt = input_prompt
    db.input_token = tiktoken_encodings(input_prompt)
    db.output_token = tiktoken_encodings(message)
    if find_bool == 1 :#SSO
        db.database_name = "SSO"
    elif find_bool == 2 :#Sharepoint
        db.database_name = "Sharepoint"
    else : #Other
        db.database_name = "None_find"
    input_cost = cal_token_cost(tiktoken_encodings(input_prompt),model_result,"input")
    output_cost = cal_token_cost(tiktoken_encodings(message),model_result,"output")
    db.cost = input_cost + output_cost
    db.time_stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    db.time_taken_to_answer = (time.time()-start)
    db.step = "result"
    db.insert_into_database()    
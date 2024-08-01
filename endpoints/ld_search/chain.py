import dotenv
import pickle
from langchain.retrievers import ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

dotenv.load_dotenv()

with open("/home/markwh/Documents/larkdown-files/larkdown_docs.pkl", 'rb') as f:
  docs = pickle.load(f)
  
  
child_splitter = RecursiveCharacterTextSplitter(
  chunk_size=400,
)

full_vectorstore = Chroma(
  collection_name = "full_documents", embedding_function = OpenAIEmbeddings()
)

store = InMemoryStore()


retriever = ParentDocumentRetriever(
    vectorstore=full_vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    search_kwargs={"k": 6}
)

retriever.add_documents(docs, ids=None)


llm = ChatOpenAI(model="gpt-4o-mini")

responder_prompt_template = """
You are helping me extract information from my journals. 

The journals are in a format called Larkdown, a private variant of Rmarkdown. 

This means that they will mostly look like markdown documents, with Rmarkdown-flavored code chunks, but with a few new considerations:
  
  1. Larkdown is built to store chat conversations between the human user and an AI. So you may see ">system", ">human", and ">ai" lines delimiting the conversation messages. Note: many Larkdown documents are merely journals for the user and do not utilize the chat syntax, in which case the extranneous system prompt can be ignored. 
  2. Header text may be present before the user begins writing; you can ignore this if it is off topic.

With these considerations in mind, please summarize any information in the document_text that is relevant to the query. Include the source location for the document when providing the info. 

query: {query}

documents: {documents_combined}
""".strip()

responder_prompt = ChatPromptTemplate.from_template(responder_prompt_template)

def dedup_docs(documents):
    unique_documents = {}
    for doc in documents:
        source = doc.metadata.get('source')
        if source not in unique_documents:
            unique_documents[source] = doc
    return list(unique_documents.values())

def combine_docs(documents):
    result = ""
    for doc in documents:
        source = doc.metadata.get("source", "")
        page_content = doc.page_content
        result += "Source:\n" + source + "\n\nContent:\n" + page_content + "\n\n"
    return result


responder = (
  {
    "documents_combined": retriever | dedup_docs | combine_docs,
    "query": RunnablePassthrough(),
  }
  | responder_prompt
  | llm 
  | StrOutputParser()
)

chain = responder
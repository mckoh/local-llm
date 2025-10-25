# %%

from langchain_community.document_loaders import WebBaseLoader, PDFMinerLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from os import getenv

API_KEY = getenv("APIKEY")

prompt = PromptTemplate(
    template="""You are an assistant for question-answering tasks.
    Use the following documents to answer the question.
    If you don't know the answer, just say that you don't know.
    Use three sentences maximum and keep the answer concise:
    Question: {question}
    Documents: {documents}
    Answer:
    """,
    input_variables=["question", "documents"],
)

llm = ChatOllama(
    model="kuffi",
    temperature=0,
)

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250,
    chunk_overlap=0
)

def load_documents():

    urls = [
        "https://www.fh-kufstein.ac.at/pdf-website/studium/ueber-das-studium/satzung/kapitel-1-aspo.pdf",
        "https://www.fh-kufstein.ac.at/pdf-website/studium/ueber-das-studium/satzung/kapitel-5-gleichstellungsplan.pdf"
    ]
    # Load documents from the URLs
    docs = [PDFMinerLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    docs_list = load_documents()


    vectorstore = SKLearnVectorStore.from_documents(
        documents=text_splitter.split_documents(docs_list),
        embedding=OllamaEmbeddings(model="llama3.1:8b"),
    )
    return vectorstore.as_retriever(k=4)

class RAGApplication:
    def __init__(self):
        self.retriever = load_documents()
        self.rag_chain = prompt | llm | StrOutputParser()


    def run(self, question):
        # Retrieve relevant documents
        documents = self.retriever.invoke(question)
        # Extract content from retrieved documents
        doc_texts = "\\n".join([doc.page_content for doc in documents])
        # Get the answer from the language model
        answer = self.rag_chain.invoke({"question": question, "documents": doc_texts})
        return answer

# %%
rag_app = RAGApplication()

# %%
print(rag_app.retriever)

# %%
rag_app.run("How many words can my Bachelor Thesis have?")

# %%

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings

from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


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


class RAGApplication:

    def __init__(self, llm="llama3.1:8b", temp=0):

        self.llm = ChatOllama(
            model=llm,
            temperature=temp,
        )
        self.rag_chain = prompt | self.llm | StrOutputParser()
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=250,
            chunk_overlap=0
        )

        self.docs_list = self.load_documents()
        self.vectorstore = self.create_vectorstore()
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 5,
            }
        )

    def load_documents(self):
        files = [
            "library/kapitel-1-aspo.pdf"
        ]

        docs = [PyPDFLoader(file).load() for file in files]
        return [item for sublist in docs for item in sublist]

    def create_vectorstore(self):

        return SKLearnVectorStore.from_documents(
            documents=self.text_splitter.split_documents(self.docs_list),
            embedding=OllamaEmbeddings(model="llama3.1:8b"),
        )

    def run(self, question):

        documents = self.retriever.invoke(question)

        doc_texts = "\\n".join([doc.page_content for doc in documents])

        answer = self.rag_chain.invoke({"question": question, "documents": doc_texts})
        return answer

# %%
rag_app = RAGApplication()

# %%
dlist = rag_app.retriever.invoke("Wieviel Zeit habe ich beim Verfassen einer Masterarbeit?")

# %%
print(dlist[0].page_content)



# %%
rag_app.run("Welche Anforderungen an Bachelorarbeiten gibt es?")

# =============================
# Imports
# =============================
from euriai.langchain import create_chat_model
from euriai.langchain import EuriaiEmbeddings

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# ⚠️ CRITICAL: import from langchain_core, not langchain.chains
from langchain_core.output_parsers import StrOutputParser

from ragas import evaluate, aevaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset

# =====================================================
# 1. Documents
# =====================================================
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

documents = [
    Document(page_content="Retrieval Augmented Generation combines retrieval with LLMs."),
    Document(page_content="RAG improves factual accuracy using external documents."),
    Document(page_content="LangChain is a framework for building LLM applications."),
]

# =====================================================
# 2. Split Docs
# =====================================================
splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=40)
docs = splitter.split_documents(documents)

# =====================================================
# 3. Vector Store
# =====================================================
embeddings = EuriaiEmbeddings(
    api_key="euri-49e67794160469861d51db4b89da0c40be0457fd1303002d1fbf682ad44c512e",
    model="text-embedding-3-small"
)

vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# =====================================================
# 4. LLM
# =====================================================

#llm = create_chat_model(api_key=api_key, model="gpt-4o-mini", temperature=0.1)
llm = ChatOpenAI(
    model="stepfun/step-3.5-flash:free",
    base_url="https://openrouter.ai/api/v1",
    api_key="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",
)

# =====================================================
# 5. Prompt
# =====================================================
prompt = ChatPromptTemplate.from_template(
    """Answer the question using the following context only.

Context:
{context}

Question:
{question}
"""
)

# =====================================================
# 6. Chain (FIXED)
# =====================================================
rag_chain = (
    {
        "docs": retriever,
        "question": RunnablePassthrough(),
    }
    | RunnableLambda(lambda x: {
        "context": format_docs(x["docs"]),
        "question": x["question"],
        "docs": x["docs"],   # keep raw docs
    })
    | RunnableLambda(lambda x: {
        "answer": llm.invoke(prompt.format(**x)).content,
        "contexts": [doc.page_content for doc in x["docs"]],
    })
)

# =====================================================
# 7. Run RAG
# =====================================================
question = "What is Retrieval Augmented Generation?"
response = rag_chain.invoke(question)

answer = response["answer"]
contexts = response["contexts"]

# =====================================================
# 8. RAGAS Dataset
# =====================================================
dataset = Dataset.from_dict({
    "question": [question],
    "answer": [answer],
    "contexts": [contexts],  # list of list
    "ground_truth": [
        "Retrieval Augmented Generation improves LLM responses by retrieving relevant documents."
    ],
})

# =====================================================
# 9. RAGAS Evaluation
# =====================================================

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

ragas_llm = LangchainLLMWrapper(llm)
ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

scores = evaluate(
    dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ],llm=ragas_llm, embeddings=ragas_embeddings, raise_exceptions=True
)

print("Answer:\n", answer)

print("\nContexts:")
for c in contexts:
    print("-", c)

print("\nRAGAS Scores:\n", scores)
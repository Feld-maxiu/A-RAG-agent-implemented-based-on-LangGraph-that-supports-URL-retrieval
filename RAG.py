import os
import time
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_classic.tools.retriever import create_retriever_tool
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field
from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from openai import RateLimitError

def invoke_with_retry(model, messages, max_retries=5, initial_delay=2):
    """Invoke model with retry logic for rate limits."""
    for attempt in range(max_retries):
        try:
            return model.invoke(messages)
        except RateLimitError as e:
            if attempt < max_retries - 1:
                delay = initial_delay * (2 ** attempt)
                print(f"Rate limited. Waiting {delay} seconds before retry {attempt + 1}/{max_retries}...")
                time.sleep(delay)
            else:
                raise e

# 替换为你的 ModelScope API Key
os.environ["OPENAI_API_KEY"] = ""
API_BASE = "https://api-inference.modelscope.cn/v1"
GRADE_PROMPT = (
    "You are a grader assessing relevance of a retrieved document to a user question. \n "
    "Here is the retrieved document: \n\n {context} \n\n"
    "Here is the user question: {question} \n"
    "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n"
    "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."
)

REWRITE_PROMPT = (
    "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
    "Here is the initial question:"
    "\n ------- \n"
    "{question}"
    "\n ------- \n"
    "Formulate an improved question:"
)

GENERATE_PROMPT = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. "
    "Use three sentences maximum and keep the answer concise.\n"
    "Question: {question} \n"
    "Context: {context}"
)

class GradeDocuments(BaseModel):  
    """Grade documents using a binary score for relevance check."""

    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )


grader_model = ChatOpenAI(
    openai_api_base=API_BASE,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model="deepseek-ai/DeepSeek-V3.2",
    temperature=0,
    max_tokens=2048,
    streaming=False,  
)

def rewrite_question(state: MessagesState):
    """Rewrite the original user question."""
    messages = state["messages"]
    question = messages[0].content
    prompt = REWRITE_PROMPT.format(question=question)
    response = invoke_with_retry(model, [{"role": "user", "content": prompt}])
    return {"messages": [{"role": "user", "content": response.content}]}

def grade_documents(
    state: MessagesState,
) -> Literal["generate_answer", "rewrite_question"]:
    """Determine whether the retrieved documents are relevant to the question."""
    question = state["messages"][0].content
    context = state["messages"][-1].content

    prompt = GRADE_PROMPT.format(question=question, context=context)
    
    try:
        response = (
            grader_model
            .with_structured_output(GradeDocuments).invoke(  
                [{"role": "user", "content": prompt}]
            )
        )
        score = response.binary_score
    except Exception as e:
        print(f"Structured output failed: {e}")
        print("Falling back to simple text response...")
        response_text = invoke_with_retry(grader_model, [{"role": "user", "content": prompt}]).content
        print(f"Raw response: {response_text}")
        
        response_lower = response_text.lower().strip()
        if "yes" in response_lower or "relevant" in response_lower:
            score = "yes"
        else:
            score = "no"

    if score == "yes":
        return "generate_answer"
    else:
        return "rewrite_question"
    

model = ChatOpenAI(
    openai_api_base=API_BASE,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model="deepseek-ai/DeepSeek-V3.2",
    temperature=0.7,
    max_tokens=2048,
    streaming=False,  
)

embeddings = OpenAIEmbeddings(
    openai_api_base="https://api-inference.modelscope.cn/v1",
    openai_api_key="",
    model="Qwen/Qwen3-Embedding-8B",  
)

urls = [
    "https://www2.scut.edu.cn/sse/jxkyryzp/list.htm",
]


docs = []
for url in urls:
    try:
        loader = WebBaseLoader(url)
        doc = loader.load()  # 加载单个网页，返回[Document]列表
        if doc:  # 只添加非空的文档
            docs.extend(doc)  # 直接扩展列表，避免嵌套
            print(f"✅ 成功加载：{url}")
        else:
            print(f"⚠️ 加载为空：{url}")
    except Exception as e:
        print(f"❌ 加载失败 {url}：{str(e)[:100]}")

print(f"\n===== 有效文档数量 =====")
print(f"有效文档数：{len(docs)}")

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500,    
    chunk_overlap=50,  
)

doc_splits = text_splitter.split_documents(docs)
print(f"\n===== 文本分割后的块数量 =====")
print(f"分割后总块数：{len(doc_splits)}")

# ===================== 5. 构建向量库和检索工具 =====================
vectorstore = InMemoryVectorStore.from_documents(
    documents=doc_splits,
    embedding=embeddings
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  


retriever_tool = create_retriever_tool(
    retriever,
    name="your_retriever",  
    description="用于检索华南理工大学软件学院的师资、招生、公告等相关信息，输入相关关键词即可检索。"
)

def generate_query_or_respond(state: MessagesState):
    response = invoke_with_retry(model.bind_tools([retriever_tool]), state["messages"])
    return {"messages": [response]}


def generate_answer(state: MessagesState):
    """Generate an answer."""
    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    response = invoke_with_retry(model, [{"role": "user", "content": prompt}])
    return {"messages": [response]}

workflow = StateGraph(MessagesState)

# Define the nodes we will cycle between
workflow.add_node(generate_query_or_respond)
workflow.add_node("retrieve", ToolNode([retriever_tool]))
workflow.add_node(rewrite_question)
workflow.add_node(generate_answer)

workflow.add_edge(START, "generate_query_or_respond")

# Decide whether to retrieve
workflow.add_conditional_edges(
    "generate_query_or_respond",
    # Assess LLM decision (call `retriever_tool` tool or respond to the user)
    tools_condition,
    {
        # Translate the condition outputs to nodes in our graph
        "tools": "retrieve",
        END: END,
    },
)

# Edges taken after the `action` node is called.
workflow.add_conditional_edges(
    "retrieve",
    # Assess agent decision
    grade_documents,
)
workflow.add_edge("generate_answer", END)
workflow.add_edge("rewrite_question", "generate_query_or_respond")

# Compile
graph = workflow.compile()

for chunk in graph.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": "要当华工软件学院副教授有什么要求?",
            }
        ]
    }
):
    for node, update in chunk.items():
        print("Update from node", node)
        update["messages"][-1].pretty_print()
        print("\n\n")
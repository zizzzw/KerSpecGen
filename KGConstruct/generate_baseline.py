
import os
import json
import re
import time
from pprint import pprint
import argparse
from typing import List
from typing_extensions import TypedDict

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langgraph.graph import END, StateGraph
from neo4j import GraphDatabase
from openai import OpenAI

# --- Constants and Configuration ---

# Set your API keys and credentials here
TAVILY_API_KEY = ""
LANGCHAIN_API_KEY = ""
DASHSCOPE_API_KEY = ""
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "neo4jneo4j"

# Set proxy if needed, otherwise leave empty
HTTP_PROXY = ""
HTTPS_PROXY = ""

# Model configurations
EMBEDDING_MODEL = "BAAI/bge-base-en"
LLM_MODEL = "llama3.1-405b-instruct"
VECTOR_DB_PATH = "../vectordb"
VECTOR_DB_COLLECTION = "seL4_md"


# --- Environment and Initial Setup ---

def setup_environment():
    """Sets up environment variables for APIs and proxies."""
    os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
    os.environ["DASHSCOPE_API_KEY"] = DASHSCOPE_API_KEY
    if HTTP_PROXY:
        os.environ["HTTP_PROXY"] = HTTP_PROXY
    if HTTPS_PROXY:
        os.environ["HTTPS_PROXY"] = HTTPS_PROXY

def initialize_retriever():
    """Initializes the embedding model and the Chroma vector store retriever."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={"device": "cpu"})
    vector_db = Chroma(
        collection_name=VECTOR_DB_COLLECTION,
        embedding_function=embeddings,
        persist_directory=VECTOR_DB_PATH,
    )
    return vector_db.as_retriever(search_kwargs={"k": 7})

# --- Language Model Interaction ---

def get_dashscope_response(messages: List[dict], api_key: str, model: str = LLM_MODEL) -> str:
    """
    Gets a response from the Dashscope API.
    Rotates through a list of API keys from a flag file.
    """
    if not api_key:
        # Fallback to rotating keys if a primary one isn't provided.
        keys = [
            # Add your list of keys here
        ]
        if not keys:
            raise ValueError("Dashscope API key is not set and no keys are available for rotation.")
        
        try:
            with open('./flag.txt', 'r') as file:
                i = int(file.read().strip())
        except (FileNotFoundError, ValueError):
            i = 0
        
        with open('./flag.txt', 'w') as file:
            file.write(str(i + 1))
        
        api_key = keys[i % len(keys)]

    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    completion = client.chat.completions.create(model=model, messages=messages, stream=False)
    return completion.choices[0].message.content

# --- Prompts Definition ---

def define_prompts():
    """Defines all prompts used in the RAG workflow."""
    
    grader_prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {document} \n\n
        Here is the user question: I want to formalize seL4, and the first step is to write specifications based on properties.\n
        Please try to provide some properties of {question}. \n
        If the document is useful for answering the question, or provide the relevant Isabelle code, grade it as useful. \n
        If the document is relevant but not useful, or not relevant at all, grade it as not useful. \n
        This is not a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' to indicate whether the document is useful. \n
        Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
        input_variables=["question", "document"],
    )

    re_write_prompt = PromptTemplate(
        template="""You are a question re-writer that converts an input question to a better version that is optimized \n 
         for vectorstore retrieval. Look at the initial question and formulate an improved one. \n
         Here is the initial question: \n\n I want to formalize the IPC of seL4, and the first step is to write specifications based on properties.\n
        Please try to provide some properties of {question}. \n
        Improved question with no preamble, answer with the improved question only:""",
        input_variables=["question"],
    )

    generation_prompt_no_example = PromptTemplate(
        input_variables=["concept", "context"],
        template="""Please help me with formalizing the seL4 OS by writing specifications based on its properties and behavior. 
        Now, based on the provided context below, suggest properties to specify. 
        Limit suggestions to what can be directly inferred from the context; avoid unnecessary assumptions.
        Concept:
        {concept}
        Context:
        {context}
        Answer:"""
    )

    merge_prompt_no_example = '''
    Merge and refine the suggested properties into concise tips.
    Combine similar or related properties to eliminate duplication.
    Ensure each tip maintains the same level of detail.
    Respond in JSON format as follows:
    {
    "specifications": [
      "Name: Detail",
      "Name: Detail",
      ...
    ]
    }
    Do not include any additional text outside of this JSON structure.
    '''
    
    return grader_prompt, re_write_prompt, generation_prompt_no_example, merge_prompt_no_example

# --- RAG Graph Definition ---

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: The user's question.
        generation: LLM generation.
        documents: A list of documents.
        better_question: A rephrased version of the question.
    """
    question: str
    generation: str
    documents: List[Document]
    better_question: str


def retrieve(state: GraphState, retriever):
    """Retrieve documents from the vector store."""
    print("---RETRIEVE---")
    question = state["question"]
    documents = retriever.get_relevant_documents(question)
    return {"documents": documents, "question": question}

def grade_documents(state: GraphState, grader_prompt):
    """Grade the relevance of retrieved documents."""
    print("---CHECK DOCUMENT RELEVANCE---")
    question = state["question"]
    documents = state["documents"]
    
    filtered_docs = []
    for d in documents:
        prompt = grader_prompt.format(question=question, document=d.page_content)
        messages = [{"role": "user", "content": prompt}]
        try:
            response = get_dashscope_response(messages, DASHSCOPE_API_KEY)
            score = json.loads(response)
            if score.get("score") == "yes":
                print("---GRADE: DOCUMENT USEFUL---")
                filtered_docs.append(d)
            else:
                print("---GRADE: DOCUMENT NOT USEFUL---")
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error grading document: {e}")
            continue
            
    return {"documents": filtered_docs, "question": question}

def _format_docs(docs: List[Document]) -> str:
    """Helper function to format documents for the LLM context."""
    formatted = []
    for doc in docs:
        content = doc.page_content # In the notebook, metadata was used. page_content is more standard.
        paragraph = doc.metadata.get('paragraph', 'N/A')
        section = doc.metadata.get('section', 'N/A')
        formatted.append(f"Paragraph:{paragraph} - Section:{section}\n{content}")
    return '\n\n'.join(formatted)


def generate_simple_rag(state: GraphState):
    """Generate an answer using a simple, single-turn RAG prompt."""
    print("---GENERATE (SIMPLE RAG MODE)---")
    question = state["question"]
    documents = state["documents"]
    
    context = _format_docs(documents)
    
    # Simple generation prompt for baseline
    prompt = f"Please generate an Isabelle/HOL specification on the topic: {question}\nBased on the following documents:\n{context}"
    messages = [
        {"role": "system", "content": "You are a helpful assistant specializing in Isabelle/HOL specifications."},
        {"role": "user", "content": prompt}
    ]
    
    try:
        generation = get_dashscope_response(messages, DASHSCOPE_API_KEY)
    except Exception as e:
        print(f"Error during generation: {e}")
        generation = f"Error generating specification for {question}: {e}"
        
    return {"generation": generation}

def generate_no_example(state: GraphState, generation_prompt: PromptTemplate, merge_prompt: str):
    """Generate an answer using a two-turn conversation for refinement."""
    print("---GENERATE (NO_EXAMPLE MODE)---")
    question = state["question"]
    documents = state["documents"]
    
    context = _format_docs(documents)
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": generation_prompt.format(context=context, concept=question)}
    ]
    
    try:
        # First round of dialogue to get initial properties
        first_generation = get_dashscope_response(messages, DASHSCOPE_API_KEY)
        
        # Second round to merge and refine
        messages.append({"role": "assistant", "content": first_generation})
        messages.append({"role": "user", "content": merge_prompt})
        second_generation = get_dashscope_response(messages, DASHSCOPE_API_KEY)
        generation = second_generation

    except Exception as e:
        print(f"Error during generation: {e}")
        generation = f"Error generating specification for {question}: {e}"
        
    return {"generation": generation}


def transform_query(state: GraphState, rewrite_prompt):
    """Transform the query to a better version for retrieval."""
    print("---TRANSFORM QUERY---")
    question = state.get('better_question') or state["question"]
    
    prompt = rewrite_prompt.format(question=question)
    messages = [{"role": "user", "content": prompt}]
    try:
        better_question = get_dashscope_response(messages, DASHSCOPE_API_KEY)
    except Exception as e:
        print(f"Error transforming query: {e}")
        better_question = question

    return {"documents": state["documents"], "better_question": better_question}

def decide_to_generate(state: GraphState):
    """Decision node to decide whether to generate an answer or re-query."""
    print("---ASSESS GRADED DOCUMENTS---")
    if not state.get("documents") or len(state["documents"]) < 2:
        print("---DECISION: NOT ENOUGH DOCS, TRANSFORM QUERY---")
        return "transform_query"
    else:
        print("---DECISION: GENERATE---")
        return "generate"

def build_rag_workflow(retriever, prompts, mode: str):
    """Builds and compiles the langgraph workflow."""
    grader_prompt, re_write_prompt, gen_prompt_no_ex, merge_prompt_no_ex = prompts

    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve", lambda state: retrieve(state, retriever))
    workflow.add_node("grade_documents", lambda state: grade_documents(state, grader_prompt))
    
    if mode == 'no_example':
        workflow.add_node("generate", lambda state: generate_no_example(state, gen_prompt_no_ex, merge_prompt_no_ex))
    else: # default to simple_rag
        workflow.add_node("generate", generate_simple_rag)
        
    workflow.add_node("transform_query", lambda state: transform_query(state, re_write_prompt))

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {"transform_query": "transform_query", "generate": "generate"},
    )
    workflow.add_edge("transform_query", "retrieve")
    workflow.add_edge("generate", END)

    return workflow.compile()

# --- Neo4j Operations ---

def get_neo4j_driver():
    """Creates and returns a Neo4j driver instance."""
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

def get_leaf_node_queries(driver) -> List[str]:
    """Fetches leaf nodes from Neo4j and formats them as queries."""
    query = """
    MATCH (parent:Query)-[:SPECIFY]->(n:Query)
    WHERE NOT (n)-->()
    RETURN parent.name AS parent_name, n.name AS leaf_node_name
    """
    with driver.session() as session:
        result = session.run(query)
        queries = [
            f"Parent: {record['parent_name']}. Current Concept to specify: {record['leaf_node_name']}"
            for record in result
        ]
    return queries

# --- Main Execution Logic ---

def run_rag_and_save(app, query: str, output_dir: str):
    """
    Runs the RAG pipeline for a single query and saves the result.
    """
    print(f"\n--- Running RAG for query: {query} ---")
    inputs = {"question": query}
    final_state = {}
    for output in app.stream(inputs):
        for key, value in output.items():
            print(f"Node '{key}':")
        pprint("---")
        final_state = value

    generation = final_state.get("generation", "No generation produced.")
    
    # Sanitize query to create a valid filename
    sanitized_query = re.sub(r'[\\/*?:"<>|]', "", query)
    output_path = os.path.join(output_dir, f"{sanitized_query}.txt")
    
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(generation)
    print(f"--- Saved baseline specification to: {output_path} ---")


def main():
    """Main function to run the baseline generation pipeline."""
    parser = argparse.ArgumentParser(description="Generate baseline specifications using a RAG pipeline.")
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["simple_rag", "no_example"], 
        default="simple_rag", 
        help="The generation mode to use."
    )
    args = parser.parse_args()

    print(f"--- Running in {args.mode} mode ---")

    setup_environment()
    
    # Initialize components
    retriever = initialize_retriever()
    prompts = define_prompts()
    rag_app = build_rag_workflow(retriever, prompts, args.mode)
    
    # Get initial queries from Neo4j
    try:
        neo4j_driver = get_neo4j_driver()
        queries = get_leaf_node_queries(neo4j_driver)
        neo4j_driver.close()
        print(f"--- Found {len(queries)} queries from Neo4j ---")
    except Exception as e:
        print(f"Could not connect to Neo4j to get queries: {e}")
        print("Using a default list of queries.")
        queries = ["Kernel Heap Access", "IPC", "Untyped Memory"]

    # Define output directory based on mode
    output_dir = f"./baseline/{args.mode}"

    # Run the RAG process for each query
    for query in queries:
        run_rag_and_save(rag_app, query, output_dir)
    
    print("\n--- Baseline generation pipeline finished. ---")


if __name__ == "__main__":
    main()


import os
import json
import re
import time
from pprint import pprint
import glob
from typing import List
from typing_extensions import TypedDict

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain.schema import Document
from langchain.chains import ConversationChain, LLMChain
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langgraph.graph import END, StateGraph
from neo4j import GraphDatabase
from openai import OpenAI
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector


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
LOCAL_LLM_MODEL = "llama3.1:70b" # For local model if ever used
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

def clear_proxies():
    """Clears proxy settings."""
    os.environ['HTTP_PROXY'] = ''
    os.environ['HTTPS_PROXY'] = ''

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

    Args:
        messages: A list of messages for the chat completion.
        api_key: The Dashscope API key.
        model: The model to use.

    Returns:
        The content of the response message.
    """
    if not api_key:
        raise ValueError("Dashscope API key is not set.")
    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    completion = client.chat.completions.create(model=model, messages=messages)
    return completion.choices[0].message.content


# --- Prompts Definition ---

def define_prompts():
    """Defines all prompts used in the RAG workflow."""
    
    # Grader Prompt
    grader_prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {document} \n\n
        Here is the user question: I want to formalize seL4, and the first step is to write specifications based on properties.\n\n
        Please try to provide some properties of {question}. \n\n
        If the document is useful for answering the question, or provide the relevent Isabelle code, grade it as useful. \n\n
        If the document is relevant but not useful, or not relevant at all, grade it as not useful. \n\n
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n\n
        Give a binary score 'yes' or 'no' score to indicate whether the document is useful to the question. \n\n
        Provide the binary score as a JSON with a single key 'score' and no premable or explanation.""",
        input_variables=["question", "document"],
    )

    # Distiller Prompt
    distill_prompt = PromptTemplate(
        template="""You are an assistant specialized in information distillation. \n 
        You are given a document and a question.
        Here is the document: \n\n {document} \n\n
        Here is the question: Please try to provide some properties of {question}. \n\n
        Here is what you need to do:
        Please distill the document while retaining all information that is useful for answering the question.
        Please answer in concise language while ensuring that as little useful information as possible is lost.
        Provide the answer with and only with the distilled documents,no openning or closing statement.""",
        input_variables=["question", "document"],
    )

    # Question Re-writer Prompt
    re_write_prompt = PromptTemplate(
        template="""You a question re-writer that converts an input question to a better version that is optimized \n 
         for vectorstore retrieval. Look at the initial and formulate an improved question. \n\n
         Here is the initial question: \n\n I want to formalize the IPC of seL4, and the first step is to write specifications based on properties.\n\n
        Please try to provide some properties of {question}. Improved question with no preamble, answer with the improved question only: \n """,
        input_variables=["question"],
    )

    # Few-shot examples for generation
    examples = [
        {
        "concept": "Activating Threads",
        "context": """
        Threads that are active always have a master Reply capability to themselves stored in their reply slot.
        This is so that a derived Reply capability can be generated immediately if they wish to issue one.
        This function sets up a new master Reply capability if one does not exist.
        definition
        "setup_reply_master thread ≡ do
        old_cap <- get_cap (thread, tcb_cnode_index 2);
        when (old_cap = NullCap) $ do
        set_original (thread, tcb_cnode_index 2) True;
        set_cap (ReplyCap thread True) (thread, tcb_cnode_index 2)
        od
        od"

        Reactivate a thread if it is not already running.
        definition
        restart :: "obj_ref ⇒ (unit,’z::state_ext) s_monad" where
        "restart thread ≡ do
        state ← get_thread_state thread;
        when (¬ runnable state ∧ ¬ idle state) $ do
        ipc_cancel thread;
        setup_reply_master thread;
        set_thread_state thread Restart;
        do_extended_op (tcb_sched_action (tcb_sched_enqueue) thread);
        do_extended_op (switch_if_required_to thread)
        od
        od"
        """,
        "answer":"""
        Reply Capability Setup: Active threads have a master Reply capability stored in their reply slot. This is so that a derived Reply capability can be generated immediately if they wish to issue one. If a master Reply capability does not exist, the function sets up a new one.
        Thread Reactivation: Reactivates a thread if it is not already running by canceling IPC, setting up a master Reply capability, setting the thread state to Restart, and re-enqueuing the thread with the scheduler.
        """
        },
        {
        "concept": "",
        "context": """
        definition
        get_message_info :: "obj_ref ⇒ (message_info,’z::state_ext) s_monad"
        where
        "get_message_info thread ≡ do
        x ← as_user thread $ get_register msg_info_register;
        return $ data_to_message_info x
        od"
        """,
        "answer":"""
        retrieves the message information of a thread by accessing the message info register and converting the data into a message info structure."""
        },
        {
        "concept": "Generic functions on top of the state monad",
        "context": """
        Apply a function to the current state and return the result without changing the state.
        definition
        gets :: "(’s ⇒ ’a) ⇒ (’s, ’a) nondet_monad" where
        "gets f ≡ get >>= (λs. return (f s))"
        Modify the current state using the function passed in.
        definition
        modify :: "(’s ⇒ ’s) ⇒ (’s, unit) nondet_monad" where
        "modify f ≡ get >>= (λs. put (f s))"
        lemma simpler_gets_def: "gets f = (λs. ({{(f s, s)}}, False))"
        lemma simpler_modify_def:
        "modify f = (λs. ({{((), f s)}}, False))"
        Execute the given monad when the condition is true, return () otherwise.
        definition
        when :: "bool ⇒ (’s, unit) nondet_monad ⇒
        (’s, unit) nondet_monad" where
        "when P m ≡ if P then m else return ()"
        Execute the given monad unless the condition is true, return () otherwise.
        definition
        unless :: "bool ⇒ (’s, unit) nondet_monad ⇒
        (’s, unit) nondet_monad" where
        "unless P m ≡ when (¬P) m"
        Perform a test on the current state, performing the left monad if the result is true or the right monad
        if the result is false.
        definition
        condition :: "(’s ⇒ bool)
        ⇒ (’s, ’r) nondet_monad
        ⇒ (’s, ’r) nondet_monad
        ⇒ (’s, ’r) nondet_monad"
        where
        "(condition P L R) ≡ λs. if (P s) then (L s) else (R s)"
        notation (output)
        condition ("(condition (_)// (_)// (_))" [1000,1000,1000] 1000)
        Apply an option valued function to the current state, fail if it returns None, return v if it returns Some
        v.
        definition
        gets_the :: "(’s ⇒ ’a option) ⇒ (’s, ’a) nondet_monad" where
        "gets_the f ≡ gets f >>= assert_opt"
        """,
        "answer":"""
        Applying Functions to State: This involves applying a function to the current state to retrieve a result without changing the state. It enables querying the state in a nondestructive manner.
        Modifying State: This operation modifies the current state using a provided function. It ensures that the state can be updated and maintained accurately within the monad.
        Conditional Execution: These operations execute monads based on specified conditions. They facilitate branching logic within computations, allowing for different actions based on the state or provided conditions.
        State-Dependent Execution: This involves performing tests on the current state to decide which monad to execute. It ensures that computations can adapt based on the state, allowing for flexible and context-aware operations.
        Applying Optional Functions to State: This operation applies functions that return optional values to the state. It ensures that computations only proceed with valid data, handling cases where data might be missing.
        """
        },
        {
        "concept": "Abstract Specification Instantiations",
        "context": """
        theory Deterministic_A
        imports
        Structures_A
        "../../lib/List_Lib"
        begin
        The kernel specification operates over states of type ’a state, which includes all of the abstract
        kernel state plus an extra field, exst of type ’a. By choosing an appropriate concrete type for ’a,
        we obtain different instantiations of this specification, at differing levels of abstraction. The abstract
        specification is thus extensible. The basic technique, and its motivation, are described in [14].
        Here, we define two such instantiations. The first yields a largely-deterministic specification by instantiating ’a with a record that includes concrete scheduler state and information about sibling ordering
        in the capability derivation tree (CDT). We call the resulting specification the deterministic abstract
        specification and it is defined below in section 19.1.
        The second instantiation uses the type unit for ’a, yielding a specification that is far more nondeterministic. In particular, the scheduling behaviour and the order in which capabilities are deleted during
        a revoke system call both become completely nondeterministic. We call this second instantiation the
        nondeterministic abstract specification and it is defined below in section 19.2.
        Translate a state of type ’a state to one of type ’b state via a function t from ’a to ’b.
        definition trans_state :: "(’a ⇒ ’b) ⇒ ’a state ⇒ ’b state" where
        "trans_state t s = (|kheap = kheap s, cdt = cdt s, is_original_cap = is_original_cap s,
        cur_thread = cur_thread s, idle_thread = idle_thread s,
        machine_state = machine_state s,
        interrupt_irq_node = interrupt_irq_node s,
        interrupt_states = interrupt_states s, arch_state = arch_state s,
        exst = t(exst s)|)"
        Truncate an extended state of type ’a state by effectively throwing away all the ’a information.
        abbreviation "truncate_state ≡ trans_state (λ_. ())"
        """,
        "answer":"""
        Deterministic Abstract Specification: The kernel specification operates over states of type 'a state, which includes all of the abstract kernel state plus an extra field, exst of type 'a. By choosing an appropriate concrete type for 'a, different instantiations of the specification at varying levels of abstraction are obtained. This specification results from instantiating 'a with a record that includes concrete scheduler state and information about sibling ordering in the capability derivation tree (CDT), yielding a largely-deterministic specification.
        Nondeterministic Abstract Specification: This specification results from using the type unit for 'a, yielding a specification with highly nondeterministic behavior, particularly in scheduling and capability deletion order during a revoke system call.
        State Transformation: Transforms the state by applying a given transformation function to the extension part of the state while keeping all other parts unchanged.
        State Truncation: Transformation of the state by truncating the extension part to a unit value, effectively discarding any extra information while keeping all other parts of the state unchanged.
        """
        },
        {
        "concept":"",
        "context": """
        Capturing the protection state in such a policy simplifies reasoning in three ways. First,the number of labels is typically smaller than the number of objects in the system-depending on the system,a label can contain thousands of objects.Second, when using the policy to reason about untrusted subjects,the policy will typically be static over each system call whereas the protection state will typically change.Finally, we can formulate the allowed effects of operations of untrusted subjects by consulting the policy rather than the more complex protection state. 
        The access rights between labels in our model are:Receive,SyncSend,AsyncSend, Reset,Grant,Write,Read,and Control.We make a distinction between synchronous and asynchronous send,because the former has different implications for confidential- ity as explained later in Section 5.3.The two other nonstandard rights are Reset and Control.The former is the authority to reset an object to its initial state for re-using resources.The latter confers the authority to completely determine the behaviour of the object,for example,by writing to a thread's registers.The Create right known from the classical take-grant models [Lipton and Snyder 1977]is subsumed here by Control.
        """,
        "answer":"""
        Access Control Policy: The policy simplifies reasoning by using labels instead of objects, which are fewer in number. This static policy helps in reasoning about untrusted subjects, whose protection state changes dynamically.
        Access Rights Between Labels: The access rights include Receive, SyncSend, AsyncSend, Reset, Grant, Write, Read, and Control. Synchronous and asynchronous send are distinguished due to their different confidentiality implications. Reset allows resetting an object to its initial state, while Control grants authority to determine the behavior of an object. The Create right is subsumed by Control in this model.
        """
        },
        {
        "concept": "Non-Interference",
        "context": """More well-defined than general information flow is the property of non-interference (Goguen and Meseguer, 1982, 1984; Rushby, 1992). This property is able to capture indirect information flows in a model if the code for both sides is known. Given a secret process S and an attacker A, it requires one to prove that the behaviour of A cannot depend on the actions of S, i.e. that S does not interfere with the behaviour of A. It is possible to establish this property by looking at execution traces of the system: Given a trace where execution of both S and A occurs, one needs to show that the result in A is the same when all S actions are removed from the trace. Unfortunately, this property is not necessarily preserved by the proof technique of refinement which is the one most commonly used to relate an implementation to its model (Jacob, 1989). This means that proving non-interference of the model only implies non-interference of the implementation when special care is taken in the implementation proof. The standard access control models do not have this problem.""",
        "answer": """Non-Interference: Capture indirect information flows in a model when the code for both the secret process (S) and the attacker (A) is known. It requires proving that the behavior of A does not depend on the actions of S, meaning S does not interfere with A. This can be established by examining execution traces and showing that the result in A remains the same when all S actions are removed."""
        }
    ]


    
    hf_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={"device": "cpu"})
    
    example_selector = SemanticSimilarityExampleSelector.from_examples(
        examples,
        hf_embeddings,
        Chroma,
        k=2,
        collection_name="rag_examples"
    )

    example_prompt = PromptTemplate(input_variables=["concept", "context", 'answer'], template="Concept:\n{concept}\nContext:\n{context}\nAnswer:\n{answer}")

    generation_prompt = FewShotPromptTemplate(
        example_prompt=example_prompt,
        example_selector=example_selector,
        prefix="""Please help me with formalizing the seL4 OS by writing specifications based on its properties and behavior. Here is an example:""",
        suffix='Now, based on the provided context below, suggest properties to specify. Limit suggestions to what can be directly inferred from the context; avoid unnecessary assumptions.  \n Concept:\n{concept}\n Context:\n{context}\n Answer:',
        input_variables=["concept", 'context']
    )

    # Merge Prompt
    merge_prompt = '''
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
    
    return grader_prompt, distill_prompt, re_write_prompt, generation_prompt, merge_prompt

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

# Node functions for the graph
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

def generate(state: GraphState, generation_prompt, merge_prompt):
    """Generate an answer using the RAG setup."""
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    
    def format_docs(docs: List[Document]) -> str:
        return "\n\n".join(doc.page_content for doc in docs)

    context = format_docs(documents)
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": generation_prompt.format(context=context, concept=question)}
    ]
    
    try:
        first_generation = get_dashscope_response(messages, DASHSCOPE_API_KEY)
        messages.append({"role": "assistant", "content": first_generation})
        messages.append({"role": "user", "content": merge_prompt})
        second_generation = get_dashscope_response(messages, DASHSCOPE_API_KEY)
        generation = second_generation
    except Exception as e:
        print(f"Error during generation: {e}")
        generation = ""
        
    return {"documents": documents, "question": question, "generation": generation}

def transform_query(state: GraphState, re_write_prompt):
    """Transform the query to a better version for retrieval."""
    print("---TRANSFORM QUERY---")
    question = state.get('better_question') or state["question"]
    
    prompt = re_write_prompt.format(question=question)
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
    if not state["documents"] or len(state["documents"]) < 2:
        print("---DECISION: NEED MORE DOCS, TRANSFORM QUERY---")
        return "transform_query"
    else:
        print("---DECISION: GENERATE---")
        return "generate"

def build_rag_workflow(retriever, prompts):
    """Builds and compiles the langgraph workflow."""
    grader_prompt, _, re_write_prompt, generation_prompt, merge_prompt = prompts

    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("retrieve", lambda state: retrieve(state, retriever))
    workflow.add_node("grade_documents", lambda state: grade_documents(state, grader_prompt))
    workflow.add_node("generate", lambda state: generate(state, generation_prompt, merge_prompt))
    workflow.add_node("transform_query", lambda state: transform_query(state, re_write_prompt))

    # Build graph
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {"transform_query": "transform_query", "generate": "generate"},
    )
    workflow.add_edge("transform_query", "retrieve") # Re-retrieve with better question
    workflow.add_edge("generate", END)

    return workflow.compile()


# --- Neo4j Operations ---

def get_neo4j_driver():
    """Creates and returns a Neo4j driver instance."""
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

def write_queries_to_neo4j(driver, parent_query, sub_queries):
    """Writes parent and sub-queries to Neo4j."""
    def add_query_and_rel(tx, parent, child):
        tx.run(
            "MERGE (q:Query {name: $parent}) "
            "MERGE (sq:Query {name: $child}) "
            "MERGE (q)-[:SPECIFY]->(sq)",
            parent=parent, child=child
        )
    
    with driver.session() as session:
        for sub_query in sub_queries:
            if parent_query != sub_query:
                session.write_transaction(add_query_and_rel, parent_query, sub_query)

def get_leaf_nodes_with_parents(driver):
    """Fetches leaf nodes and their parents from Neo4j."""
    query = """
    MATCH (parent:Query)-[:SPECIFY]->(n:Query)
    WHERE NOT (n)-->()
    RETURN parent.name AS parent_name, n.name AS leaf_node_name
    """
    with driver.session() as session:
        result = session.run(query)
        queries = []
        log_name_map = {}
        for record in result:
            parent_name = record['parent_name']
            leaf_name = record['leaf_node_name']
            q = f"Parent: {parent_name}. Current Concept to specify: {leaf_name}"
            
            # This logic was in the notebook to map concepts to log file names
            if parent_name == 'seL4':
                log_name_map[q] = leaf_name
            elif parent_name in ['The Monad','Adding Exceptions','Catching and Handling Exceptions','Hoare Logic','Syntax']:
                 log_name_map[q] = 'Nondeterministic State Monad with Failure'
            else:
                 log_name_map[q] = parent_name
            queries.append(q)
    return queries, log_name_map

# --- Main Execution Logic ---

def extract_sub_queries(result_json_str: str) -> List[str]:
    """Extracts specifications from the generated JSON string."""
    try:
        data = json.loads(result_json_str)
        return data.get("specifications", [])
    except json.JSONDecodeError:
        return []

def run_recursive_rag(app, driver, query, log_name_map, depth=0, max_depth=1):
    """Runs the RAG workflow recursively."""
    if depth >= max_depth:
        return

    print(f"\n--- Running RAG for query (Depth {depth}): {query} ---")
    inputs = {"question": query}
    final_state = {}
    for output in app.stream(inputs):
        for key, value in output.items():
            print(f"Node '{key}':")
        final_state = value
    
    result = final_state.get("generation", "")
    new_queries = extract_sub_queries(result)
    
    if not new_queries or len(new_queries) <= 1:
        return

    # Log results
    log_dir = "./KG/llama405b"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{log_name_map.get(query, 'unknown')}.txt")
    with open(log_file, "a") as f:
        f.write(f"Depth: {depth}, Query:\n{query}\nResult:\n")
        for q in new_queries:
            f.write(f"{q}\n")

    # Update Neo4j and recurse
    match = re.search(r"Current Concept to specify: (?P<concept>.+)$", query)
    parent_query = match.group("concept") if match else query
    
    write_queries_to_neo4j(driver, parent_query, new_queries)

    for new_q in new_queries:
        run_recursive_rag(app, driver, new_q, log_name_map, depth + 1, max_depth)


def main():
    """Main function to run the CRAG pipeline."""
    setup_environment()
    
    # Initialize components
    retriever = initialize_retriever()
    prompts = define_prompts()
    rag_app = build_rag_workflow(retriever, prompts)
    neo4j_driver = get_neo4j_driver()

    # Get initial queries from Neo4j
    queries, log_name_map = get_leaf_nodes_with_parents(neo4j_driver)

    # Run the recursive RAG process
    for query in queries:
        run_recursive_rag(rag_app, neo4j_driver, query, log_name_map)
    
    neo4j_driver.close()
    print("\n--- Pipeline execution finished. ---")


if __name__ == "__main__":
    main()

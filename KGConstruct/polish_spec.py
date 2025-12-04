import os
import json
import time
from neo4j import GraphDatabase
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.chat_models import ChatOllama
from langchain.chains import ConversationChain, LLMChain
from langchain_openai import ChatOpenAI
from openai import OpenAI

# --- Constants and Configuration ---

# Set your API keys and endpoints here or via environment variables
os.environ["TAVILY_API_KEY"] = ""
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["OPENAI_API_KEY"] = "" # Used for ChatOpenAI

# It's recommended to set these as environment variables
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4jneo4j")

# Path to the file containing properties extracted from code
PROPERTIES_FILE_PATH = "../PropertyExtract/output-qwen-max.jsonl"
# Directory to save the polished specification trees
OUTPUT_DIR = "./polish/llama405b/"

# Mapping of specification nodes to their corresponding source files
NODE_TO_FILE_MAPPING = {
    "Platform Definitions": "./spec/machine/ARM/Platform.thy",
    "ARM Machine Types": "./spec/design/m-skel/ARM/MachineTypes.thy",
    "Machine Types": "./spec/design/m-skel/ARM/MachineTypes.thy",
    "Kernel Events": "./spec/design/skel/Event_H.thy",
    "Common, Architecture-Specific Data Types": "./spec/design/skel/ARM/Arch_Structs_B.thy",
    "ARM Machine Instantiation": "./spec/abstract/ARM/Machine_A.thy",
    "Machine Accessor Functions": "./spec/abstract/MiscMachine_A.thy",
    "Error and Fault Messages": "./spec/abstract/ExceptionTypes_A.thy",
    "Access Rights": "./spec/abstract/CapRights_A.thy",
    "ARM-Specific Virtual-Memory Rights": "./spec/abstract/VMRights_A.thy",
    "ARM-Specific Data Types": "./spec/abstract/ARM/Arch_Structs_A.thy",
    "Machine Operations": "./spec/machine/ARM/MachineOps.thy",
    "Basic Data Structures": "./spec/abstract/Structures_A.thy",
    "Abstract Specification Instantiations": "./spec/abstract/Deterministic_A.thy",
    "Basic Kernel and Exception Monads": "./spec/abstract/Exceptions_A.thy",
    "Accessing the Kernel Heap": "./spec/abstract/KHeap_A.thy",
    "Accessing CSpace": "./spec/abstract/CSpaceAcc_A.thy",
    "Accessing the ARM VSpace": "./spec/abstract/ARM/ArchVSpaceAcc_A.thy",
    "ARM Object Invocations": "./spec/abstract/ARM/ArchInvocation_A.thy",
    "Kernel Object Invocations": "./spec/abstract/InvocationLabels_A.thy",
    "Retyping and Untyped Invocations": "./spec/abstract/Retype_A.thy",
    "ARM VSpace Functions": "./spec/abstract/ARM/ArchVSpace_A.thy",
    "IPC Cancelling": "./spec/abstract/IpcCancel_A.thy",
    "CSpace": "./spec/abstract/CSpace_A.thy",
    "Toplevel ARM Definitions": "./spec/abstract/ARM/Arch_A.thy",
    "Scheduler": "./spec/abstract/Schedule_A.thy",
    "Threads and TCBs": "./spec/abstract/Tcb_A.thy",
    "IPC": "./spec/sep-abstract/Ipc_SA.thy",
    "Interrupts": "./spec/abstract/Interrupt_A.thy",
    "Kernel Invocation Labels": "./spec/design/skel/InvocationLabels_H.thy",
    "Decoding System Calls": "./spec/abstract/Decode_A.thy",
    "An Initial Kernel State": "./spec/abstract/ARM/Init_A.thy",
    "System Calls": "./spec/abstract/Syscall_A.thy",
    "Nondeterministic State Monad with Failure": "./spec/abstract/Nondeterministic State Monad with Failure.thy"
}

# --- LLM Prompts ---

REDUCE_DUPLICATES_PROMPT = """
Please eliminate any duplicate specifications from the following Specification Tree, which is a hierarchical structure where the leaf nodes represent individual specifications and the root node represents the overarching topic.
Please note:
i. Only delete a specification when the semantics are exactly the same.
ii. Only determine specification repeatability between nodes at the same level under the same parent node.
iii. Do not change the highest level of the Specification Tree.
(Code) or (Both) tag:
i. if you merge a (Code)/(Both) and a No-tag specification, reserve (Code)/(Both).
ii. if you merge a (Code) and a (Both) specification, reserve (Both).
Your response should strictly follow the JSON format of the provided Specification Tree (each specification is in the "name:description" format), and respond only with the updated Specification Tree, with no additional explanation or statement.
{specification}
[Updated Specification Tree]:
"""

IDENTIFY_PURPOSE_PROMPT = """
Please summarize the following specifications into one specification (named [code-spec]) in JSON format.
[code-spec] example 1:
{{
  "name": "Underspecification of Behaviour: The model leaves as much behaviour as possible underspecified, making fewer assumptions about the hardware."
}}
[code-spec] example 2:
{{
  "name": "Restriction of Side Effects: Functions in the machine interface are restricted to only affect the machine_state component of the system."
}}
Note that you should only learn the format from the examples; do not use their information to summarize the specification. Each [code-spec] should have a name and an explanation, combined as "name:explanation". Your answer should be a JSON object with a single key "name" and a value of the [code-spec] in the "name:explanation" format.
Specifications to be summarized into one ([code-spec]):
{code}
"""

IDENTIFY_IMPLEMENTED_SPEC_PROMPT = """
Please review the following [Specification Tree] and determine if the [code-spec] is already included in the tree.
- If the [code-spec] is similar to any specification (referred to as [specification A]) in the [Specification Tree], please reply with only the following JSON format, with no additional explanation or statement:
{{
    "name":"(Both)[specification A]"
}}
- If the specification is not similar to any existing specification, please reply with only the following JSON format:
{{
    "name":"(Code)content of [code-spec]"
}}
[Specification Tree]:
{specification}
"""

UPDATE_SPEC_TREE_PROMPT = """
[Origin Specification Tree] maintains a tree structure among specifications, with nodes at the same depth having similar granularity.
Please update the [Origin Specification Tree] according to the [contexts].
- If a specification A in the [contexts] starts with (Code), add "(Code)specification A" to the [Origin Specification Tree] at the appropriate level.
- If a specification A in the [contexts] starts with (Both), add (Both) to the corresponding specification in the [Origin Specification Tree].
Respond only with the final updated [Origin Specification Tree] in JSON format, with no additional commentary. Do not change the highest level of the [Origin Specification Tree].
Make sure that your answer follows the following format, containing "name" and "children" keys:
{{
  "name": "",
  "children": [
    {{
      "name": "",
      "children": []
    }},
    {{
      "name": "",
      "children": []
    }}
  ]
}}
[Origin Specification Tree]:
{specification}
[contexts]:
{context}
"""

# --- Neo4j Database Functions ---

def get_specify_related_nodes(tx, root_node="seL4"):
    """Fetches all nodes directly connected to the root node via a 'SPECIFY' relationship."""
    query = "MATCH (n {name: $root_node})-[:SPECIFY]->(relatedNode) RETURN relatedNode.name AS name"
    result = tx.run(query, root_node=root_node)
    return [record["name"] for record in result]

def _get_tree_recursive(tx, root_node_name):
    """Helper function to recursively build a tree from the graph."""
    query = """
    MATCH (n {name: $root_node_name})
    OPTIONAL MATCH (n)-[:SPECIFY]->(child)
    RETURN n.name AS name, collect(child.name) AS children
    """
    result = tx.run(query, root_node_name=root_node_name)
    record = result.single()
    if record:
        name = record["name"]
        children_names = [child for child in record["children"] if child is not None]
        children = [_get_tree_recursive(tx, child_name) for child_name in children_names]
        return {"name": name, "children": children}
    else:
        return {"name": root_node_name, "children": []}

def get_spec_tree(driver, root_node_name):
    """Retrieves a specification tree from Neo4j starting from a root node."""
    with driver.session() as session:
        tree = session.read_transaction(_get_tree_recursive, root_node_name)
        return json.dumps(tree, indent=2, ensure_ascii=False)

def _delete_children_recursive(tx, node_name):
    """Helper function to recursively delete all children of a node."""
    query = "MATCH (parent:Query {name: $node_name})-[:SPECIFY]->(child) RETURN child"
    result = tx.run(query, node_name=node_name)
    children = [record["child"] for record in result]
    for child in children:
        child_name = child['name']
        _delete_children_recursive(tx, child_name)
        tx.run("MATCH (n:Query {name: $name}) DETACH DELETE n", name=child_name)

def delete_spec_children(driver, node_name):
    """Deletes all children of a given node, but not the node itself."""
    with driver.session() as session:
        session.write_transaction(_delete_children_recursive, node_name)

def _insert_tree_recursive(tx, tree, parent=None):
    """Helper function to recursively insert a tree structure into the graph."""
    node_name = tree["name"]
    tx.run("MERGE (n:Query {name: $name})", name=node_name)
    if parent:
        relation_query = """
        MATCH (parent:Query {name: $parent_name}), (child:Query {name: $child_name})
        CREATE (parent)-[:SPECIFY]->(child)
        """
        tx.run(relation_query, parent_name=parent, child_name=node_name)
    for child in tree.get("children", []):
        _insert_tree_recursive(tx, child, node_name)

def insert_spec_tree(driver, updated_spec_tree_json):
    """Parses a JSON tree and inserts it into the Neo4j database."""
    tree = json.loads(updated_spec_tree_json)
    with driver.session() as session:
        session.write_transaction(_insert_tree_recursive, tree)

# --- Core Logic Functions ---

def get_properties_for_node(node_name, file_path):
    """Reads the properties file and returns a list of properties for a given node."""
    property_list = []
    target_file = NODE_TO_FILE_MAPPING.get(node_name, '')
    if not target_file:
        return []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                if data.get('title') == target_file:
                    property_list.append({"property": data['property']})
    except FileNotFoundError:
        print(f"Warning: Properties file not found at {file_path}")
    return property_list

def setup_llms():
    """Initializes and returns the language models for the pipeline."""
    # This model is used for general conversation and summarization steps.
    llm = ChatOpenAI(
        model="llama3.1-405b-instruct",
        temperature=0,
        max_tokens=None,
        timeout=300,
        max_retries=2,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    # This model is used for tasks that require strict JSON output.
    json_llm = ChatOllama(model='llama3.1:70b', temperature=0, format='json', num_ctx=12288)
    
    # This client is used for streaming responses.
    openai_client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    return llm, json_llm, openai_client

def formalize_specifications(context_list, json_tree, llm, json_llm, openai_client):
    """
    Takes a list of code-related contexts and an existing specification tree,
    and uses LLMs to formalize, merge, and update the tree.
    """
    useful_context = ""

    for context in context_list:
        memory = ConversationBufferWindowMemory(k=2)
        conversation = ConversationChain(llm=llm, memory=memory, verbose=False)

        # Step 1: Identify the purpose of the Isabelle code property.
        prompt_purpose = PromptTemplate(input_variables=["code"], template=IDENTIFY_PURPOSE_PROMPT)
        purpose_response = conversation.predict(input=prompt_purpose.format(code=context))
        print("Purpose Response:", purpose_response)
        time.sleep(15)

        # Step 2: Identify if the summarized spec is already implemented in the tree.
        prompt_identify_spec = PromptTemplate(input_variables=["specification"], template=IDENTIFY_IMPLEMENTED_SPEC_PROMPT)
        implemented_spec_response = conversation.predict(input=prompt_identify_spec.format(specification=json_tree))
        print("Implemented Specifications Response:", implemented_spec_response)
        useful_context += f"context:{implemented_spec_response}\\n"
        time.sleep(15)
      
    # Step 3: Update the specification tree with the new contexts.
    update_spec_tree_str = UPDATE_SPEC_TREE_PROMPT.format(specification=json_tree, context=useful_context)
    
    print("Updating specification tree...")
    full_reply_content = ""
    completion = openai_client.chat.completions.create(
        model="llama3.1-405b-instruct",
        messages=[{'role': 'user', 'content': update_spec_tree_str}],
        stream=True
    )
    for chunk in completion:
        delta = chunk.choices[0].delta
        if hasattr(delta, 'content') and delta.content is not None:
            full_reply_content += delta.content
            print(delta.content, end='', flush=True)
    
    updated_spec_tree_response = full_reply_content.strip()
    print("\nUpdated Tree Response Received.")

    # Step 4: Reduce duplicates in the newly generated tree.
    reduce_duplicates_chain = LLMChain(
        llm=json_llm,
        prompt=PromptTemplate(input_variables=["specification"], template=REDUCE_DUPLICATES_PROMPT)
    )
    try:
        spec_tree_response = reduce_duplicates_chain.run(specification=updated_spec_tree_response)
        print("Reduced Specification Tree:", spec_tree_response)
    except Exception as e:
        print(f"Error during duplicate reduction: {e}")
        # Log the failed prompt for debugging
        with open('log.txt', 'a') as f:
            f.write(update_spec_tree_str + "\\n---\\n")
        return None
    
    return spec_tree_response

# --- Main Execution ---

def main():
    """Main function to run the specification polishing pipeline."""
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    driver = None
    try:
        # Initialize database driver and LLMs
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        driver.verify_connectivity()
        print("Successfully connected to Neo4j.")
        
        llm, json_llm, openai_client = setup_llms()
        
        # Get the list of nodes to process
        with driver.session() as session:
            nodes_to_process = session.read_transaction(get_specify_related_nodes, "seL4")
        
        print(f"Found {len(nodes_to_process)} nodes to process: {nodes_to_process}")

        for node in nodes_to_process:
            print(f"--- Processing node: {node} ---")
            
            # Get properties (code snippets) related to the current node
            properties = get_properties_for_node(node, PROPERTIES_FILE_PATH)
            if not properties:
                print(f"No properties found for node '{node}'. Skipping.")
                continue
            
            print(f"Found {len(properties)} properties for '{node}'.")
            
            # Get the current specification tree for the node
            json_tree = get_spec_tree(driver, node)
            
            # Run the main formalization pipeline
            result = formalize_specifications(properties, json_tree, llm, json_llm, openai_client)
            
            if result:
                # Save the result to a file
                output_path = os.path.join(OUTPUT_DIR, f"{node}.txt")
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(result)
                print(f"Saved polished specification to {output_path}")

                # Update the database
                print(f"Updating Neo4j for node '{node}'...")
                delete_spec_children(driver, node)
                insert_spec_tree(driver, result)
                print("Update complete.")
            else:
                print(f"Skipping database update for '{node}' due to an error in formalization.")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if driver:
            driver.close()
            print("Neo4j connection closed.")

if __name__ == "__main__":
    main()

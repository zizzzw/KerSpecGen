# KerSpecGen

KerSpecGen is an automatic formalization project for kernel specifications, consisting of three main steps:

## 1. PropertyExtract

In PropertyExtract, the seL4 specification file (.thy) is split, and the Qwen/llama3.1-405B large model interface is used to extract the kernel specification properties corresponding to each specification code, forming a "code-Property" pair dataset.


## 2. KGConstruct

In KGConstruct, relevant documentation of seL4 is collected, and RAG technology and large models are used to recursively split the main functions of the kernel until it is split to the property level. Eventually, a knowledge graph of seL4 is formed and stored in neo4j.


## 3. SpecGen

In SpecGen, the "specification - property" pairs are divided. One part is used for fine-tuning the code generation model (using Llama-Factory as the fine-tuning tool), and the other part is used as the test set. Through GraphRAG technology, the previously constructed knowledge graph is utilized to help fine-tune the model and generate a complete specification file.


# Deployment 
## Install and run Ollama
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen:72b
ollama serve
```
## Install and run neo4j
```bash
cd ~
wget https://dist.neo4j.org/neo4j-community-5.20.0-unix.tar.gz
tar -xvzf neo4j-community-5.20.0-unix.tar.gz
cd neo4j-community-5.20.0/bin
./neo4j start
```
## Installation dependencies
```bash
conda create -n OSAF python=3.9
conda activate OSAF
pip install -r requirements.txt
```

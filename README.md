# KerSpecGen

KerSpecGen是一个内核规约的自动形式化项目，包含三个主要步骤：

## 1. PropertyExtract

在PropertyExtract中，对seL4的规约文件（.thy）进行拆分，并使用Qwen/llama3.1-405B大模型接口来抽取每一条规约代码对应的内核规约性质，形成“code-property”对数据集。


## 2. KGConstruct

在KGConstruct中，收集seL4的相关文档，并利用RAG技术和大模型对内核的主要功能进行递归拆分，直到拆分到性质层面，最终形成seL4的知识图谱并存储到neo4j中。


## 3. SpecGen

在SpecGen中，对“规约-性质”对进行划分，将一部分用于微调代码生成模型（使用Llama-Factory作为微调工具），另一部分作为测试集。通过GraphRAG技术，利用之前构建的知识图谱，帮助微调模型生成完整的规约文件。


# 部署
## 安装并运行Ollama
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen:72b
ollama serve
```
## 安装并运行neo4j
```bash
cd ~
wget https://dist.neo4j.org/neo4j-community-5.20.0-unix.tar.gz
tar -xvzf neo4j-community-5.20.0-unix.tar.gz
cd neo4j-community-5.20.0/bin
./neo4j start
```
## 安装依赖
```bash
conda create -n OSAF python=3.9
conda activate OSAF
pip install -r requirements.txt
```

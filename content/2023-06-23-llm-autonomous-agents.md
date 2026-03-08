---
title: "LLM Powered Autonomous Agents"
date: 2023-06-23
author: Lilian Weng
tags:
  - nlp
  - language-model
  - agent
  - steerability
  - prompting
summary: "An overview of building autonomous agents with LLM as the core controller, covering planning, memory, and tool use components."
---

Building agents with LLM (large language model) as its core controller is a cool concept. Several proof-of-concept demos, such as AutoGPT, GPT-Engineer, and BabyAGI, serve as inspiring examples. The potentiality of LLM extends beyond generating well-written copies, stories, essays, and programs; it can be framed as a powerful general problem solver.

## Agent System Overview

In a LLM-powered autonomous agent system, LLM functions as the agent's brain, complemented by several key components:

- **Planning**
  - Subgoal and decomposition: The agent breaks down large tasks into smaller, manageable subgoals, enabling efficient handling of complex tasks.
  - Reflection and refinement: The agent can do self-criticism and self-reflection over past actions, learn from mistakes and refine them for future steps.
- **Memory**
  - Short-term memory: All the in-context learning as utilizing short-term memory of the model to learn.
  - Long-term memory: This provides the agent with the capability to retain and recall information over extended periods, often by leveraging an external vector store and fast retrieval.
- **Tool use**
  - The agent learns to call external APIs for extra information that is missing from the model weights, including current information, code execution capability, and access to proprietary information sources.

## Component One: Planning

A complicated task usually involves many steps. An agent needs to know what they are and plan ahead.

### Task Decomposition

**Chain of thought** (CoT; Wei et al. 2022) has become a standard prompting technique for enhancing model performance on complex tasks. The model is instructed to "think step by step" to utilize more test-time computation to decompose hard tasks into smaller and simpler steps.

**Tree of Thoughts** (Yao et al. 2023) extends CoT by exploring multiple reasoning possibilities at each step. It first decomposes the problem into multiple thought steps and generates multiple thoughts per step, creating a tree structure. The search process can be BFS or DFS with each state evaluated by a classifier or majority vote.

Task decomposition can be done:
1. By LLM with simple prompting like `"Steps for XYZ.\n1."`
2. By using task-specific instructions
3. With human inputs

### Self-Reflection

Self-reflection is a vital aspect that allows autonomous agents to improve iteratively by refining past action decisions and correcting previous mistakes.

**ReAct** (Yao et al. 2023) integrates reasoning and acting within LLM by extending the action space to be a combination of task-specific discrete actions and the language space. The prompt template incorporates explicit steps for LLM to think:

```
Thought: ...
Action: ...
Observation: ...
... (Repeated many times)
```

**Reflexion** (Shinn & Labash 2023) is a framework to equip agents with dynamic memory and self-reflection capabilities. After each action $a_t$, the agent computes a heuristic $h_t$ and optionally may decide to reset the environment to start a new trial depending on the self-reflection results.

## Component Two: Memory

### Types of Memory

Memory can be defined as the processes used to acquire, store, retain, and later retrieve information. There are several types of memory in human brains:

1. **Sensory Memory**: The earliest stage of memory, providing the ability to retain impressions of sensory information after the original stimuli have ended.

2. **Short-Term Memory** (STM) or **Working Memory**: It stores information that we are currently aware of and needed to carry out complex cognitive tasks. Short-term memory has the capacity of about 7 items (Miller 1956) and lasts for 20-30 seconds.

3. **Long-Term Memory** (LTM): Can store information for a remarkably long time, ranging from a few days to decades, with essentially unlimited storage capacity.

We can roughly consider the following mappings:

- Sensory memory as learning embedding representations for raw inputs;
- Short-term memory as in-context learning, restricted by the finite context window length;
- Long-term memory as the external vector store accessible via fast retrieval.

### Maximum Inner Product Search (MIPS)

The external memory can alleviate the restriction of finite attention span. A standard practice is to save the embedding representation into a vector store that supports fast maximum inner-product search (MIPS).

Common choices of ANN algorithms for fast MIPS:

- **LSH** (Locality-Sensitive Hashing): Introduces a hashing function such that similar input items are mapped to the same buckets with high probability.
- **ANNOY** (Approximate Nearest Neighbors Oh Yeah): The core data structure are random projection trees.
- **HNSW** (Hierarchical Navigable Small World): Builds hierarchical layers of small-world graphs where bottom layers contain actual data points.
- **FAISS** (Facebook AI Similarity Search): Applies vector quantization by partitioning the vector space into clusters.
- **ScaNN** (Scalable Nearest Neighbors): Main innovation is anisotropic vector quantization.

## Component Three: Tool Use

Tool use is a remarkable and distinguishing characteristic of human beings. Equipping LLMs with external tools can significantly extend the model capabilities.

**MRKL** (Karpas et al. 2022), short for "Modular Reasoning, Knowledge and Language", is a neuro-symbolic architecture for autonomous agents containing a collection of "expert" modules where the LLM works as a router.

Both **TALM** (Tool Augmented Language Models; Parisi et al. 2022) and **Toolformer** (Schick et al. 2023) fine-tune a LM to learn to use external tool APIs.

**HuggingGPT** (Shen et al. 2023) is a framework to use ChatGPT as the task planner to select models available in HuggingFace platform. The system comprises 4 stages:

1. **Task planning**: LLM parses user requests into multiple tasks
2. **Model selection**: LLM distributes tasks to expert models
3. **Task execution**: Expert models execute on specific tasks
4. **Response generation**: LLM receives execution results and provides summarized results

## Challenges

After going through key ideas and demos of building LLM-centered agents, several common limitations emerge:

- **Finite context length**: The restricted context capacity limits the inclusion of historical information, detailed instructions, API call context, and responses.
- **Challenges in long-term planning and task decomposition**: Planning over a lengthy history and effectively exploring the solution space remain challenging.
- **Reliability of natural language interface**: The reliability of model outputs is questionable, as LLMs may make formatting errors and occasionally exhibit rebellious behavior.

## Citation

Cited as:

> Weng, Lilian. (Jun 2023). "LLM-powered Autonomous Agents". Lil'Log. https://lilianweng.github.io/posts/2023-06-23-agent/.

## References

[1] Wei et al. "Chain of thought prompting elicits reasoning in large language models." NeurIPS 2022

[2] Yao et al. "Tree of Thoughts: Deliberate Problem Solving with Large Language Models." arXiv preprint arXiv:2305.10601 (2023).

[3] Yao et al. "ReAct: Synergizing reasoning and acting in language models." ICLR 2023.

[4] Shinn & Labash. "Reflexion: an autonomous agent with dynamic memory and self-reflection" arXiv preprint arXiv:2303.11366 (2023).

[5] Karpas et al. "MRKL Systems: A modular, neuro-symbolic architecture that combines large language models, external knowledge sources and discrete reasoning." arXiv preprint arXiv:2205.00445 (2022).

[6] Schick et al. "Toolformer: Language Models Can Teach Themselves to Use Tools." arXiv preprint arXiv:2302.04761 (2023).

[7] Shen et al. "HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in HuggingFace" arXiv preprint arXiv:2303.17580 (2023).

[8] Park, et al. "Generative Agents: Interactive Simulacra of Human Behavior." arXiv preprint arXiv:2304.03442 (2023).

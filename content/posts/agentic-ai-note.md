---
title: Agentic AI Lecture Note
tags:
  - Agentic AI
  - Large Language Model
  - Software Engineering
categories: Work Note
abbrlink: 4e596c51
copyright: true
date: 2025-11-02 19:22:56
mathjax:
---

最近花了一些时间学习DeepLearning.AI的[Agentic AI](https://learn.deeplearning.ai/courses/agentic-ai)，并且拿到了[证书](https://learn.deeplearning.ai/certificates/e6e86963-d660-41ef-b97f-c9c8c398eb3f)。一共花了六个小时左右。内容偏向于介绍Agentic AI的一些基本概念和应用，课程质量很高，但是由于是入门课程，所以内容比较浅显，没有涉及到太多的数学和代码实现。

回想起来，上一次学 Andrew Ng 的课程还是在 2016 或 2017 年，在 Coursera 上学他的 Machine Learning。已经将近九年前的事了，时间飞逝啊。

接下来我将总结这门课的核心内容，并回答一个问题：为什么截至 2025 年 11 月 2 日，Claude Code 是最成功的 Agentic 类产品？

注意到这门课使用的是[AI Suite](https://github.com/andrewyng/aisuite)来做具体的实现，这与现在大家使用较多的 LangGraph、OpenAI Agent SDK 等略有区别，所以笔记里不会涉及很多的代码实现细节。

<!-- more -->

# Introduction to Agentic Workflows
这门课里讲的Agentic AI，实际上讲的是Agentic AI workflows，定义为An agentic AI workflow is a process where an LLM-based app
executes multiple steps to complete a task. 相比于Non-agentic workflow只需一次调用，不涉及流程。而Agentic workflow则是“有流程、有记忆、有反馈”的智能体系统，能适应更复杂的场景。

比如以写文章为例子，Agentic AI workflow生成文章的流程包括自动分解、规划、执行多步任务，调用其他工具等等，而不是只做一次性输出。
![](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/202511031525768.png)

Agentic AI workflow可以是高度自主的，比如在执行过程中自己来思考要把任务分为哪几步，调用什么工具，甚至是自行编写代码来完成任务，也可以是低自主的，预先规定好步骤和要调用的工具的，只有在类似于文本和图片生成的步骤才有一定的自主性。

![](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/202511031531702.png)

Agentic AI 的优势主要有以下几点：

- 性能更好：在使用相同模型的情况下，由于能够使用工具，能够获得更好的性能表现，例如在写代码的时候可以使用代码检查工具和运行单元测试来提高代码质量，而不只是单纯的代码补全。
- 并行执行：例如同时执行多个搜索工作。
- 模块化：可以灵活替换工作流中的某一个步骤使用的工具或者模型。

设计 Agentic AI 最关键的一步是对任务进行拆解，识别哪一步可以由 Agentic AI 来完成。目前主要有两类：

1. AI 模型：例如文本生成、信息提取与总结、PDF 转文本、语音生成、图像分析与生成。
2. 工具使用：例如网页搜索、数据库查询、RAG、计算与数据分析。

Agentic AI 的评估是开发过程中最重要的一部分，可以分为客观评价和主观评价。
- 客观评价：例如通过代码检查与单元测试；再例如在搜索过程中，是否使用了重要的信息源（若是研究报告，好的报告一定会引用知名期刊）。
- 主观评价：常见做法是 LLM-as-a-Judge，但并非最佳实践，后续会详细展开。

同时要通过检查 trace 来进行错误分析与评估。


常用的设计模式有这几种

1. Reflection
2. Tool use
3. Planning
4. Multi-agent collaboration 

这里面最重要的就是Reflection和 Tool use，接下来会详细展开。

# Reflection Design Pattern

Reflection(反思)是指通过固定步骤对LLM的初次输出再次进行思考和分析的过程，拿人类写代码举例子，在写完代码之后运行单测可以得到代码的结果，然后根据这个代码可以去迭代这个结果。

这个在代码里面来做并没有很难，这是课程上的代码例子
```python
def generate_draft(topic: str, model: str = "openai:gpt-4o") -> str: 
    
    # Define your prompt here. A multi-line f-string is typically used for this.
    prompt = f"""
    You are an expert academic writer. Write a well-structured essay draft on the following topic:

    Topic: "{topic}"

    The essay should include:
    - A clear introduction with a thesis statement.
    - 2–3 body paragraphs that elaborate on the main ideas.
    - A strong conclusion summarizing the main points.

    The tone should be formal and academic, and the essay should be coherent, logically organized, and written in full sentences.
    """ 

    
    # Get a response from the LLM by creating a chat with the client.
    response = CLIENT.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=1.0,
    )

    return response.choices[0].message.content

def reflect_on_draft(draft: str, model: str = "openai:o4-mini") -> str:

    # Define your prompt here. A multi-line f-string is typically used for this.
    prompt = f"""
    You are an expert writing instructor. Please carefully review the following essay draft:

    ---
    {draft}
    ---

    Provide a detailed reflection on the essay, including:
    1. The strengths of the essay (e.g., clarity, structure, argumentation, tone).
    2. The weaknesses or areas for improvement (e.g., coherence, evidence, flow, grammar).
    3. Specific and actionable suggestions to improve the essay in the next revision.

    Be concise but insightful, and focus on helping the author improve their work.
    """

    # Get a response from the LLM by creating a chat with the client.
    response = CLIENT.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=1.0,
    )

    return response.choices[0].message.content

def revise_draft(original_draft: str, reflection: str, model: str = "openai:gpt-4o") -> str:

    # Define your prompt here. A multi-line f-string is typically used for this.
    prompt = f"""
    You are an expert editor. Revise the following essay using the feedback provided.
    
    Goals:
    - Address every actionable point in the feedback.
    - Improve clarity, coherence, argument strength, transitions, and overall flow.
    - Preserve the author’s intent and key ideas; do not change the topic.
    - Keep length roughly similar (±15%) unless feedback suggests otherwise.
    - Do not invent facts or citations. If evidence is requested but unavailable, strengthen reasoning and clarify limits.
    
    Output requirement:
    - Return ONLY the final revised essay. No preface, bullets, or meta commentary.

    --- ORIGINAL DRAFT ---
    {original_draft}
    --- END ORIGINAL DRAFT ---

    --- REFLECTION FEEDBACK ---
    {reflection}
    --- END REFLECTION FEEDBACK ---
    """

    # Get a response from the LLM by creating a chat with the client.
    response = CLIENT.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=1.0,
    )

    return response.choices[0].message.content
essay_prompt = "Should social media platforms be regulated by the government?"

# Agent 1 – Draft
draft = generate_draft(essay_prompt)
print("📝 Draft:\n")
print(draft)

# Agent 2 – Reflection
feedback = reflect_on_draft(draft)
print("\n🧠 Feedback:\n")
print(feedback)

# Agent 3 – Revision
revised = revise_draft(draft, feedback)
print("\n✍️ Revised:\n")
print(revised)  
```
流程上，就是添加一个新的 agent 做 reflection，提供 feedback，然后再写一次。
Reflection 在大多数任务上都能提升输出质量。对于 reflection 带来的效果评估，这里给了两个例子：

## Evaluating the impact of reflection
### 主观评估（图表生成）
![](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/202511032109283.png)
最简单的方案就是之前提到的 LLM-as-a-Judge：使用一个模型生成画图代码之后，用另一个模型来评价生成的图。但这并不准确：一方面有点像“garbage in, garbage out”，拿模型来评价模型；另一方面有研究表明模型会更偏好第一个选项（position bias）。因此可以在 LLM-as-a-Judge 场景下加入 rubric-based grading（基于评分量表的评估）。例如在评价生成的图表时，提示词可以这样写：

```
Assess the attached image against this
quality rubric. Each item should receive a
score for 1 (true) or 0 (false). Return the scores
for each item as a json object
1. Has clear title
2. Axis labels present
3. Appropriate chart type
4. Axes use appropriate numerical range
5. …

```
### 客观评估（SQL 查询生成）
![](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/202511032114633.png) 
在这种情况下我们就可以根据真实数据来生成评价数据集了。因此可以Build a dataset of ground truth examples来作为评价。

除此之外，能够用代码来评价的通常更容易。这就是接下来要讲的 Reflection with external feedback（结合外部反馈的反思）。


#### Reflection with external feedback

这一步通常在加入 reflection 之后做，它的效果曲线大致如下：

![](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/202511032134411.png)
拿 SQL 生成来举例子，大概的流程代码长这样：

```python
def run_sql_workflow(
    db_path: str,
    question: str,
    model_generation: str = "openai:gpt-4.1",
    model_evaluation: str = "openai:gpt-4.1",
):
    """
    End-to-end workflow to generate, execute, evaluate, and refine SQL queries.

    Steps:
      1) Extract database schema
      2) Generate SQL (V1)
      3) Execute V1 → show output
      4) Reflect on V1 with execution feedback → propose refined SQL (V2)
      5) Execute V2 → show final answer
    """

    # 1) Schema
    schema = utils.get_schema(db_path)
    utils.print_html(
        schema,
        title="📘 Step 1 — Extract Database Schema"
    )

    # 2) Generate SQL (V1)
    sql_v1 = generate_sql(question, schema, model_generation)
    utils.print_html(
        sql_v1,
        title="🧠 Step 2 — Generate SQL (V1)"
    )

    # 3) Execute V1
    df_v1 = utils.execute_sql(sql_v1, db_path)
    utils.print_html(
        df_v1,
        title="🧪 Step 3 — Execute V1 (SQL Output)"
    )

    # 4) Reflect on V1 with execution feedback → refine to V2
    feedback, sql_v2 = refine_sql_external_feedback(
        question=question,
        sql_query=sql_v1,
        df_feedback=df_v1,          # external feedback: real output of V1
        schema=schema,
        model=model_evaluation,
    )
    utils.print_html(
        feedback,
        title="🧭 Step 4 — Reflect on V1 (Feedback)"
    )
    utils.print_html(
        sql_v2,
        title="🔁 Step 4 — Refined SQL (V2)"
    )

    # 5) Execute V2
    df_v2 = utils.execute_sql(sql_v2, db_path)
    utils.print_html(
        df_v2,
        title="✅ Step 5 — Execute V2 (Final Answer)"
    )

```

# Tool Use

这一部分是我认为今年 Agentic AI 爆发的原因之一。MCP 的出现极大地降低了为模型适配工具的难度。

## What are tools

LLM 本质上只是一个文本生成模型，它并不具备直接调用工具的能力；其“调用工具”的能力完全来自于它能够输出一段可被执行环境识别的代码或指令。

比如如果问LLM现在是几点，在学习这门课之前我想象中的流程一直是

LLM收到请求->启动一个新的线程来调用`datetime.now().strftime("%H:%M:%S")`得到结果->主线程收到结果返回给我

实际上是

后端收到请求并转发给 LLM → LLM 输出可执行的代码或标记（例如 `""" FUNCTION def get_current_time(): """`）→ 后端检测到结果里带有 `FUNCTION`，将生成的内容传递给对应工具执行 → 后端得到工具的返回结果并发回 LLM → LLM 生成新的最终结果返回给我。

![](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/202511032147035.png)

这也就解释了为啥工具调用能力是最近一年很多新的模型的训练方向，本质上是模型你能够按照你给的prompt的标准给出对应的格式的回答的能力。

例如
```markdown
You have access to a tool called
get_current_time for a specific timezone.
To use it, return the following exactly:
FUNCTION:
get_current_time("timezone")
```
模型的工具调用能力好坏完全取决于它能否精确地输出 `FUNCTION:get_current_time("timezone")` 这一行。也就是说，如果想让一个 Agentic AI 有工具调用能力，模型必须具备以下两个方面的能力：

- Agentic AI 知道自己能够调用哪些工具：这与 Agentic AI 的提示词（prompt）相关，必须在发送请求时明确告诉它有哪些工具可用。这也解释了我之前的两个疑惑：其一，为什么 MCP 会占用 Claude Code 的 context window——因为这些工具说明必须包含在请求的上下文里；其二，为什么同样的模型在不同产品中的工具调用能力不同——为了降低成本，厂商最直接的办法是缩小 context，一些工具相关的上下文可能会被丢弃。也因此，当我们明确告诉它使用某个工具时，它就更可能去调用，因为我们把该工具的描述显式放进了上下文。
- 依样画葫芦的能力。这是模型本身的能力。这也与我的体感一致：去年很多模型（尤其是 GPT）还无法完全按照提供的最新文档与编码规范来写代码，容易“胡编乱造”；而今年经过特调的模型在这方面有明显提升。这也解释了为什么 Claude 系列在写代码方面体验更好，这与模型的“智力”未必相关，而更多与其“听话程度”（遵循格式与指令的能力）相关。
  

总结一下，要让模型知道他有这个工具，就必须得在prompt里面给他写出来。这就会有一些令人头痛的问题

## Tool syntax

就拿得到当前的时间来说，他实际的调用流程长这样
```python
def get_current_time():
    """Returns the current time as a string"""
    return datetime.now().strftime("%H:%M:%S")

client = ai.Client()
response = client.chat.completions.create(
    model="openai:gpt-4o",
    messages=messages,
    tools=[get_current_time],
    max_turns=5,
)
```
假设我的工具还有参数，那么调用就会变成

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_current_time",
        "description": "Returns current time for the given timezone.",
        "parameters": {
            "timezone": {
                "type": "string",
                "description": "The IANA time zone string, e.g., 'America/New_York' or 'Pacific/Auckland'."
            }
        }
    }
}]
```
如果参数越来越多、工具越来越多，那么这一段就会变得越来越长、越来越难以维护，因此大家用得比较多的有两个解决办法：

### Code execution
这是一个很有意思的办法，对于一些简单的工具，比如加减乘除，我们可以让LLM直接输出工具的代码，例如

![](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/202511032156263.png)

在做这个的时候要注意，这些代码一定要在 sandbox 里运行，例如独立的 Docker 容器，否则风险很大。

实际上，我的体感是去年的 ChatGPT 就在做这件事：当问题稍微复杂一些时，通过“思考”过程可以看到它是通过写 Python 代码来执行的。他们内部显然也非常认同这个方向。这也是 Codex 最早的思路——一个运行在云端的应用，生成代码并执行测试来产出 PR，整体都运行在一个 sandbox 里；后来才仿照 Claude Code 出了 CLI 版本。

去年很流行的“智商检测”问题：“strawberry 里面有几个 r？”现在也有很多模型是通过生成（或执行）代码来完成的。

### MCP

MCP 的概念在这里就不再多讲了，这里主要说它是怎么解决之前提到的问题的。

在没有 MCP 的情况下：

- 每个 App（如 Slack、GDrive、GitHub）都需要自己去接入多个 LLM 工具或代理。

- 假设有 m 个应用和 n 个工具，就需要 m × n 条独立的集成通道。这会导致大量重复开发和维护成本。

MCP（Model Context Protocol） 的核心思想是：所有 App 和 Tools 都通过一个 共享的 MCP Server 进行通信。
也就是说：
- 每个 App 只需要与 MCP Server 对接一次（m 条连接）；
- 每个 Tool 也只需要注册到 MCP Server 一次（n 条连接）；

```css
App1 ─┐
App2 ─┼─> Shared MCP Server <─┬─ Tool1 (Slack)
App3 ─┘                        ├─ Tool2 (GitHub)
                               └─ Tool3 (GDrive)
```
MCP Server 负责：
- 管理工具描述（JSON schema、metadata）；
- 接收来自不同 App 的请求；
- 统一调度和调用相应的 Tool；
- 将结果返回给对应的 App 或 Agent。

它的作用很像一个 Kong 这样的 API Gateway，或者 Backend for Frontend（BFF）。这显著减少了为 LLM 开发和适配工具所需的工作量。


# Practical Tips for Building Agentic AI

在开发Agentic AI的时候，不要陷入长时间的理论和架构讨论，最好的方法是

1. 快速构建 MVP（quick and dirty to start）
2. 基于结果构建评估，构建一个小评估数据集，哪怕只有20个，根据评价结果来找出容易出错的环节来改进
3. 持续改进评估系统，随着Agent的迭代来优化


## Evaluations (evals)

评价主要从两个维度来进行

- 评估方法：使用代码来评估（Objective） vs LLM AS A Judge(Subjective)
- 真实值可用性：有标准答案 vs 无标准答案

![](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/202511041254577.png)

1. 有真实值 + 代码评估：最客观可靠。例如判断发票日期是否与预期一致，或者用正则表达式匹配格式和关键信息。
2. 有真实值 + LLM 评估：一般适用于 Deep Research 或总结类型（例如 Notebook LLM）的任务。例如在生成研究综述时，好的研究报告就必须引用某些期刊与来源；在生成调研报告时，一般必须包含若干品牌；在生成学习笔记时，则必须包含几个关键字。
3. 无真实值+代码评估：通过简单的规则来进行基础校验，比如生成内容的长度。
4. 无真实值+LLM评估：非常主观灵活的评价标准，一般情况下不推荐作为best practice使用。


## Error Analysis and prioritizing next steps

在有了 eval 之后，就可以通过分析执行的轨迹（trace），对比每一步的输出与预期结果，计算每一步的错误率，从而集中优化错误最多、影响最大的环节。

![](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/202511041331799.png)

例如，在一个客服系统中可能会有这三步

![](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/202511041333477.png)

1. 生成 SQL 查询
2. 通过 SQL 去数据库查询数据
3. 将结果返回给顾客

这时候就可以列一个表

![](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/202511041334343.png)

运行一些例子，然后分析每一步的结果，来得出在哪一步出错最多。

## Component-level evaluations

除了端到端的整体评估，对单个组件进行评估同样很重要。这能让你更快速、精准地测试和优化特定模块，而无需从头到尾运行整个工作流。

![](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/202511041336067.png)

例如在一个research agent中，就可以单独对web research进行评估和优化。

## 优化组件的性能

一般来说，Agentic AI workflow 分为 LLM 组件和非 LLM 组件。

当 LLM 组件是瓶颈时，可以从以下方面入手：

- Improve your prompts: Add more explicit instructions; add one or more concrete examples to the prompt (few-shot prompting)
- Try a new model: Try multiple LLMs and use evals to pick the best
- Split up the step: Decompose the task into smaller steps 
- Fine-tune a model: Fine-tune on your internal data to improve performance

当非 LLM 组件（例如 Web search、RAG 的文本检索、代码执行、已训练的传统 ML 模型等）是瓶颈时，可以从以下方面入手：

- Tune hyperparameters of component: Web search — number of results, date range; RAG — change similarity threshold, chunk size; ML models — detection threshold
- Replace the component: Try a different web search engine, RAG provider, etc. 

作为开发者，培养对模型的直觉很重要，需要了解每种模型适合什么任务，以及如何在性能、延迟和成本之间取得平衡。
培养直觉的方法有：

- Play with models often
  -  Having a personal set of evals might be helpful
  -  Read other people’s prompts for ideas of how to best use models

- Use different models in your agentic workflows
  - Which models work for which types of tasks?
  - Use a framework/SDK or model provider that allows changing models easily.

## Latency, cost optimization

延迟和成本固然重要，但是在初期不应该过度考虑，通常的策略是先提升准确率，确保Agent正常工作，再针对性优化延迟和成本。

# Patterns for highly autonomous agents

之前四节讲的是少量或半自主性的 Agentic AI，最后也简要讲了高自主性的 Agentic AI 该如何设计。

## Planning workflows

在这种模式下，我们给 LLM 提供一系列可用的工具，并要求 LLM 规划出完成任务的工具调用步骤，让系统按照这个计划进行。比如在之前提到的客服系统中。

![](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/202511041400422.png)

就可以让 Planning Agent 生成 JSON 格式的执行计划

```json
{
  "plan": [
    {
      "step": 1,
      "description": "Find round sunglasses",
      "tool": "get_item_descriptions",
      "args": { "query": "round sunglasses" }
    },
    {
      "step": 2,
      "description": "Check available stock",
      "tool": "check_inventory",
      "args": { "items": "results from step 1" }
    },
    …
  ]
}
```
这样就方便我们后续把这个拆分作为每一步的Agent的输入。

但是这样也会有问题，比如

![](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/202511041424781.png)

我们提问：“Which month had the highest sales of hot chocolate?”（哪一个月的热巧克力销量最高？）

Planning Workflow 将会是这样的：
1. 用 filter_rows 工具筛选出一月的热巧克力数据。
2. 用 get_column_mean 求平均销量。
3. 再筛选二月，再求平均；三月、四月……重复直到十二月。
4. 最后比较这些平均值，找到最高的月份。

这样就会有以下几个问题：

1. Brittle（脆弱）
   
   这种方法对输入结构依赖太强。如果 CSV 文件列名从 coffee_name 改成 drink_name，或者日期格式稍有不同，整个流程就会出错。模型也可能忘记前几步的结果，比如「Step 3 results」引用不到，稍有变化就会崩溃。

2. Inefficient（低效）
   
   LLM 每执行一步都得重新生成下一步指令。要跑 12 个月，就得调用 12 次 filter_rows + 12 次 get_column_mean。每次调用之间还要等模型回应和上下文传递，速度很慢且算力浪费。

3. Continuously dealing with edge cases（不断处理边缘情况）
   
   每次数据有点不同就得写补丁，开发者永远在修“特殊情况”，而不是改进算法：
   - 有的月份没有热巧克力 → 代码报错，要加判断。
   - 数据缺失或格式不对 → 要加异常处理。
   - 有时列顺序不一样、文件名不同 → 又得改逻辑。

所以更好的方案是 Planning with code execution

## Planning with code execution

![](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/202511041427014.png)

直接让系统生成代码，而不是JSON格式的步骤，相对来说就更加灵活

## Multi-agentic workflows & Communication patterns for multi-agent systems

最后这里课程只是简单的讲了一下，当任务过于复杂时，可以由多个专门的智能体一起完成，常见的模式有

1. 串行智能体：像流水线一样工作，前一个 Agent 的输出是后一个 Agent 的输入。
2. 分层智能体：一个主管智能体充当 manager，负责分解与分配子任务，并进行结果汇总与整合。

![](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/202511041434016.png)

# 总结

这门课可以作为一个很好的新手入门大纲，帮助了解 Agent 开发。基本上只要稍微懂一些 Python 和 Jupyter Notebook，以及提示词工程，就可以理解并完成课程，不涉及太多代码。

最有用的部分就是Reflection和Tool Use。

接下来回答两个问题：

1. 为什么Claude Code是目前最成功的Agentic类产品？
  
- 产品只涉及代码生成，有明确且可量化的评价体系，例如测试和 Lint。这也是我认为为什么现在 Agentic Coding 产品这么多的最大原因：eval 是开发过程中最重要也最困难的部分，而“写代码”的评估相对其他场景（例如 Deep Research）更容易做成客观评价。
- MCP。MCP 极大地增强了 Claude Code 通过调用工具获取外部信息并获得外部反馈的能力。
- 模型“听话”，依样画葫芦的能力很强。
- Context and Prompt Management：Claude Code 的开发者显然意识到哪些内容对写代码最重要，因此在设计阶段就预先定义了固定的上下文。
![](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/202511041443224.png)

2. 下一个最成功的Agentic类产品会是什么？

我认为会是浏览器。ChatGPT Atlas 目前已经发布，我的体验是它做得很成功。

- 浏览器在执行操作时能够很好地避免很多安全问题，例如个人账号的登录，这些都可以由用户本人在本地完成与存储。这在此前各类产品的 Agent 模式（例如 Manus 和 ChatGPT）中是很大的痛点。之前它们的模式有点像“云原神”：在远端服务器上开启一个虚拟机，由你的语言指令让 AI 在虚拟机上操作浏览器与操作系统；这种情况下多窗口跳转等操作体验较差（亲身体验）。而多数用户真正需要的是让 AI 操作自己的浏览器和系统，由自己完成登录、付款等关键环节，把重复性操作交给 Agent。这有点像最初的 Codex 与 Codex CLI/Claude Code 的区别。
- 浏览器场景下的 Agentic 产品，评价体系相对清晰。例如自动化表单填写、信息抽取、网页导航等任务，都可以通过结果准确率、完成效率等客观指标进行评估，便于持续优化和迭代。
- MCP 等协议的普及，尤其是 Playwright MCP 极大降低了 Agent 与浏览器之间的集成门槛，让模型能够像调用 API 一样灵活地操作网页、抓取数据、完成复杂任务。只要模型能够精确地输出调用指令，就能实现高度自动化的浏览体验。
- 浏览器天然具备极强的工具属性，是信息获取、任务执行和外部世界交互的核心入口，并具备很强的可扩展性与生态潜力。一旦 Agentic 浏览器产品成熟，就会像互联网时代的 Google 一样，成为新的流量入口。

---
title: Agentic AI Lecture Notes
tags:
  - Agentic AI
  - Large Language Model
  - Software Engineering
categories: Work Note
abbrlink: 72cf705
copyright: true
date: 2025-11-02 20:22:56
mathjax:
---

I recently spent some time taking DeepLearning.AI’s [Agentic AI](https://learn.deeplearning.ai/courses/agentic-ai) course and earned the [certificate](https://learn.deeplearning.ai/certificates/e6e86963-d660-41ef-b97f-c9c8c398eb3f). It took around six hours. The content focuses on core concepts and applications of Agentic AI; the course quality is high, but as an intro it stays fairly light and doesn’t dive much into math or implementation details.

Thinking back, the last time I took an Andrew Ng course was in 2016 or 2017—his Machine Learning on Coursera. Nearly nine years ago; time flies.

Next I’ll summarize the core ideas from the course and answer one question: Why, as of Nov 2, 2025, is Claude Code the most successful agentic product?

Note that the course uses [AI Suite](https://github.com/andrewyng/aisuite) for hands-on implementation, which differs from commonly used stacks like LangGraph and OpenAI Agent SDK. So these notes won’t go deep into code implementation details.

<!-- more -->

# Introduction to Agentic Workflows
In this course, “Agentic AI” really means agentic AI workflows, defined as: An agentic AI workflow is a process where an LLM-based app executes multiple steps to complete a task. Compared with non-agentic workflows (a single call, no process), agentic workflows have process, memory, and feedback, and can adapt to more complex scenarios.

For example, in writing an article, an agentic workflow will automatically decompose and plan, execute multi-step tasks, call other tools, etc., instead of just producing a one-shot output.
![](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/202511031525768.png)

Agentic workflows can be highly autonomous—for instance, during execution they can decide how to break down the task, which tools to call, and even write code—or they can be low-autonomy, with predefined steps and tools, where only certain steps like text or image generation are somewhat autonomous.

![](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/202511031531702.png)

Key advantages of Agentic AI:

- Better performance: With the same model, tool use often improves performance. For coding, for example, you can run linters and unit tests to raise quality rather than pure completion.
- Parallel execution: e.g., run multiple searches at the same time.
- Modularity: you can flexibly swap the tool or model used for any step in the workflow.

The most critical step in designing an Agentic AI is decomposing the task and identifying which parts can be handled agentically. Two main categories:

1. AI models: text generation, information extraction and summarization, PDF-to-text, speech generation, image analysis and generation.
2. Tool use: web search, database queries, RAG, computation and data analysis.

Evaluation is the most important part of development and includes objective and subjective components.
- Objective: e.g., linting and unit tests; or in search tasks, whether important sources are used (good research reports cite reputable journals).
- Subjective: LLM-as-a-Judge is common, but not best practice—we’ll expand on this later.

Also inspect traces to analyze errors and evaluate behavior.


Common design patterns include:

1. Reflection
2. Tool use
3. Planning
4. Multi-agent collaboration 

The most important are Reflection and Tool use; details next.

# Reflection Design Pattern

Reflection means adding a fixed step to think about and analyze the LLM’s first-pass output. By analogy to coding: after writing code, you run tests, see results, and iterate based on the feedback.

This isn’t difficult to implement; here’s a course example:
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
Workflow-wise, you add a reflection agent that supplies feedback, then rewrite.
Reflection improves output quality for most tasks. Two examples of evaluating its impact:

## Evaluating the impact of reflection
### Subjective evaluation (chart generation)
![](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/202511032109283.png)
The simplest approach is LLM-as-a-Judge: after a model writes plotting code, use another model to grade the chart. This is not very accurate: on one hand it’s “garbage in, garbage out”—a model judging a model; on the other, research shows position bias for the first option. So in LLM-as-a-Judge we can add rubric-based grading. For example, when grading a chart, the prompt might be:

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
### Objective evaluation (SQL query generation)
![](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/202511032114633.png) 
Here you can build a dataset of ground-truth examples from real data and evaluate against that.

In general, what can be evaluated via code is easier. That leads to Reflection with external feedback.


#### Reflection with external feedback

This is typically added after basic reflection; its performance curve looks like this:

![](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/202511032134411.png)
Take SQL generation as an example; the flow might look like:

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

This is, in my view, one of the reasons agentic AI has taken off this year. MCP dramatically lowers the barrier to adapt tools for models.

## What are tools

An LLM is fundamentally a text generator. It doesn’t directly “call tools”; its ability to “use tools” comes entirely from outputting code or directives that an execution environment recognizes.

If you ask an LLM for the current time, before this course I imagined the flow as:

The LLM receives the request → spawns a new thread to call `datetime.now().strftime("%H:%M:%S")` → the main thread returns the result to me

In reality:

The backend receives the request and forwards it to the LLM → the LLM outputs executable code or markers (e.g., `""" FUNCTION def get_current_time(): """`) → the backend detects `FUNCTION` and passes the content to the corresponding tool → the backend returns the tool’s result to the LLM → the LLM generates the final answer for me.

![](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/202511032147035.png)

This explains why tool-use ability has become a training focus: it’s about producing outputs exactly in the specified format per the prompt.

For example:
```markdown
You have access to a tool called
get_current_time for a specific timezone.
To use it, return the following exactly:
FUNCTION:
get_current_time("timezone")
```
Whether a model can invoke tools well depends on whether it can precisely output the line `FUNCTION:get_current_time("timezone")`. So, if we want an agentic AI to use tools, the model needs two abilities:

- Knowing which tools are available: this depends on prompting; you must explicitly tell it which tools exist at request time. This also explains two observations: (1) why MCP consumes Claude Code’s context window—because tool descriptions must live in the request context; (2) why the same model has different tool-usage ability across products—vendors often shrink context to cut costs, dropping some tool descriptions. When we explicitly tell it to use a tool, it’s more likely to do so because the tool description is in context.
- The ability to follow formats exactly. This is a property of the model itself. It matches my impression: last year many models (especially GPT) struggled to fully follow the latest docs and style guides, often “making things up”; this year, instruction-following has improved markedly. That’s one reason Claude feels better for coding—not necessarily “smarter,” but more “obedient” to formats and instructions.
  

In short, to make the model aware of a tool, you must describe it in the prompt. This leads to some headaches.

## Tool syntax

Take “get current time” as an example; the actual call looks like:
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
If the tool takes parameters, it becomes:

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
As parameters and tools grow, this becomes long and hard to maintain. Two common mitigations:

### Code execution
This is interesting: for simple tools (e.g., arithmetic), we can have the LLM output the tool code directly, e.g.:

![](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/202511032156263.png)

Be careful to run such code in a sandbox (e.g., an isolated Docker container) to avoid risk.

In fact, I suspect last year’s ChatGPT did this: for slightly more complex questions, you can see in its “thinking” process that it writes Python code and runs it. Internally they clearly agree with this direction. This was also the original Codex approach—a cloud app that generates code and runs tests to produce PRs, entirely in a sandbox; later they released a CLI version similar to Claude Code.

Last year’s popular “IQ test” question—“How many r’s are there in ‘strawberry’?”—is often solved by generating (or executing) code.

### MCP

I won’t reintroduce MCP here; instead, how it solves the earlier problems.

Without MCP:

- Each app (Slack, GDrive, GitHub, etc.) integrates many LLM tools/agents itself.

- With m apps and n tools, you build m × n bespoke integrations. Lots of duplicated effort and maintenance.

With MCP (Model Context Protocol), all apps and tools communicate through a shared MCP server.
That is:

- Each app integrates with the MCP server once (m connections);
- Each tool registers with the MCP server once (n connections);

```css
App1 ─┐
App2 ─┼─> Shared MCP Server <─┬─ Tool1 (Slack)
App3 ─┘                        ├─ Tool2 (GitHub)
                               └─ Tool3 (GDrive)
```
The MCP server:
- Manages tool descriptions (JSON schema, metadata);
- Receives requests from different apps;
- Routes and invokes the appropriate tool;
- Returns results to the corresponding app or agent.

It’s similar to an API gateway like Kong, or a Backend for Frontend (BFF). This significantly reduces the work to develop and adapt tools for LLMs.


# Practical Tips for Building Agentic AI

When developing agentic systems, don’t get stuck in long theoretical or architectural debates. The best approach:

1. Build an MVP quickly (quick and dirty to start)
2. Build evaluations from results—create a small eval set, even just ~20; use results to find error-prone steps and improve them
3. Continuously improve evals as the agent iterates


## Evaluations (evals)

Two axes for evaluation:

- Evaluation method: code-based (Objective) vs. LLM-as-a-Judge (Subjective)
- Ground truth availability: with gold answers vs. without

![](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/202511041254577.png)

1. Ground truth + code-based evals: most objective and reliable. E.g., check invoice date matches the expected date, or use regex to validate formats and key fields.
2. Ground truth + LLM evals: common for deep research or summarization (e.g., notebook LLM). For research summaries, good reports must cite certain journals/sources; for market studies, they must cover certain brands; for learning notes, include key terms.
3. No ground truth + code-based evals: apply simple heuristics for basic checks, such as content length.
4. No ground truth + LLM evals: highly subjective and flexible; generally not recommended as best practice.


## Error Analysis and prioritizing next steps

With evals in place, analyze execution traces. Compare each step’s output to the expectation, compute error rates per step, and focus optimization on the highest-error, highest-impact parts.

![](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/202511041331799.png)

For example, a customer support system might have three steps:

![](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/202511041333477.png)

1. Generate an SQL query
2. Query the database via SQL
3. Return the result to the customer

You can then tabulate results like:

![](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/202511041334343.png)

Run some examples and analyze each step’s result to see where errors concentrate.

## Component-level evaluations

Besides end-to-end evaluation, component-level evaluation is important. It lets you test and optimize specific modules more quickly and precisely without running the entire workflow.

![](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/202511041336067.png)

For instance, inside a research agent, you can separately evaluate and optimize the web research component.

## Optimizing component performance

Broadly, an agentic workflow consists of LLM and non-LLM components.

When the LLM component is the bottleneck, try:

- Improve prompts: add explicit instructions; add concrete examples (few-shot)
- Try a new model: evaluate multiple LLMs and pick the best
- Split the step: decompose the task further
- Fine-tune a model: fine-tune on internal data to improve performance

When non-LLM components (web search, RAG retrieval, code execution, traditional ML models, etc.) are bottlenecks, try:

- Tune component hyperparameters: web search—result count, date range; RAG—similarity threshold, chunk size; ML—detection threshold
- Replace the component: try a different search engine, RAG provider, etc.

As developers, cultivate intuition for models—what tasks each model suits, and how to balance performance, latency, and cost.
Ways to build intuition:

- Play with models often
  - Keep a personal set of evals
  - Read others’ prompts for ideas on effective usage

- Use different models in your agentic workflows
  - Which models work for which tasks?
  - Use frameworks/SDKs or providers that make model swapping easy.

## Latency, cost optimization

Latency and cost matter, but early on you shouldn’t over-optimize them. Typical strategy: improve accuracy first to make the agent reliable, then optimize latency and cost.

# Patterns for highly autonomous agents

The previous four sections cover low- or semi-autonomous agentic systems. Finally, a brief look at designing highly autonomous ones.

## Planning workflows

In this mode, we give the LLM a set of available tools and ask it to plan a sequence of tool calls; the system then follows the plan. For example, in the customer support system mentioned earlier.

![](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/202511041400422.png)

The planning agent can produce a JSON execution plan:

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
This makes it easy to pass each step to the corresponding agent as input.

But there are issues, for example:

![](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/202511041424781.png)

We ask: “Which month had the highest sales of hot chocolate?”

The planning workflow might be:
1. Use `filter_rows` to filter January’s hot chocolate data.
2. Use `get_column_mean` to compute average sales.
3. Repeat for February, March, April … through December.
4. Compare the averages and find the maximum.

Problems:

1. Brittle
   
   This approach depends too much on input structure. If the CSV’s `coffee_name` column becomes `drink_name`, or the date format changes a bit, the process breaks. The model may also forget earlier results (e.g., “Step 3 results”), and small changes make it fail.

2. Inefficient
   
   The LLM must generate the next step after every step. For 12 months, that’s 12 `filter_rows` + 12 `get_column_mean` calls, with model round-trips and context passing in between—slow and compute-wasteful.

3. Continuously dealing with edge cases
   
   Every data quirk needs a patch, so you’re always fixing “special cases” instead of improving the algorithm:
   - Some months have no hot chocolate → errors; you need guards.
   - Missing or malformed data → more exception handling.
   - Different column orders or filenames → logic changes again.

So a better approach is Planning with code execution.

## Planning with code execution

![](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/202511041427014.png)

Have the system generate code directly instead of JSON steps—it’s more flexible.

## Multi-agentic workflows & Communication patterns for multi-agent systems

When tasks get complex, multiple specialized agents can collaborate. Common patterns:

1. Pipeline agents: like an assembly line; each agent’s output feeds the next.
2. Hierarchical agents: a manager agent decomposes and assigns sub-tasks, then aggregates results.

![](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/202511041434016.png)

# Summary

This course is a solid beginner’s outline for understanding agent development. With a bit of Python and Jupyter Notebook plus prompt engineering, you can follow and complete it; there’s not much heavy code.

The most useful parts are Reflection and Tool Use.

Two questions to close:

1. Why is Claude Code currently the most successful agentic product?
  
- The product centers on code generation with clear, quantifiable evaluation—tests and lint. That’s a big reason agentic coding products are thriving: evals are the most important and hardest part of development, and “coding” evals are comparatively objective.
- MCP. MCP greatly enhances Claude Code’s ability to call tools for external information and feedback.
- The model “follows instructions” very well.
- Context and prompt management: Claude Code’s developers clearly know what matters for coding and design a fixed context accordingly.
![](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/202511041443224.png)

2. What might be the next most successful agentic product?

I think it will be the browser. ChatGPT Atlas has launched, and in my experience it’s very well done.

- Browsers avoid many security issues during operation—for example, personal account login can be handled and stored locally by the user. This has been a major pain point in prior agent modes (e.g., Manus and ChatGPT). Those products felt like “cloud gaming”: a remote VM where AI operates the browser/OS via your language commands; multi-window navigation and similar actions feel clunky (from experience). What most users actually want is for AI to operate their own browser and system, with the user handling login/payment locally, and repetitive actions delegated to the agent. This is similar to the distinction between the original Codex and Codex CLI/Claude Code.
- In-browser agentic products have relatively clear evaluation frameworks. Tasks like automated form filling, information extraction, and web navigation can be evaluated via accuracy and efficiency, enabling continuous optimization.
- The spread of protocols like MCP—especially Playwright MCP—greatly reduces the integration cost between agents and browsers, letting models operate pages, extract data, and complete complex tasks as if calling APIs. As long as the model can output precise call directives, you get highly automated browsing.
- The browser is a powerful tool by nature: the core entry point for information access, task execution, and interaction with the external world, with strong extensibility and ecosystem potential. Once an agentic browser matures, it could become a new traffic gateway much like Google in the early web era.

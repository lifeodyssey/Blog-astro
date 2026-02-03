---
title: 速通Andrew Ng的Agentic AI课程：六小时的收获与思考
tags:
  - Agentic AI
  - Large Language Model
  - MCP
  - Tool Use
  - Reflection
categories: Work Note
copyright: true
abbrlink: 6774acb2
date: 2025-11-04 10:00:00
draft: true
---

TLDR

- **Agentic workflow本质**：有流程、有记忆、有反馈的LLM应用。最值得立刻做的是Reflection和Tool Use。
- **做不做得好关键在评估**：优先客观指标（测试/SQL执行/规则校验），主观打分用rubric收敛偏差，配合trace定位问题。
- **MCP的真正价值**：把"接工具"这件事标准化了，从m×n的集成地狱变成m+n的轻松对接。
- **为什么Claude Code能成功**：不是因为模型多聪明，而是因为代码场景的eval好做、模型够"听话"、MCP给力。


<!-- more -->
花了6个小时刷完DeepLearning.AI的[Agentic AI](https://learn.deeplearning.ai/courses/agentic-ai)课程。课程里的一些洞察确实解答了我之前的困惑：为什么MCP会占用context？为什么同样的模型在不同产品里表现差这么多？为什么代码生成类Agent最先成熟？

完整版笔记在[这里](https://lifeodyssey.github.io/posts/4e596c51.html)。

# Introduction to Agentic Workflows

课程里讲的Agentic AI，准确说是Agentic workflow——让LLM驱动的应用通过多步流程把事情做完。

拿写文章举例子，传统方式是你给个标题，模型一次性吐出全文；Agentic方式是自动拆成：列大纲→分段写→交叉检查→润色，每一步都能调用工具（搜索资料、检查语法、生成图表），步骤间保持记忆和上下文。

关键特征就三个：**有流程、有记忆、有反馈**。

从自主程度看，可以分两类：
- **高自主**：自己决定分几步、用什么工具、甚至写代码来解决问题
- **低自主**：步骤和工具都预设好，模型只在关键生成点发挥作用


Agentic workflow的优势在于：
- **性能更好**：同样的模型，加了工具就能做更多事。写代码时能跑测试和lint，不只是瞎补全。
- **可以并行**：多个搜索任务一起跑，不用串行等待。
- **模块化**：某一步的模型或工具随时能换，不用推倒重来。

设计Agentic AI最关键是任务拆解。哪些给模型做，哪些用工具：
- **模型擅长**：文本生成、信息提取总结、语音图像处理
- **工具擅长**：网页搜索、数据库查询、计算分析、外部API调用

# Reflection Design Pattern

Reflection说白了就是"让模型自己检查一遍再改"。但这个简单的idea效果出奇地好。

基本套路是三段式：

1. Draft（初稿）
先让模型把初稿写出来，不求完美，把主要内容和结构定下来。

2. Reflection（反思）
换个角度（可以是同一个模型的不同prompt，也可以是另一个模型）来审视初稿：
- 优点在哪（结构、论证、语气）
- 问题在哪（逻辑、证据、流畅度）
- 具体怎么改（可执行的建议，不要空话）

3. Revision（修订）
基于反馈修订，注意几个原则：
- 每条反馈都要处理
- 保持原意不跑偏
- 篇幅别差太多（±15%）
- 不能瞎编事实和引用

## Reflection with External Feedback

纯Reflection能提升，但加入外部反馈后效果更明显。我最喜欢的例子是SQL生成：

```python
# 简化的流程
def sql_with_reflection(question, db_path):
    # 1. 提取schema
    schema = get_schema(db_path)

    # 2. 生成SQL V1
    sql_v1 = generate_sql(question, schema)

    # 3. 执行V1，拿到实际输出
    result_v1 = execute_sql(sql_v1, db_path)

    # 4. 基于实际输出反思，生成V2
    # 关键：把真实执行结果作为反馈
    sql_v2 = refine_sql(question, sql_v1, result_v1, schema)

    # 5. 执行V2得到最终答案
    result_v2 = execute_sql(sql_v2, db_path)
    return result_v2
```

这种方式特别适合"能用代码验证"的场景——跑测试、执行查询、编译检查，都是天然的外部反馈源。

效果提升有个曲线：纯Reflection能涨一截，加外部反馈再涨一截，但会逐渐趋缓。所以别指望无限迭代能无限提升。

# Tool Use

这部分改变了我对LLM"调用工具"的理解。

## 工具调用的本质

之前我一直以为LLM调用工具是这样的：
> LLM收到请求 → 启动新线程调用函数 → 返回结果

实际上是这样的：
> 后端收到请求 → LLM输出特定格式文本（如`FUNCTION:get_time()`） → 后端识别这个标记 → 执行对应工具 → 把结果塞回LLM → LLM生成最终回复

**LLM本质上只会生成文本，"调用工具"完全是靠输出特定格式让外部环境去执行。**

这就解释了几个现象：

1. **为什么MCP会占用Claude Code的context？**
   因为工具描述必须放在prompt里，模型才知道有哪些工具可用。每个工具的名称、参数、描述都要占地方。

2. **为什么同样的模型在不同产品里工具调用能力不同？**
   省成本最直接的办法就是砍context。一些产品可能把部分工具描述给丢了。这也是为什么明确告诉它"用XX工具"时更容易成功——你把该工具的描述显式放进了上下文。

3. **为什么Claude系列写代码体验好？**
   不是因为它更"聪明"，而是它更"听话"——能精确按照你给的格式和规范输出。这是专门训练过的能力。

## 工具集成的痛点

假设你要让模型能获取当前时间，最简单的实现：

```python
def get_current_time():
    return datetime.now().strftime("%H:%M:%S")

# 调用时
response = client.chat.completions.create(
    model="gpt-4",
    messages=messages,
    tools=[get_current_time],
)
```

但如果工具带参数：

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_current_time",
        "description": "Returns current time for timezone",
        "parameters": {
            "timezone": {
                "type": "string",
                "description": "IANA timezone like 'America/New_York'"
            }
        }
    }
}]
```

参数越多、工具越多，这段JSON越来越长，维护起来头疼。所以出现了两个解决方案：

## 方案1：Code Execution

对简单工具，直接让LLM生成代码来执行。比如问"计算123.45 * 67.89"，与其定义乘法工具，不如让模型输出Python代码。

这方向很有意思：
- ChatGPT早就这么干了，稍微复杂的问题都能看到它在写Python
- 这两年很火的模型智商检测问题"strawberry有几个r"，很多模型就是生成`len([c for c in 'strawberry' if c=='r'])`来答
- OpenAI的Codex最早也是云端sandbox执行，后来才学Claude Code出了CLI版

注意：**代码必须在sandbox里跑**，不然风险极大。

## 方案2：MCP（Model Context Protocol）

MCP是Anthropic搞的标准化协议。问题场景是这样的：

**没有MCP时：**
- m个应用（Slack、GitHub、GDrive...）
- n个AI工具/模型
- 需要m×n个集成，每个都要单独开发维护

**有了MCP：**
```
应用端                    工具端
App1 ─┐                 ┌─ Slack API
App2 ─┼─> MCP Server <──┼─ GitHub API
App3 ─┘                 └─ GDrive API
```
- 每个App只需要接一次MCP（m个连接）
- 每个Tool也只接一次MCP（n个连接）
- 总共m+n个集成，大幅降低复杂度

MCP Server负责：
- 管理工具描述（JSON schema、元数据）
- 接收请求、路由到对应工具
- 统一鉴权、限流、监控

本质上就是个API Gateway，但专门为LLM场景优化。这也是为什么Claude Code能快速接入各种工具——标准化了，接入成本低。

# Practical Tips for Building Agentic AI

课程里给出的推荐是：

1. **快速搭个MVP**，quick and dirty is ok，先跑通流程
2. **准备小评测集**，20条也够，关键是能快速验证改动效果
3. **基于评测迭代**，每次改完跑一遍，看指标是涨是跌

## 评估体系

评估是Agentic AI的最重要的一步。一般有两个维度：

|  | 有真实值 | 无真实值 |
|---|---|---|
| **代码评估** | ✅ 最可靠<br>（日期匹配、格式校验） | 基础校验<br>（长度、关键词） |
| **LLM评估** | 适合总结类<br>（是否覆盖关键点） | ❌ 尽量避免<br>（太主观） |

课程里推荐的practice是：
- 能用代码评估的绝不用LLM
- LLM评估要用rubric（评分表）约束，逐项打分
- 无真实值的场景，想办法构造一些（哪怕是弱标签）

## 错误分析：看Trace找瓶颈

把每一步的输入输出都记下来，跑一批样本，统计各步错误率。

举个客服机器人的例子：
```
用户问题 → [生成SQL] → [查数据库] → [组织回复] → 最终答案
            20%错误      5%错误       8%错误
```

一目了然，SQL生成是瓶颈，优先优化这步（改prompt、加few-shot、或引入reflection）。

## 组件优化策略

**LLM组件卡住时：**
- 改prompt：更明确的指令，加few-shot示例
- 换模型：不同模型有不同强项
- 拆步骤：太复杂就分解成小步
- 微调：有数据的话，针对性训练

**非LLM组件卡住时：**
- 调参数：搜索条数、RAG阈值、chunk大小
- 换组件：试试不同的搜索引擎、向量库

**关于延迟和成本：**
先别管，把准确率做稳了再说。前期过早优化就是"用更快的方式做错事"。

## 培养对模型的直觉

几个小建议：
- 多动手试不同模型，维护自己的小评测集
- 多看别人的prompt，尤其是能稳定work的
- 用支持快速切换模型的框架，方便对比


# 核心洞察：为什么Claude Code是目前最成功的Agentic产品？

想明白这个，就理解了Agentic AI的现状：

1. **代码场景的评估好做**
   - 有测试、有lint、有编译，全是客观指标
   - 错了就是错了，对了就是对了，没有中间地带
   - 这是为什么Agentic Coding产品遍地开花——eval简单意味着迭代快

2. **MCP的加持**
   - 文件操作、终端命令、Git操作，全部标准化
   - 接入新工具成本低，生态容易起来

3. **模型够"听话"**
   - Claude不是最"聪明"的，但它最会按格式输出
   - 这对工具调用至关重要——差一个字符都可能调用失败

4. **Context管理精准**
   - Claude Code知道写代码最需要什么context
   - 文件树、当前文件、相关定义，都预先设计好了固定格式
   - 不浪费token在无关信息上


# 总结

这门课适合快速入门，不深但实用。最有价值的是Reflection和Tool Use的部分。

几个关键takeaway：
- 做Agent，评估比算法重要。没有eval就是瞎调。
- 外部反馈是提升质量的捷径（测试结果、执行输出都是好反馈）
- MCP不是银弹，但确实降低了工具集成的门槛
- 模型"听话"比"聪明"更重要（至少在工具调用场景）


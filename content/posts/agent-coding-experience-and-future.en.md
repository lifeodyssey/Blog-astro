---
title: Agent Coding Experience and Future
tags: Software Engineering
categories: Work Note
abbrlink: cb8288cd
copyright: true
lang: en
date: 2025-06-28 00:00:00
mathjax:
---

This article was written on June 28, 2025. Due to the rapid changes in the AI era, the facts and my attitudes described in this article may change dramatically even after just one day. Please refer to the latest situation.

<!-- more -->

## Core Concept Definitions

Before sharing my experience, let me define the core concepts of this article (my own definitions):

**Agent Coding** is software development using AI programming assistants (Coding Agents), where the Coding Agent completes tasks without active human intervention and prompting.

**Vibe Coding**: Completely generating code through dialogue without any manual modification or code review

**AI-assisted Coding**: Human-AI collaboration mode, combining AI generation with human review, which is currently the most practical approach

**Coding Agent**: Specific AI programming assistant tools, such as Cursor, Claude Code, Augment, etc.

This article mainly explores practical experience and future prospects under the Agent Coding methodology.

## Rapid Iteration Speed

When I [compared ChatGPT O1 and Claude 3.5 Sonnet](https://lifeodyssey.github.io/posts/aec625cb.html) last year, they could only help me write simple scripts, and at best handle some difficult algorithm problems. In summary, these tools could write good code at a scale of about 500 lines. In less than a year, these large language models have evolved to be able to write complete and even relatively complex business logic and products. Every day on social media, people with no programming experience use v0, lovable, and cursor to launch new products. From personal experience, the iteration speed of LLMs and related products has shortened from years (early 2023 to mid-2024) to quarters or even months now. For example, GitHub Copilot was initially the best product, Cursor was the absolute king of AI-assisted coding in March-April this year, with Roo Code, Cline, and Windsurf having their place. After Sonnet 4 came out, Claude Code took the lead directly, Cursor was caught up by Augment due to active dumbing down and limited call quotas, and GitHub Copilot has completely become a low-cost API call pool in recent months. In vibe coding products, initially only v0 existed, then Lovable gained a foothold with its better artistic style generation, and later Google launched AI Studio and Google Stitch...

The evolution speed of these tools gives me a feeling of being in a different world even after just one day. I've been surfing the internet intensively, actively using every new product released, and unsubscribing from products that don't work well after intensive use (for example, my chat bot has evolved from initially subscribing to ChatGPT, then to Claude, then to cheaper third-party APIs plus GUIs like [Cherry Studio](https://github.com/CherryHQ/cherry-studio) plus company's Gemini; my Coding Agent also switched from Cursor to Augment; I review my usage experience before renewal each month to unsubscribe from products), so the generational gap isn't obvious to me. But when I shared my experience with Cursor and Augment with a friend who was still using GitHub Copilot recently, he experienced the feeling of "the Qing Dynasty has fallen," "Beijing successfully bid for the Olympics," "Tianyi 3G is fast."

Besides these, the engineering practices for building AI agents have iterated several rounds, from initial embedding, to using grep for pure full-text search, to using databases for queries, to "no amount of optimization beats model companies updating their models." It feels like some experiences haven't had time to settle before being iterated away. Yesterday we were just learning prompt engineering, chain of thought, function calling, and today we're discussing LangGraph, LangChain, MCP.

## Core Practices of Agent Coding

Next, let me share some of my experiences and insights from practicing Agent Coding.

Let me first present this traditional diagram:

![Data, information, knowledge, insight, wisdom](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/data1.jpg)

As an internet surfing expert, I've been following ChatGPT's capabilities since it first came out. At that time, I evaluated it as roughly at the level of a college graduate. From that point on, GPT completely changed my process of learning new knowledge. The data and information parts no longer require me to read documents one by one or flip through books one by one. I can directly learn and think with questions, starting from knowledge.

Why mention this? Please look at the table below:

|              | LLM Knows       | LLM Doesn't Know       |
| ------------ | ------------- | --------------- |
| **You Know**   | Both you and LLM know | LLM assumes you know   |
| **You Don't Know** | You assume LLM knows | Neither LLM nor you know |

I found that whether I'm using LLMs or doing Agent Coding, our work area can be divided into these four quadrants:

- **Both you and LLM know**: This part of work was once one of the main reasons I thought LLMs had little impact in 2023 and the first half of 2024. For things both I and LLM know, I often felt LLM wrote poorly and inefficiently (requiring several conversations to correct errors). After Claude 3.5 and O1 were released in 2024, more precisely, after they could perform thinking, LLMs far exceeded my work efficiency in this part. I almost only need 2-3 conversations to start reviewing the code it writes, only needing to modify some minor parts before direct use.

- **You assume LLM knows**: This work was once my biggest frustration with LLMs. Classic examples include LLM's knowledge base often being outdated, lacking the latest APIs; LLM not understanding dependency relationships between classes and objects during work, nor knowing the sequence of tasks; and it not being clear about context in specific situations. But after a series of tools emerged, this part has been significantly improved and enhanced, with signs of being resolved. I'll discuss these tools later.

- **LLM assumes you know**: This is where I make the most errors in daily work. Classic scenarios include working with languages and libraries I'm not familiar with. In these places, you and LLM don't have equal information. My current only solution is to ask more questions, conduct line-by-line code review, understand the meaning of each line of code; additionally, generate tests, even open another AI to help you build test scenarios and test cases based on your requirements. I may not understand the details, but I need to ensure the input and output are exactly what I expect.

- **Neither LLM nor you know**: This is currently the only scenario where I write code without any LLM assistance. Currently, all code LLM writes for me in this area is 💩.

So when using LLMs or doing Agent Coding, I believe the most important core is context - ensuring the Coding Agent and you have the same context, that the Coding Agent and I are discussing the same thing; for parts I don't know, I need to ensure that in my use case, the input and output of generated code are exactly what I expect.

## Some Useful Tips

1. **Code review and refactor**: Currently, most of my collaboration with Coding Agents is with a pair coding attitude. Speaking of pair coding, this comes from years of experience at TW - identifying code smells, test-driven development, tasking. Consciously using these processes to collaborate with Coding Agents can greatly improve speed while ensuring quality. Additionally, I recently feel Coding Agents often make the mistake of writing a lot of redundant code - some written but not deleted after I rejected it, some original classes or methods not removed after being deprecated. These all need code review.

2. **Plan and Ask**: When getting any requirement, especially story-level requirements, never let the Coding Agent start writing code directly. Instead, use ask or plan mode to break down tasks, write them in md files, review its solution, then start a new conversation to let it begin writing. This can also greatly save tokens and reduce API costs.

3. **Let the Agent RTFM**: When solving high-level or open problems, you can let it search for the latest solutions. For example, if I need to introduce a library to solve my needs, I can let it search PyPI, npm, Maven, or GitHub for libraries with the most users/stars, and provide at least 5 solutions for deep research. Some processes I often do:
   - Paste a link to the docs and ask LLM to read it first
   - Ask LLM to find out the state-of-the-art
   - Use the Task tool and have LLM conduct deep-research on a particular topic

4. **Use MCP**: This series of tools can greatly improve the "you assume LLM knows" work scenario. Here are some I use daily:
   - [Context7](https://github.com/upstash/context7): This can solve part of the problem of AI not knowing the latest documentation
   - [Sequential Thinking](https://github.com/modelcontextprotocol/servers/tree/main/src/sequentialthinking): This can improve Coding Agent's logical thinking ability, letting AI know "what I need to do first and what to do next"
   - [Memory](https://github.com/modelcontextprotocol/servers/tree/main/src/memory): This allows Coding Agents to retain previous conversation context in each new conversation, reducing "I just said this, why don't you remember" situations. Cursor rules and Augment memory can achieve similar effects
   - [Shrimp task manager](https://github.com/cjo4m06/mcp-shrimp-task-manager): This allows Coding Agents to actively do tasking
   - [Feedback Enhanced](https://github.com/Minidoracat/mcp-feedback-enhanced): This can speed up feedback progress. Specifically, during Coding Agent collaborative programming, we can't follow (or keep up with its speed) the Coding Agent to see every file it modifies, so sometimes we find it has changed a bunch of things that are completely different from what you expected. Using this, it will stop and ask for your feedback halfway through your instructions, rather than after completing everything, allowing early termination
   - [Playwright](https://github.com/microsoft/playwright-mcp): This allows LLMs to interact with and screenshot web frontend pages, which should be helpful for frontend work. I haven't used it much yet
   
   Other useful MCPs can be found at [Smithery.ai](https://smithery.ai/). I'm currently only using a small portion.

5. **Precise Prompt**: This is still very important. The clearer and more explicit the prompt, the more accurate the generated code. Additionally, LLMs are often lazy. In such cases, telling it to use specific MCPs, telling the model to think, to deep think and ultrathink, can often yield better results. You can refer to [Extended thinking tips](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/extended-thinking-tips).

6. **Stop and interrupt early, do not attempt one shot**: These LLMs often have various problems in long conversations. Interrupt and adjust promptly. And one conversation usually cannot complete a larger requirement.

## Perhaps a Future Possibility: No Line-by-Line Code

After multiple attempts with Claude Code, I believe that in the future, people may really not need to write code line by line. CLI-mode Coding Agents can naturally perform parallel multitasking (for example, using git worktree and letting Claude Code use Claude Code), although costs keep rising (according to my estimates, if I use Claude Code in this mode for development every day, it might cost at least $500 per month, but can improve speed by more than double). This natural multitasking and parallel capability brought by CLI mode gives me the feeling that when doing Agent Coding, what I'm doing is no longer pair coding, but only high-level tasks like task breakdown and code acceptance. With good task breakdown, opening several git worktrees allows parallel execution of multiple tasks without interference.

After reading [How I Use Claude Code](http://spiess.dev/blog/how-i-use-claude-code), I've been practicing and trying a [project](https://github.com/lifeodyssey/req-to-code) that uses Gemini CLI to automatically generate code based on md files on GitHub Actions and submit PRs. [Here](https://github.com/lifeodyssey/req-to-code/pull/1/commits/7f5015e0b1138288720d37f959ad19a479276bcb) is a successful example, although it only did LeetCode 001 Two Sum (limited by my financial ability, I don't have that much money to buy APIs for complex projects).

The entire process is roughly:

`Requirement.md → GitHub Actions → Gemini CLI → Generated Code → Auto PR`

The most core step here only requires these few lines of code:

```yml

      - name: Generate Code with Gemini CLI
        id: generate
        env:
          GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
        run: |
          echo "🤖 Starting code generation with Gemini CLI..."

          # Read requirement content
          REQUIREMENT_CONTENT=$(cat "${{ steps.process.outputs.requirement_file }}")
          FEATURE_NAME="${{ steps.process.outputs.feature_name }}"

          echo "📝 Processing requirement: $FEATURE_NAME"
          echo "📄 Requirement file: ${{ steps.process.outputs.requirement_file }}"

          # Create output directory
          mkdir -p generated

          # Use Gemini CLI in non-interactive mode
          echo "🤖 Generating code with Gemini CLI..."

          # Create a more specific prompt for code generation
          PROMPT="You are a code generation assistant. Based on the following requirement, you must create actual JavaScript files using the write_file tool.

          REQUIREMENT:
          $REQUIREMENT_CONTENT

          INSTRUCTIONS:
          1. Use the write_file tool to create a JavaScript file named '${FEATURE_NAME}.js'
          2. Implement the solution with proper JSDoc comments
          3. Export the function for use in other modules
          4. Follow modern ES6+ syntax
          5. Include comprehensive error handling
          6. Add unit tests if appropriate

          You MUST use the write_file tool to create actual files. Do not just provide code examples."
          # ============================================
          # 🔥CORE STEP: Using stdin with enhanced prompt and YOLO mode
          # ============================================
          # All the above is preparation work, the magic happens in this single line:
          echo "$PROMPT" | gemini --yolo > "generated/${FEATURE_NAME}_output.txt" 2>&1
          # ☝️ That's it! One command to complete AI code generation

          # Check if generation was successful
          if [ $? -eq 0 ]; then
            echo "✅ Code generation completed successfully"
            echo "📄 Gemini CLI output:"
            head -20 "generated/${FEATURE_NAME}_output.txt"
          else
            echo "❌ Code generation failed"
            cat "generated/${FEATURE_NAME}_output.txt"
            exit 1
          fi

          # Store output info for next steps
          echo "output_file=generated/${FEATURE_NAME}_output.txt" >> $GITHUB_OUTPUT
          echo "feature_name=$FEATURE_NAME" >> $GITHUB_OUTPUT
```

The requirement.md here can be replaced with reading new story cards from Jira, GitHub Actions can be replaced with backend services to control conversation length and workflow (like the previously mentioned tips, git worktree and let Claude Code use Claude Code), after PR generation there can be new actions calling Gemini CLI to review code, and during code generation a backend dashboard can be added to allow real-time feedback and adjustments.

Under this workflow, I only need to write clear requirements and prompts, then I can sip tea while exploiting my agent to write code.

## Future

I still can't imagine how far Agent Coding will develop. Maybe it will be like the previously popular low-code platforms - making a big splash but only solving part of the problems. Maybe it will really bring about my imagined "I only need to write clear requirements and prompts." Maybe LLM development will hit bottlenecks causing it to only complete certain steps. But undeniably, the future process and methods of building software will be vastly different. At the beginning of this year, I saw the [Vibe Coding Manifesto](http://vibemanifesto.org/), excerpted below:

> 💜 Flow over friction – Ride the wave, don't fight it.

> 💜 Iteration over perfection – Perfection is obsolete if you can always reroll.

> 💜 Augmentation over automation – AI is a collaborator, not a replacement.

> 💜 Product thinking over code crafting – What matters is what you build, not how you write it.

> 💜 Rerolling over debugging – If fixing takes too long, regenerate.

> 💜 Human taste over technical constraints – The best tech serves great taste, not the other way around.

This reminds me of the Agile Code Manifesto from over twenty years ago. What I can do now is quickly learn how to use this new generation of spinning jenny.

Finally, thanks to the thoughtworks AIFSD for providing Claude Code, allowing me to experience the latest models and conduct experiments.

---

**Language versions:**
- [Chinese](https://lifeodyssey.github.io/posts/6d590b83.html)
- [Japanese](https://lifeodyssey.github.io/posts/ad479b93.html)

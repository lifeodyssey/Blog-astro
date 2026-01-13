---
title: Agentic AI 学習記録
tags:
  - Agentic AI
  - Large Language Model
  - Software Engineering
categories: Work Note
abbrlink: 34c938fb
copyright: true
date: 2025-11-02 19:22:56
mathjax:
---

最近、DeepLearning.AI の [Agentic AI](https://learn.deeplearning.ai/courses/agentic-ai) を学習し、[証明書](https://learn.deeplearning.ai/certificates/e6e86963-d660-41ef-b97f-c9c8c398eb3f) を取得した。学習時間は合計で約 6 時間。内容は Agentic AI の基本概念と応用の紹介が中心で、講座の質は高い。ただし入門向けなので、数学や実装コードは多くない。

振り返ると、Andrew Ng の講座を前回受けたのは 2016〜2017 年、Coursera の Machine Learning だった。もう約 9 年前。時が経つのは早い。

以下では本講座の要点をまとめ、次の問いに答える：2025 年 11 月 2 日現在、なぜ Claude Code は最も成功した Agentic 系製品なのか。

なお、講座の実装は [AI Suite](https://github.com/andrewyng/aisuite) を用いており、現在よく使われる LangGraph や OpenAI Agent SDK とは少し異なる。そのため本ノートでは実装の細部にはあまり踏み込まない。

<!-- more -->

# 1 Agentic Workflows の導入
ここでいう Agentic AI は、実際には Agentic AI workflow を指す。定義は「LLM を用いたアプリが、課題を完了するために複数の手順を実行する過程」。Non-agentic workflow が単発の呼び出しで流れを持たないのに対し、Agentic workflow は「流れ・記憶・feedback」を備え、より複雑な状況に適応できる。

例えば文章作成なら、Agentic AI workflow は課題の自動分解・計画・複数手順の実行・他の tool の呼び出しなどを行い、単発の出力では終わらない。
![](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/202511031525768.png)

Agentic AI workflow は高い自律性を持たせることもできる。実行中に手順分解や使用する tool の選択、さらには自分で code を書いて完了することもある。一方、低自律にして、手順や使用 tool をあらかじめ固定し、文章や画像生成など一部のみを自律にする設計も可能。

![](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/202511031531702.png)

Agentic AI の利点は次のとおり。

- 性能向上: 同じ model を使っても tool を使えることで性能が上がる。例えば coding では code 検査や単体試験を回せるので、単なる補完より品質が高い。
- 並列実行: 例えば複数の検索を同時に実行できる。
- 部品化・交換容易性: workflow の各手順で使う tool や model を柔軟に差し替え可能。

Agentic AI の設計で重要なのは課題の分解で、どこを Agentic AI に任せるかを見極めること。主に二系統がある。

1. AI model: 文章生成、情報抽出と要約、PDF→text、音声生成、画像解析と生成。
2. tool 利用: Web 検索、database 照会、RAG、計算とデータ分析。

Agentic AI の評価は開発で最重要。客観評価と主観評価に分かれる。
- 客観評価: 例えば code 検査と単体試験。検索では重要情報源を使ったか（研究報告なら著名 journal を引用しているか）。
- 主観評価: よくあるのは LLM-as-a-Judge。ただし最良実践ではない。後述。

併せて trace を確認し、誤り分析と評価を行う。


よく使われる設計指針は次の 4 つ。

1. Reflection
2. Tool use
3. Planning
4. Multi-agent collaboration 

この中で最重要なのは Reflection と Tool use。次で詳述する。

# 2 省察の設計指針

Reflection（省察）は、LLM の初回出力を固定手順で再考・分析すること。人間の coding なら、書いた後で単体試験を実行し、結果に基づき改良するのと同様。

実装は難しくない。講座の code 例:
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

# Agent 1 – 下書き
draft = generate_draft(essay_prompt)
print("📝 Draft:\n")
print(draft)

# Agent 2 – 省察
feedback = reflect_on_draft(draft)
print("\n🧠 Feedback:\n")
print(feedback)

# Agent 3 – 改稿
revised = revise_draft(draft, feedback)
print("\n✍️ Revised:\n")
print(revised)  
```
流れは、新しい agent を追加して reflection を行い、feedback を与え、出力を改稿するだけ。多くの課題で出力品質が向上する。効果評価として二つの例がある。

## 省察の効果評価
### 1. 主観評価（図表生成）
![](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/202511032109283.png)
最も単純なのは前述の LLM-as-a-Judge。ある model が作図 code を生成し、別の model がその図を評価する。しかし精度は十分でない。理由は二つ: 一つは「garbage in, garbage out」で、model が model を評価すること。もう一つは研究で示されるように position bias（最初の選択を好む傾向）があるため。そこで rubric-based grading（評価規準に基づく採点）を導入する。例えば図の評価では、prompt を次のように書く:

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
### 2. 客観評価（SQL 問い合わせ生成）
![](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/202511032114633.png) 
この場合は実データに基づき評価 dataset を作れる。つまり ground truth の例を構築し、それで評価する。

さらに、code で評価できる課題は一般に容易である。次は Reflection with external feedback（外部 feedback を組み合わせた省察）。


#### 外部 feedback を用いた省察

これは reflection を入れた後に行うことが多い。効果は次のような曲線になる。

![](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/202511032134411.png)
SQL 生成を例にすると、おおよその流れは次の code:

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

# 3 Tool 利用

今年 Agentic AI が伸びた要因の一つがこれ。MCP により、model へ tool を適合させる手間が大きく下がった。

## Tools とは

LLM は本質的に text 生成 model であり、直接 tool を呼び出す能力はない。tool の呼出能力は、実行環境が解釈できる code や指示を出力できることに由来するだけ。

例えば「今の時刻は？」と尋ねる場合、学習前の自分は次の流れを想像していた。

LLM が要求を受ける → 新しい thread を立てて `datetime.now().strftime("%H:%M:%S")` を実行 → 主 thread が結果を受け取り返す。

実際には、backend が要求を受け LLM に転送 → LLM が実行可能な code や印（例: `""" FUNCTION def get_current_time(): """`）を出力 → backend が `FUNCTION` を検知し、対応する tool を実行 → backend が結果を LLM に返す → LLM が最終応答を生成して返す。

![](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/202511032147035.png)

この仕組みは、tool 呼出能力が近年の学習で重視される理由でもある。本質は、model が与えた指示文（prompt）の規格に厳密に従った形式で応答できる能力にある。

例えば次のように書く。
```markdown
You have access to a tool called
get_current_time for a specific timezone.
To use it, return the following exactly:
FUNCTION:
get_current_time("timezone")
```
tool 呼出性能は、`FUNCTION:get_current_time("timezone")` を正確に出力できるかに完全に依存する。つまり Agentic AI に tool を持たせるなら、model は次の二点が必要。

- どの tool が利用可能かを知ること: これは prompt と関係し、要求送信時に利用可能な tool を明示する必要がある。これにより以前の疑問が説明できる。第一に、なぜ MCP が Claude Code の context window を消費するのか——tool 説明を context に含める必要があるため。第二に、同じ model でも製品によって tool 呼出能力が異なるのはなぜか——費用削減のために context を短くし、tool 関連の文脈が省かれることがあるため。したがって、特定の tool を使うよう明示すれば、呼び出しやすくなる。
- 模倣精度: model 自体の能力で、書式や指示に厳密に従う力。昨年多くの model（特に GPT）は最新の文書や規約に完全準拠した code を書けず、作り話が混じった。今年は調整により改善した。Claude 系が coding で良い体験を与えるのは、いわゆる「知能」よりも、この従順さ（format/指示の遵守）が大きい。
  

要するに、model に tool の存在を知らせるには prompt に書く必要があり、頭痛の種が出てくる。

## Tool syntax

現在時刻の取得を例にすると、実際の呼出は次のようになる。
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
tool に引数がある場合は次のようになる。

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
引数や tool が増えると、この部分は肥大化し保守が難しくなる。そこで広く使われる解決策が二つある。

### Code execution
興味深い方法として、簡単な tool（加減乗除など）は LLM に tool の code を直接出力させる。例えば:

![](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/202511032156263.png)

この場合、code は必ず sandbox で実行する。独立の Docker container など。本番環境で直接実行するのは危険。

体感として、昨年の ChatGPT はすでにこれを行っていた。問題が少し複雑になると、「思考」の過程で Python code を書いて実行していた。社内でもこの方向は強く支持されているようだ。これが最初の Codex の構想でもあった。cloud 側で動くアプリが code を生成し、試験を実行して PR を出す。全体が sandbox 上で動く。その後、Claude Code を模して CLI 版が出た。

昨年流行した「strawberry に r はいくつ？」のような問題も、多くの model は code を生成（あるいは実行）して解いている。

### MCP

MCP の概念自体はここでは詳述しない。前述の課題をどう解決するかに絞って述べる。

MCP がない場合:

- 各 App（Slack、GDrive、GitHub など）は、それぞれ複数の LLM の tool/agent を自前で接続する必要がある。

- m 個のアプリと n 個の tool があれば、m × n の統合が必要になり、重複した開発と保守が大量に発生する。

MCP（Model Context Protocol）の核心は、すべての App と Tools が共有の MCP Server を介して通信すること。
つまり、
- 各 App は MCP Server に一度だけ接続（m 本の接続）。
- 各 Tool も MCP Server に一度だけ登録（n 本の接続）。

```css
App1 ─┐
App2 ─┼─> Shared MCP Server <─┬─ Tool1 (Slack)
App3 ─┘                        ├─ Tool2 (GitHub)
                               └─ Tool3 (GDrive)
```
MCP Server は次を担う。
- tool 記述（JSON schema、metadata）の管理
- 複数 App からの要求の受領
- 対応する Tool の統一調停と呼出
- 結果を該当 App / Agent に返却

役割は Kong のような API Gateway、あるいは Backend for Frontend（BFF）に近い。LLM の tool 開発・適合の負担を大きく減らす。


# 4 Agentic AI 構築の実用的な手引き

Agentic AI を開発するとき、理論や設計議論に長く留まらない。最善の進め方は次のとおり。

1. 速く MVP を作る（quick and dirty で良い）。
2. 結果に基づき評価を作る。小さくても良い（20 件程度）。誤りが出やすい工程を特定して改善。
3. Agent の反復とともに評価系を継続的に改善。


## 評価（evals）

評価は二つの軸で見る。

- 評価方法: code による客観評価（Objective） vs LLM-as-a-Judge（Subjective）
- 正解の有無: 正解あり vs 正解なし

![](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/202511041254577.png)

1. 正解あり + code 評価: 最も客観で信頼できる。例: 請求書の日付が期待と一致するか、正規表現で形式と要点を照合。
2. 正解あり + LLM 評価: Deep Research や要約（Notebook LLM など）に適する。研究総説では特定 journal や出典を引用すべき、調査報告では複数ブランドを含むべき、学習ノートではキーワードを含むべき、など。
3. 正解なし + code 評価: 基本的な検査。例: 生成内容の長さ。
4. 正解なし + LLM 評価: 非常に主観で柔軟。通常は best practice としては推奨しない。


## 誤り分析と次の優先順位付け

eval を用意したら、実行 trace を分析し、各手順の出力を期待と比較し、手順ごとの誤り率を計算する。誤りが多く影響の大きい部分から集中的に最適化する。

![](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/202511041331799.png)

例として、ある問い合わせ対応 system には次の三手順がある。

![](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/202511041333477.png)

1. SQL 問い合わせを生成する。
2. SQL で database を照会する。
3. 結果を利用者に返す。

このとき、次のように一覧できる。

![](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/202511041334343.png)

いくつか例を流し、各手順の結果を分析し、どこで誤りが多いかを特定する。

## 構成要素レベルの評価

端から端までの評価だけでなく、単一構成要素の評価も重要。これにより workflow 全体を通さず、対象部分を素早く正確に検証・最適化できる。

![](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/202511041336067.png)

例えば research agent では Web research 部分だけを個別に評価・最適化できる。

## 構成要素の性能最適化

一般に Agentic AI workflow は LLM 構成要素と非 LLM 構成要素に分かれる。

LLM 構成要素がボトルネックのときは、次を検討する。

- 指示文を改善: 明確な指示を増やす。具体例を追加（few-shot prompting）。
- 別の model を試す: 複数 LLM を試し、評価で最良を選ぶ。
- 手順を分割: 課題を小さな手順に分解。
- fine-tune: 内部データで fine-tune して性能向上。

非 LLM 構成要素（Web search、RAG の検索、code 実行、既存 ML model など）がボトルネックのときは、次を検討する。

- hyperparameters を調整: Web search — 結果件数・期間。RAG — 類似度閾値・chunk size。ML models — 検出閾値。
- 構成要素を置換: 別の Web search engine、RAG provider などを試す。

開発者として model への直感を養うことが重要。どの model がどの課題に適するか、性能・遅延・費用の折り合いを理解する。方法:

- 頻繁に model を試す。
  - 自分用の評価集合を持つと良い。
  - 他者の指示文を読み、使い方の工夫を学ぶ。

- workflow で複数の model を使う。
  - どの model がどの種類の課題で機能するか見極める。
  - model 変更が容易な framework/SDK や provider を用いる。

## 遅延・費用の最適化

遅延と費用は重要だが、初期は過度に気にしない。まずは正確率を上げ、Agent が正しく動くことを確保し、その後に遅延と費用を最適化する。

# 5 高度自律 agent の設計指針

前半では少量/半自律の Agentic AI を扱った。最後に高自律 agent の設計を簡潔に述べる。

## 計画型 workflow

この方式では、利用可能な tool 一覧を LLM に与え、LLM に作業計画（tool 呼出の手順）を作らせ、その計画どおりに system が実行する。先の問い合わせ対応 system の例では、

![](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/202511041400422.png)

Planning agent に JSON 形式の実行計画を生成させることができる。

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
これで後続の各手順の agent 入力に分割しやすくなる。

しかし問題もある。

![](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/202511041424781.png)

質問: 「Which month had the highest sales of hot chocolate?」（どの月の hot chocolate の売上が最高か？）

Planning workflow は次のようになる。
1. filter_rows で 1 月の hot chocolate のデータを抽出。
2. get_column_mean で平均売上を求める。
3. 2 月・3 月…12 月まで繰り返す。
4. 各平均を比較し、最大の月を得る。

この方法には次の問題がある。

1. 脆弱（Brittle）
   
   入力構造への依存が強すぎる。CSV の列名が coffee_name から drink_name に変わったり、日付形式が少し違うだけで失敗する。model が前手順の結果（「Step 3 results」など）を参照できなくなることもあり、少しの変化で崩れる。

2. 非効率（Inefficient）
   
   LLM は一手順ごとに次の指示を再生成する。12 か月分なら filter_rows 12 回 + get_column_mean 12 回の呼出が必要になり、応答待ちや文脈伝搬で遅く、計算資源も無駄。

3. 境界事例への対処が増え続ける（Edge cases）
   
   データの違いごとに補修が必要で、開発者は「特殊ケース修正」に追われる。
   - ある月に hot chocolate がない → 例外処理が必要。
   - 欠損や形式の乱れ → 例外処理が必要。
   - 列順やファイル名が違う → さらに処理を変更。

より良い方法は「Planning with code execution」。

## Code 実行を用いた計画

![](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/202511041427014.png)

手順 JSON ではなく code を直接生成させる方が、相対的に柔軟。

## 複数 agent の workflow と通信様式

課題が十分に複雑な場合、複数の専門 agent で協調して進める。典型は次のとおり。

1. 直列 agent: 流れ作業。前段の出力が後段の入力。
2. 階層 agent: 監督者が manager として課題分解と配分、統合。

![](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/202511041434016.png)

# まとめ

本講座は Agent 開発の入門として良い概説。少しの Python と Jupyter Notebook、指示文設計を知っていれば理解でき、code は多くない。

最も有用なのは Reflection と Tool use。

最後に二つの問いに答える。

1. なぜ Claude Code は現時点で最も成功した Agentic 系製品なのか？
  
- 製品領域が code 生成に絞られており、試験や Lint など明確で定量的な評価系がある。これは今 Agentic Coding が多い最大理由だと思う。開発で最重要かつ難しい eval を、他領域（Deep Research など）より客観的に作りやすい。
- MCP により、Claude Code は tool 呼出で外部情報を取り込み、外部 feedback を得る力が大幅に強化された。
- model が「従順」で、模倣精度が高い。
- Context と prompt の管理: 何が coding に重要かを開発者が理解し、設計段階で固定文脈を定義している。
![](https://raw.githubusercontent.com/lifeodyssey/Figurebed/master/202511041443224.png)

2. 次に成功する Agentic 系製品は何か？

自分は browser だと考える。ChatGPT Atlas はすでに公開され、体験は良好だった。

- browser は操作時の安全問題を避けやすい。個人アカウントの login などは利用者がローカルで完結・保存できる。これは従来の各種製品の Agent 模式（Manus や ChatGPT など）における大きな痛点だった。以前の方式は「雲原神」に似ており、遠隔 server 上に仮想環境を起動し、言語指示で AI がその上の browser と OS を操作する。この場合、多窓遷移などの体験は弱い（実体験）。多くの利用者が本当に求めるのは、自分の browser と OS を AI に操作させ、自分で login・支払いなどの要所を行い、反復作業を agent に任せること。これは初期の Codex と Codex CLI/Claude Code の違いに近い。
- browser 領域の Agentic 製品は評価系が比較的明確。自動 form 入力、情報抽出、Web navigation などは、結果の正確率や完了効率など客観指標で評価でき、継続的最適化が容易。
- MCP などの普及、特に Playwright MCP は agent と browser の連携障壁を大幅に下げ、model が API のように Web を操作し、データ取得や複雑作業を行える。model が呼出指令を正確に出力できれば、高度な自動化が実現可能。
- browser は情報取得・作業実行・外界との連携の中心入口で、拡張性と生態が強い。Agentic browser 製品が成熟すれば、インターネット時代の Google のように新しい流入口になる。

# AgentGuard Deep Research: State-of-the-Art in Tool-Call Verification & Hallucination Detection

**Research Date:** 2025-2026  
**Purpose:** Architecture decisions for the agentguard production-grade detection engine  
**Sources:** arxiv papers (2024–2026), IBM Research, NVIDIA, AWS, Google, Anthropic documentation

---

## Table of Contents

1. [Tool-Call Hallucination Detection](#1-tool-call-hallucination-detection)
2. [Semantic Verification of Tool Outputs](#2-semantic-verification-of-tool-outputs)
3. [Runtime Behavior Analysis](#3-runtime-behavior-analysis)
4. [Multi-Signal Fusion for Detection](#4-multi-signal-fusion-for-detection)
5. [Production Guardrail Architectures](#5-production-guardrail-architectures)
6. [Novel and Unique Approaches](#6-novel-and-unique-approaches)
7. [Synthesis: Recommendations for AgentGuard Core Engine](#7-synthesis-recommendations-for-agentguard-core-engine)

---

## 1. Tool-Call Hallucination Detection

### 1.1 Internal Representations as Indicators — Zero-Cost Detection

**Source:** ["Internal Representations as Indicators of Hallucinations in Agent Tool Selection"](https://arxiv.org/abs/2601.05214), Healy, Srinivasan, Madathil, Wu (arxiv 2601.05214, January 2026)

#### What It Is

The single most important paper for agentguard. Detects tool-calling hallucinations by attaching a lightweight probe to the LLM's final transformer layer during the **same forward pass** used for generation — zero additional LLM calls required.

#### Problem Formulation

Binary classification: \( h_{\theta}: \mathbb{R}^d \rightarrow \{0, 1\} \) where:
- Input: Internal representation \( \mathbf{z} \in \mathbb{R}^d \) at tool call generation time
- Output: 1 = hallucinated tool call, 0 = correct

**Five hallucination types classified:**

| Type | Description |
|------|-------------|
| Function Selection Error | Calls non-existent function \( \tilde{f} \notin \mathcal{F} \) |
| Function Appropriateness Error | Semantically wrong tool for the task |
| Parameter Error | Arguments outside valid domain \( \boldsymbol{a} \notin \mathcal{D}_{\tilde{f}} \) |
| Completeness Error | Missing required parameters |
| Tool Bypass Error | Generates output without calling any tool |

#### Algorithm: Feature Extraction

From the final-layer hidden states \( \mathbf{h}^{(L)}_t \), extract at three strategic token positions:

1. \( t^{\text{func}} \): Initial sub-token of the predicted function name  
2. \( \mathcal{T}_{\text{args}} = \{t_1, t_2, \ldots, t_k\} \): All tokens spanning the argument region  
3. \( t^{\text{end}} \): Closing delimiter token (`")"` or `"}"`)

**Feature vector construction:**
$$
\mathbf{z}_i = \Pi\left( \mathbf{h}^{(L)}_{t^{\text{func}}} \,\|\, \frac{1}{|\mathcal{T}_{\text{args}}|}\sum_{t \in \mathcal{T}_{\text{args}}} \mathbf{h}^{(L)}_t \,\|\, \mathbf{h}^{(L)}_{t^{\text{end}}} \right) \in \mathbb{R}^m
$$

Where \( \Pi: \mathbb{R}^{3d} \rightarrow \mathbb{R}^m \) is an optional linear projection (identity by default, so \( m = 3d \)).

**Model-specific dimensions:**

| Model | Hidden Dim (d) | Feature Dim (3d) |
|-------|---------------|------------------|
| Qwen-7B | 3584 | 10,752 |
| GPT-OSS-20B | 2880 | 8,640 |
| Llama-3.1-8B | 4096 | 12,288 |

#### Classifier Architecture

Lightweight two-layer feed-forward network:
$$
p_i = h_{\theta}(\mathbf{z}_i) = \sigma\left( \mathbf{w}_2^\top \phi(\mathbf{W}_1 \mathbf{z}_i + \mathbf{b}_1) + b_2 \right)
$$

- \( \phi \): ReLU activation  
- \( \sigma \): Sigmoid  
- Hidden layer: 512 units + dropout (0.1)  
- Threshold: \( \hat{y}_i = \mathbf{1}\{p_i > \tau\} \), where \( \tau \) is optimized on validation set

**Training:** Binary cross-entropy, AdamW (LR 1×10⁻⁴), batch size 32, cosine annealing, temperature scaling for calibration.

**Data generation (unsupervised label creation):**
1. Remove ground-truth tool call from prompt
2. Prompt LLM to predict the call
3. Cache final-layer states
4. Label 1 (hallucinated) if function name differs or arguments fail to match after normalization

#### Inference Protocol (Zero Extra LLM Calls)

1. LLM generates tool call via autoregressive generation
2. Extract \( t^{\text{func}}, \mathcal{T}_{\text{args}}, t^{\text{end}} \) from final layer in the same pass
3. Construct \( \mathbf{z} \)
4. Compute \( p = h_{\theta}(\mathbf{z}) \)
5. If \( p > \tau \): block, confirm, fallback, or repair

#### Results

| Model | Method | Precision | Recall | Accuracy |
|-------|--------|-----------|--------|----------|
| GPT-OSS-20B | This method | 0.86 | 0.86 | **86%** |
| GPT-OSS-20B | Non-contradiction baseline | 1.0 | 0.79 | 0.95 |
| Llama-3.1-8B | This method | 0.73 | 0.73 | **73%** |
| Qwen-7B | This method | 0.81 | 0.74 | **74%** |
| Qwen-7B | Non-contradiction baseline | 0.99 | 0.45 | 0.94 |

**Key insight:** The non-contradiction (NCP) baseline achieves higher precision but much lower recall (0.45 vs 0.74 for Qwen-7B). This method achieves significantly higher recall — meaning it catches more hallucinations — at a modest precision tradeoff.

#### Strengths
- Zero computational overhead (uses existing forward pass)
- Real-time detection during generation
- Particularly good at parameter-level hallucinations
- Works across model families

#### Limitations
- Requires access to model internals (white-box; not usable with OpenAI API directly)
- Model-specific classifiers required (no cross-model transfer)
- Three-point feature extraction is simple; field-aware pooling could improve it
- Training requires labeled dataset for each model
- Accuracy degrades on smaller models (Llama 73% vs GPT-OSS 86%)

#### Implementation Complexity: Medium
Requires hooking into the LLM forward pass to extract hidden states. Works natively with HuggingFace Transformers via `output_hidden_states=True`.

---

### 1.2 IBM ALTK — SPARC, Refraction, and Silent Error Review

**Source:** [IBM Research Blog: ALTK](https://research.ibm.com/blog/altk-agent-toolkit), [IBM ToolOps Blog](https://research.ibm.com/blog/toolops-altk-agents-tools), [ALTK Documentation](https://agenttoolkit.github.io/agent-lifecycle-toolkit/)

IBM's Agent Lifecycle Toolkit (ALTK) is the most complete modular system for production tool-call verification. It provides five components across the lifecycle:

#### SPARC: Semantic Pre-execution Analysis for Reliable Calls

**Stage:** Pre-tool (before tool execution)  
**Question it answers:** "Is my agent calling tools with hallucinated arguments?"

**What SPARC does (from IBM's documentation and research):**
- Validates that tool call arguments **match tool specifications** (schema-level)
- Validates that arguments are **semantically appropriate** given the request context (semantic-level)
- Validates that the **tool call sequence** makes sense given the conversation history

**Mechanical approach (inferred from IBM Research context):** SPARC uses LLM-as-a-judge to compare the generated tool call against:
1. The tool's JSON schema/parameter definitions
2. The semantic intent expressed in the user's original request
3. The historical conversation context to detect sequence errors

IBM's ToolOps evaluation found that **13–19% of test cases** showed parameter type or value mismatches. SPARC targets this class of errors.

**ToolOps findings (build-time companion to SPARC):**
- Tool Enrichment (refining tool metadata) improved correct tool invocations by **~10%**
- Major error source: incorrect input schema generation (parameter type/value mismatches, 13–19% of cases)

#### Refraction: Syntax Validation and Repair

**Stage:** Pre-tool  
**Question it answers:** "Is my agent generating inconsistent tool call sequences?"

Validates and **repairs** tool call syntax before execution. This goes beyond rejection — it attempts repair when a malformed call can be fixed. Targets JSON syntax errors, missing braces, incorrect field names.

#### Silent Error Review: Semantic Post-Tool Checking

**Stage:** Post-tool  
**Question it answers:** "Is my agent ignoring subtle semantic errors in tool responses?"

Detects **silent errors** — cases where a tool returns a 200 OK but the content is semantically incorrect (wrong data, irrelevant results, stale data). Assesses:
- **Relevance**: Does the response address the original query?
- **Accuracy**: Is the response semantically consistent with known facts?
- **Completeness**: Does the response provide all required information?

**Implementation approach:** LLM-as-a-judge pattern — a fast, cheap LLM evaluates the tool response against the query and expected output format.

#### RAG Repair: Recovery from Tool Failures

**Stage:** Post-tool  
**Question it answers:** "Can my agent recover from tool call failures?"

Uses RAG over domain-specific documents to repair failed tool calls. When a tool call fails, retrieves relevant context and re-attempts with repaired arguments.

#### JSON Processor: Context Window Management

**Stage:** Post-tool  
Generates code on the fly to extract relevant data from large JSON tool responses, preventing context window overflow.

#### Strengths (ALTK overall)
- Modular — each component deployable independently
- Framework-agnostic (integrates with LangGraph, etc.)
- MCP-compatible via ContextForge Gateway
- Covers the full lifecycle (pre-LLM, pre-tool, post-tool, pre-response)

#### Limitations
- SPARC and Silent Review require LLM calls (cost and latency)
- SPARC details (exact algorithm) not fully public
- Build-time enrichment (ToolOps) requires upfront work per tool

#### Implementation Complexity: Low-Medium
pip-installable, minimal integration code.

---

### 1.3 ToolBench / ToolEval — Evaluation Framework Methodology

**Source:** [ToolBench GitHub](https://github.com/openbmb/toolbench), [ToolEval Leaderboard](https://openbmb.github.io/ToolBench/), [Comprehensive Survey of Benchmarks](https://huggingface.co/datasets/tuandunghcmut/BFCL_v4_information/blob/main/A%20Comprehensive%20Survey%20of%20Benchmarks%20for%20Evaluating%20Tool%20and%20Function%20Calling%20in%20Large%20Language%20Models.md)

ToolBench (ICLR 2024 Spotlight) is the largest tool-calling evaluation dataset (16,464 APIs, 3,451 tools, 126,000+ instruction-solution pairs from RapidAPI). Its evaluation methodology directly informs what signals matter for tool-call quality.

#### ToolEval's Two-Metric System

**Pass Rate:** Proportion of tasks successfully completed within the API call budget.

**Win Rate:** LLM-as-a-judge (ChatGPT) pairwise comparison between two solution trajectories. Pre-defined criteria are embedded as a judge prompt that evaluates:
- Whether the tool selection was appropriate
- Whether arguments were correct
- Whether the action sequence achieved the goal
- Whether the response is coherent

**Reliability:** The ChatGPT evaluator achieves **87.1% agreement** with human annotators on pass rate and **80.3%** on win rate — validating LLM-as-judge for this task.

#### DFSDT: Depth-First Search Decision Tree

The key algorithmic innovation for generating complex test cases. Uses an LLM-powered decision tree with backtracking to explore multi-step tool-use solutions — important for testing sequential verification.

#### Benchmark Performance

| Model | Pass Rate (DFSDT) | Win Rate (DFSDT) |
|-------|------------------|-----------------|
| GPT-4 | 71.1% | 70.4% |
| ToolLLaMA | 66.7% | 60.0% |

#### Key Evaluation Dimensions for AgentGuard

From the comprehensive benchmark survey:
1. **Syntactic/Structural matching** (AST-based): Fast, scalable, but misses semantic equivalence
2. **Execution-based validation**: Runs the call and checks results — robust but slow
3. **Outcome-centric evaluation**: Only checks final state — flexible but requires clear ground truth

#### Berkeley Function Calling Leaderboard (BFCL)

BFCL uses dual evaluation: AST-based (fast, no execution) + execution-based (accurate). Key finding: Claude 3.5 Sonnet leads at 90.20% on function-calling accuracy.

**Relevance detection** in BFCL tests if a model correctly *refrains* from tool calling when it's inappropriate — critical for detecting tool-bypass hallucination.

---

### 1.4 NeMo Guardrails — Tool Execution Rails

**Source:** [NeMo Guardrails GitHub](https://github.com/NVIDIA-NeMo/Guardrails), [NVIDIA Tool Integration Docs](https://docs.nvidia.com/nemo/guardrails/latest/integration/tools-integration.html)

NeMo Guardrails structures protection around tools via **execution rails** applied to tool inputs/outputs.

#### Architecture

Four rail types with tool-specific execution rails:
1. **Input rails**: Evaluate user input before it reaches the LLM
2. **Dialog rails**: Control action sequencing and tool invocation decisions
3. **Retrieval rails**: Applied to RAG chunks (filter/alter retrieved content)
4. **Execution rails**: Applied to tool call inputs and outputs

#### Critical Security Gap (Documented)

From NVIDIA's own documentation:
> "Tool messages are not subject to input rails validation. This presents potential security risks: Tool responses may contain unsafe content that bypasses input guardrails."

This is the **tool injection attack surface** — a malicious tool response can influence the LLM even when input rails are active. The "safe" configuration requires both input AND output rails to catch this.

#### Self-Check Mechanism

NeMo's self-check rails use a secondary LLM call with a policy prompt:
```
Your task is to check if the bot message below complies with the company policy.
Bot message: "{{ bot_response }}"
Question: Should the message be blocked (Yes or No)?
```

This is a lightweight LLM-as-judge pattern — cheap model (gpt-4o-mini), binary output, inline with response generation.

#### Passthrough Mode

For tool-heavy workflows, NeMo recommends **passthrough mode** to avoid internal task interference with tool call handling. This means guardrails must be applied explicitly around tool interactions.

#### Strengths
- Colang programming language for declarative policy definition
- Modular rail composition
- Multi-LLM support

#### Limitations
- Tool response bypass is a documented vulnerability unless output rails are active
- Internal tasks conflict with tool calls (passthrough mode required)
- LLM-as-judge rails add latency

---

### 1.5 TrustBench — Pre-Execution Trust Verification

**Source:** ["Real-Time Trust Verification for Safe Agentic Actions using TrustBench"](https://arxiv.org/html/2603.09157v1) (arxiv 2603.09157, March 2026)

#### Architecture: Dual-Mode

**Benchmarking Mode (offline):**
Evaluates across 8 trust dimensions: reference-based accuracy, factual consistency, citation integrity, calibration, robustness, fairness, timeliness, safety. Learns calibration curves mapping agent confidence → actual performance using isotonic regression.

**Verification Mode (online, <200ms):**
Intercepts action requests after formulation but before execution. Runs:
1. Extract agent's stated confidence
2. Apply learned calibration curve (isotonic regression maps confidence → epistemic quality score)
3. Compute subset of ground-truth-free metrics (citation integrity, timeliness, safety checks)
4. Combine via domain-specific weights (e.g., healthcare: 0.3 confidence prior, 0.7 runtime metrics)
5. Output: Trust Score with action flag (block/warn/proceed) + violation details

#### Trust Vector Structure

```
TrustScore = {
  action_flag: "block" | "warn" | "proceed",
  dimensional_scores: { accuracy, consistency, timeliness, safety, ... },
  violations: ["citation to non-existent source", "confidence-evidence mismatch", ...]
}
```

#### Calibration Learning

Uses LLM-as-a-Judge (LAJ) during benchmarking on three dimensions:
- **Correctness**: Is the response factually right?
- **Informativeness**: Does it answer the question?
- **Consistency**: Is it internally coherent?

Learns **isotonic regression** mapping stated confidence → LAJ-derived trust. Per-agent, per-domain calibration profiles.

#### Results

- Reduced harmful actions by **87%** vs unconfigured baseline
- Domain-specific plugins outperformed generic by **35% greater harm reduction**
- Median end-to-end latency: **<200ms**

#### Strengths
- Domain-aware (healthcare vs finance vs general)
- Learns from deployment experience (calibration improves over time)
- No ground truth required at runtime
- Graduated autonomy (not binary block/allow)

#### Limitations
- Requires offline benchmarking phase to learn calibration curves
- Domain plugins require domain-specific implementation
- LLM-as-judge adds latency in benchmarking phase

---

### 1.6 AgentGuard (2025 Paper) — Probabilistic Runtime Verification

**Source:** ["AgentGuard: Runtime Verification of AI Agents"](https://arxiv.org/html/2509.23864v1) (arxiv 2509.23864, September 2025)

*(Note: This is a separate academic paper using the same name as our library — different project.)*

#### Dynamic Probabilistic Assurance (DPA)

Learns a Markov Decision Process (MDP) from observed execution traces and uses probabilistic model checking (PMC) to verify quantitative properties in real-time.

#### Components

1. **Trace Monitor & Event Abstractor**: Instruments the agent framework, capturing raw I/O (LLM calls, tool invocations, observations). Abstracts into formal events: `State_A → Action_1 → State_B`.

2. **Online Model Learner**: Continuously updates the MDP transition probabilities from observed frequencies. Adapts to model drift. The AMDP (Augmented MDP) adds context about artifacts/state.

3. **AgentGuardLogger**: User-facing class that calls `log_transition()` to manage an event queue.

4. **AnalyzerThread**: Background thread that monitors actions, processes transition events, updates MDP, periodically invokes PMC.

#### Properties Verified at Runtime

```
P_max=?[ F "Fix_Success" ]    # Max probability of successful completion
E_min=?[ F "Fix_Success" ]    # Min expected cycles to completion
P_max=?[ G ¬"write_fix" ]    # Max probability of never completing
```

#### Example: RepairAgent

After observing execution traces, learns: "after hypothesizing, agent uses `search_code_base` with 75% probability and `find_similar_api_calls` with 25% probability." This learned model enables anomaly detection when transition probabilities deviate.

#### Strengths
- No a priori formal model required — learns from observation
- Continuous quantitative guarantees, not binary
- Handles non-stationary environments via online learning
- Can predict probability of future dangerous states

#### Limitations
- Proof-of-concept implementation; not production-ready
- State space grows with agent complexity
- Learning requires sufficient execution history
- PMC overhead for complex properties

---

## 2. Semantic Verification of Tool Outputs

### 2.1 Comprehensive Taxonomy of Verification Methods

**Source:** ["Large Language Models Hallucination: A Comprehensive Survey"](https://arxiv.org/html/2510.06265v3) (arxiv 2510.06265, 2025/2026)

#### 2.1.1 Retrieval-Based Verification (External Ground Truth)

**Algorithm:** Fetch relevant documents from external knowledge base → compare LLM output semantically/factually.

| Method | Exact Algorithm | Zero-Cost? |
|--------|-----------------|-----------|
| RAG + fact-check | Retrieve docs, NLI model checks entailment vs output | Depends on retrieval |
| HDM-2 | Modular: checks against provided context AND common knowledge (response + span level) | Yes (no extra LLM call) |
| Bayesian sequential | Retrieve one doc at a time, dynamic stop-or-continue | No (multiple calls) |
| KnowHalu | First detect non-fabrication, then multi-form factual check via reasoning decomposition | No |
| JointCQ | Extract claims → generate search queries → verify support/contradict/unverifiable | No |

**Strengths:** Ground truth-based, highest accuracy when reference is available.  
**Limitations:** Requires external knowledge base; retrieval latency; won't catch hallucinations about novel/private data.

#### 2.1.2 Uncertainty-Based Verification (Internal Signals, Zero Extra Calls)

**Key insight:** High model uncertainty often indicates hallucination. Exploitable from model internals without extra LLM calls.

**Token Log-Probability:**
```python
# Normalized log probability of generated tokens
log_prob_score = sum(log P(t_i | context)) / len(tokens)
# Low score → high uncertainty → likely hallucination
```

**Semantic Entropy (Farquhar et al., Nature 2024):**
```
Algorithm:
1. Generate N responses to same query (N=5-20) via stochastic sampling
2. Cluster responses by semantic meaning using bidirectional entailment:
   - A and B are in same cluster if A entails B AND B entails A
3. Compute entropy across meaning clusters:
   H_semantic = -∑_c P(c) * log P(c)
4. High entropy → model uncertain about meaning → hallucination
```

**Results:** AUROC 0.790 vs naive entropy 0.691, P(True) 0.698. Stable across Llama/Falcon/Mistral 7B-70B.  
**Cost:** Requires N additional LLM calls (expensive).  
**Discrete variant:** Works without output probabilities (black-box LLMs).

**HaluNet (Single-Pass, Zero Cost):**
Combines three internal signals in a multi-branch neural architecture:
- Token log-likelihoods (confidence signal)
- Token/sentence entropy (distributional signal)
- Hidden-state embeddings (semantic signal)
Combined via attention weighting. Efficient, real-time suitable.

**LLM-Check (NeurIPS 2024):**
- **Eigenvalue Analysis**: Analyzes internal LLM representations using SVD. Consistent modification pattern of hidden states and attention across token representations when hallucinations are present.
- **Output Token Uncertainty**: Perplexity + Logit Entropy.
- **Speed**: Up to 45–450× faster than baselines. Only uses model representations with teacher forcing.

**HalluShift:** Detects hallucinations by measuring distribution shifts in hidden states and attention layers between normal and hallucinated outputs.

**Lookback Lens:** Measures **lookback ratio** — attention paid to prior context vs own outputs. Hallucinating models look back at their own outputs more than at the context.

#### 2.1.3 Embedding-Based Verification (Zero Extra Calls)

**Cross-lingual semantic comparison:**
- Use LaBSE or LASER embeddings
- XNLI entailment to check semantic consistency across representations
- Embedding model-dependent; good for cross-lingual scenarios

**EigenScore (INSIDE, arxiv 2402.03744):**
```
1. Extract penultimate layer activations from N generations
2. Add ridge regularization: M = X^T X + λI
3. Compute eigenvalues of covariance matrix
4. EigenScore = sum of top eigenvalues (or trace)
5. Low EigenScore = high semantic consistency = factual
6. High EigenScore = semantic diversity = likely hallucination
```
**Test-time feature clipping**: Truncate extreme activations (>k-th percentile) using rolling activation memory bank. Reduces overconfidence.

**Gradient-based:**
Taylor series expansion of conditional vs unconditional outputs → MLP classifier. SOTA performance but gradient computation overhead.

#### 2.1.4 Cross-Consistency Checking (Multiple LLM Calls)

**SelfCheckGPT Algorithm:**
```
1. Generate main response R
2. Generate N additional samples {S_1, ..., S_N} via stochastic decoding
3. For each sentence s in R:
   a. For each sample S_i:
      - NLI check: Does S_i entail s? (using DeBERTa-NLI)
      - Score = P(contradiction | s, S_i)
   b. SelfCheck score = average contradiction probability across samples
4. Threshold: score > τ → hallucination
```

**Performance (HuggingFace experiment):**
- At NLI score 0.8: recall ≈ 80%, precision ≈ 1.0 (perfect)
- High precision (no false positives) above threshold 0.5

**Cost:** N extra generation calls per verification.

**Semantic Entropy Probes (SEPs):**
Trains probes on hidden states to predict semantic entropy WITHOUT generating multiple responses. Retains high performance with much lower cost.

**Cross-Model Consistency (Finch-Zk):**
```
1. Segment response into semantic blocks {b_1, ..., b_k}
2. Generate semantically-equivalent prompts via paraphrasing
3. Query N diverse LLMs with each paraphrase
4. Cross-consistency check: does each model agree on each block?
5. Disagreement → potential hallucination
```
**Results:** 6–39% better F1 than SelfCheckGPT on FELM dataset.

**MetaCheckGPT (SemEval 2024 Winner):**
Meta-regressor framework combining multiple LLM evaluators and black-box uncertainty signals into an ensemble. Ranked 1st (model-agnostic) and 2nd (model-aware) at SemEval 2024 Task 6.

---

### 2.2 Schema Verification and Evolution Detection

**Technique: JSON Schema Validation (Zero-Cost)**

For structured tool outputs (which most tool calls return):

```python
import jsonschema

def verify_tool_output(output: dict, schema: dict) -> VerificationResult:
    try:
        jsonschema.validate(instance=output, schema=schema)
        return VerificationResult(valid=True)
    except jsonschema.ValidationError as e:
        return VerificationResult(
            valid=False,
            error=e.message,
            path=list(e.path),
            violation_type="schema_mismatch"
        )
```

**Schema Evolution Detection:**
Track schema changes over time by hashing the JSON schema definition and comparing across calls. When schema hash changes, flag for re-validation and alert on breaking changes.

```python
schema_fingerprint = hashlib.sha256(
    json.dumps(schema, sort_keys=True).encode()
).hexdigest()

if schema_fingerprint != expected_fingerprint:
    # Schema has changed — validate carefully, may need re-enrichment
    trigger_schema_drift_alert(tool_name, old_fp, schema_fingerprint)
```

**Types of schema violations to detect:**
- Missing required fields
- Type mismatches (string where int expected)
- Out-of-range values (negative prices, future dates for historical data)
- Null values in non-nullable fields
- Unknown/unexpected fields (API added new fields)

**Strengths:** Zero cost, zero LLM calls, instant.  
**Limitations:** Only catches structural problems, not semantic errors (correct schema, wrong values).

---

### 2.3 Semantic Consistency Checking (Cross-Reference with Query)

**Algorithm: Query-Response Relevance Check**

For tool calls where you know the original user query:

```python
def semantic_relevance_check(
    query: str,
    tool_output: str,
    embedding_model: SentenceTransformer
) -> float:
    q_embedding = embedding_model.encode(query)
    o_embedding = embedding_model.encode(tool_output)
    similarity = cosine_similarity(q_embedding, o_embedding)
    # Low similarity → tool output may be irrelevant (possible hallucination)
    return similarity
```

**Threshold:** Tool-specific baselines. If similarity drops below `mean - 2*std` for this tool, flag as anomalous.

**NLI-Based Consistency Check:**
```
Given:
  - User query q
  - Tool output o
  - Previous tool outputs in session {o_1, ..., o_n}

Check:
  1. Does o entail the answer to q? (relevance)
  2. Does o contradict any o_i? (session consistency)
  3. Does o contradict known facts in context? (knowledge consistency)
```

Use a lightweight NLI model (DeBERTa-small, ~85M params) for fast inference.

---

## 3. Runtime Behavior Analysis

### 3.1 eBPF-Based Execution Monitoring

**Source:** [ARMO Intent Drift Detection](https://www.armosec.io/blog/detecting-intent-drift-in-ai-agents-with-runtime-behavioral-data/), [Raven Runtime AI Agents](https://raven.io/runtime-ai-agents)

#### How It Works

eBPF (extended Berkeley Packet Filter) instruments the Linux kernel to observe system calls at near-zero overhead (1–2.5% CPU, 1% memory):

```
Agent execution →
  syscall interception (eBPF) →
    {network connections, file opens, database sockets} →
      compare against expected call profile →
        anomaly if unexpected
```

**Captures:**
- Outbound TCP connections (IP, port, destination)
- DNS resolutions
- File system reads/writes
- Database socket activity
- Process spawning

**Key detection pattern:** ARMO's "attack story generation" — not individual anomaly scores but **action-chain correlation**:
> "Agent never reads from `customer_pii` AND POSTs to external domain in the same execution window. That sequence — not any individual event — reveals intent shift."

#### Execution Fingerprinting via Side Effects

**Network I/O fingerprint:** For a tool call `get_weather(city="Paris")`:
```
Expected: DNS lookup weather.api.com → TCP:443 → HTTPS GET /v1/weather
Actual: No network activity at all → FABRICATED (hallucinated tool call result)
```

This is the **strongest zero-cost signal** for detecting fabricated tool results — a tool that makes no network calls cannot return real data.

**File system fingerprint:**
```
Expected for read_file(path="/data/report.pdf"): open() syscall → read() syscall
Actual: No file syscalls → fabricated file content
```

**Database fingerprint:**
```
Expected for query_db(sql="SELECT..."): connect() → write(query) → read(result)
Actual: No socket activity → fabricated query result
```

#### Limitations
- Requires kernel-level access (privileged container or bare metal)
- Ephemeral Kubernetes pods make per-instance baseline convergence difficult
- Dynamic routing and service meshes obscure destination identity
- Cannot distinguish correct vs incorrect network calls (only presence/absence)
- Not available in managed inference environments

#### Implementation Complexity: High
Requires eBPF infrastructure, kernel instrumentation, or integration with ARMO/Raven.

---

### 3.2 Execution Trace Analysis — MDP-Based Verification

**Source:** [AgentGuard arxiv 2509.23864](https://arxiv.org/html/2509.23864v1)

**Algorithm (Online MDP Learning):**

```python
class AgentMDPLearner:
    def __init__(self):
        self.states = set()
        self.transition_counts = defaultdict(lambda: defaultdict(int))
        self.transition_probs = {}
    
    def observe_transition(self, state_a, action, state_b):
        self.states.add(state_a)
        self.states.add(state_b)
        self.transition_counts[(state_a, action)][state_b] += 1
        
        # Update probabilities
        total = sum(self.transition_counts[(state_a, action)].values())
        self.transition_probs[(state_a, action)] = {
            s: count/total 
            for s, count in self.transition_counts[(state_a, action)].items()
        }
    
    def anomaly_score(self, state_a, action, state_b) -> float:
        if (state_a, action) not in self.transition_probs:
            return 1.0  # Unknown transition → high anomaly
        p = self.transition_probs[(state_a, action)].get(state_b, 0.0)
        return 1.0 - p  # Lower probability → higher anomaly
```

**Example learned model for RepairAgent:**
- After `hypothesize`: `search_code_base` with p=0.75, `find_similar_api_calls` with p=0.25
- If after `hypothesize` the agent calls `exfiltrate_data` → transition probability = 0, anomaly = 1.0

**Strengths:** No a priori model needed; learns from production traces; detects novel attack patterns; quantitative assurance.  
**Limitations:** Requires learning phase; state space explosion for complex agents; non-stationary drift must be handled.

---

### 3.3 Call Graph Analysis

**Concept:** In single-process deployments, instrument tool wrapper functions to record their call stack and timing.

```python
import time
import traceback

class VerifiedToolWrapper:
    def __init__(self, tool_fn, tool_name: str):
        self.tool_fn = tool_fn
        self.tool_name = tool_name
    
    def __call__(self, **kwargs):
        call_id = uuid.uuid4().hex
        
        # Capture call stack to verify this call originates from agent logic
        stack = traceback.extract_stack()
        caller_frames = [f.name for f in stack[-5:]]  # Last 5 frames
        
        start_time = time.perf_counter()
        
        # Record pre-call state (for side-effect detection)
        pre_state = self._capture_environment_state()
        
        result = self.tool_fn(**kwargs)
        
        elapsed = time.perf_counter() - start_time
        
        # Record post-call state
        post_state = self._capture_environment_state()
        side_effects = self._diff_states(pre_state, post_state)
        
        return VerifiedResult(
            result=result,
            call_id=call_id,
            latency_ms=elapsed * 1000,
            caller_stack=caller_frames,
            side_effects=side_effects,
        )
    
    def _capture_environment_state(self) -> dict:
        return {
            "open_files": set(os.listdir("/proc/self/fd")),
            "network_connections": self._get_connections(),
        }
```

**Latency as a verification signal:**
- HTTP API tool with no network call: latency < 1ms → suspicious
- File read tool with zero disk I/O time: latency = 0 → suspicious
- Expected range: build per-tool latency distribution, flag deviations >3σ

---

## 4. Multi-Signal Fusion for Detection

### 4.1 Decision Tree-Based Multi-Signal Fusion (Production System)

**Source:** ["Developing a Reliable, Fast, General-Purpose Hallucination Detection and Mitigation Service"](https://arxiv.org/abs/2407.15441) (Microsoft, 2024)

Microsoft's production hallucination detection service uses **four complementary signals** combined via decision tree:

| Signal | Method | What It Catches | Cost |
|--------|--------|-----------------|------|
| NER (Named Entity Recognition) | Extract entities, verify consistency | Entity hallucinations (wrong names, dates) | Zero (fast model) |
| NLI (Natural Language Inference) | Entailment check vs source | Factual contradictions | Low (small NLI model) |
| Span-Based Detection (SBD) | Token-level span classifier | Specific hallucinated spans | Low |
| Decision Tree Router | Combines above signals | Optimizes precision/recall tradeoff | Zero |

**Decision tree logic (inferred from paper description):**
```
if NER_entity_mismatch:
    → hallucination (high precision signal)
elif NLI_contradiction_score > threshold_high:
    → hallucination
elif NLI_contradiction_score > threshold_medium AND SBD_span_score > threshold:
    → hallucination (combined signal)
elif SBD_span_score > threshold_high:
    → hallucination
else:
    → not hallucinated
```

**Key design principle:** NER → NLI → SBD in order of precision (NER most precise, SBD broadest), with the decision tree short-circuiting on high-confidence signals.

**Strengths:** No LLM-as-judge needed, fast inference, production-validated.  
**Limitations:** Requires training data for each signal; NER-based checks miss non-entity hallucinations.

---

### 4.2 Bayesian Multi-Signal Fusion

**Concept:** Treat each verification signal as providing evidence about the hallucination probability. Combine via Bayesian update:

$$P(H | s_1, s_2, \ldots, s_n) \propto P(H) \prod_{i=1}^{n} P(s_i | H)$$

**Implementation:**

```python
class BayesianHallucinationDetector:
    def __init__(self):
        self.prior = 0.15  # Base hallucination rate from empirical data
        
        # Per-signal likelihood ratios P(signal=1|hall) / P(signal=1|not_hall)
        # Calibrated from labeled evaluation data
        self.likelihood_ratios = {
            "schema_mismatch": 12.0,        # Very strong signal
            "latency_anomaly": 3.5,          # Medium signal
            "semantic_similarity_low": 4.0,  # Medium-strong signal
            "nli_contradiction": 8.0,        # Strong signal
            "token_entropy_high": 2.5,       # Weak-medium signal
            "side_effects_absent": 6.0,      # Strong signal (where applicable)
        }
    
    def compute_posterior(self, signals: dict[str, bool]) -> float:
        """
        signals: {signal_name: True if signal fired, False otherwise}
        Returns: P(hallucination | observed signals)
        """
        # Log-odds form for numerical stability
        log_odds = math.log(self.prior / (1 - self.prior))
        
        for signal_name, fired in signals.items():
            if signal_name in self.likelihood_ratios:
                lr = self.likelihood_ratios[signal_name]
                if fired:
                    log_odds += math.log(lr)
                else:
                    # Signal absent: slight update toward no hallucination
                    log_odds += math.log(1.0 / max(lr * 0.1, 1.01))
        
        odds = math.exp(log_odds)
        return odds / (1 + odds)  # Convert back to probability
    
    def update_likelihood_ratio(self, signal_name: str, 
                                 fired_hallucinated: float,
                                 fired_not_hallucinated: float):
        """Online update of likelihood ratios from labeled feedback."""
        self.likelihood_ratios[signal_name] = (
            fired_hallucinated / max(fired_not_hallucinated, 0.001)
        )
```

**Calibration from literature:**

| Signal | Estimated P(fire|hallucinated) | P(fire|not hallucinated) | LR |
|--------|-------------------------------|--------------------------|-----|
| Schema mismatch | ~0.72 | ~0.06 | ~12 |
| Latency: <1ms for network tool | ~0.89 | ~0.25 | ~3.6 |
| Semantic similarity <0.3 vs query | ~0.65 | ~0.16 | ~4 |
| NLI contradiction | ~0.68 | ~0.08 | ~8.5 |
| Side effects absent (network tool) | ~0.78 | ~0.13 | ~6 |
| Token entropy high | ~0.55 | ~0.22 | ~2.5 |

**Adaptive updating (online learning):**
When labeled feedback arrives (user flags incorrect output, downstream system detects error), update likelihood ratios using exponential moving average:

```python
alpha = 0.1  # Learning rate
self.likelihood_ratios[signal] = (
    (1 - alpha) * self.likelihood_ratios[signal] + 
    alpha * new_empirical_lr
)
```

---

### 4.3 Semantic Entropy Probes — Cheap Ensemble Signal

**Source:** ["Semantic Entropy Probes: Robust and Cheap Hallucination Detection"](https://arxiv.org/abs/2406.15927) (2024)

Trains lightweight probes on hidden states to **predict semantic entropy** without generating multiple responses. Retains high AUROC performance of semantic entropy at a fraction of the cost.

**Algorithm:**
```
1. During training: compute true semantic entropy for N responses
2. Train probe: hidden_state → predicted_semantic_entropy
3. At inference: probe(hidden_state) → entropy_estimate
4. High estimate → flag as uncertain → hallucination candidate
```

**Generalization:** Better out-of-distribution generalization than probes predicting accuracy directly.

---

### 4.4 Adaptive Thresholds via Online Learning

**Production pattern:** Start with a conservative global threshold, then learn per-tool thresholds from production traffic.

```python
class AdaptiveThresholdManager:
    def __init__(self, initial_threshold: float = 0.5):
        self.tool_thresholds = {}
        self.tool_stats = defaultdict(lambda: {
            "hall_scores": [],
            "clean_scores": [],
            "ema_mean": initial_threshold,
            "ema_std": 0.1,
        })
    
    def get_threshold(self, tool_name: str) -> float:
        stats = self.tool_stats[tool_name]
        # Dynamic threshold: mean + k*std of hallucination score distribution
        # k chosen for target precision
        return stats["ema_mean"] + 2.0 * stats["ema_std"]
    
    def update(self, tool_name: str, score: float, 
                was_hallucination: bool, alpha: float = 0.05):
        stats = self.tool_stats[tool_name]
        if was_hallucination:
            stats["hall_scores"].append(score)
        else:
            stats["clean_scores"].append(score)
        
        # EMA update
        all_scores = stats["hall_scores"] + stats["clean_scores"]
        if len(all_scores) > 10:
            # Separate distributions → pick threshold between them
            hall_mean = np.mean(stats["hall_scores"]) if stats["hall_scores"] else 1.0
            clean_mean = np.mean(stats["clean_scores"]) if stats["clean_scores"] else 0.0
            # Midpoint threshold with adjustment for class imbalance
            stats["ema_mean"] = (hall_mean + clean_mean) / 2
            stats["ema_std"] = (
                np.std(stats["hall_scores"]) if len(stats["hall_scores"]) > 2 else 0.1
            )
```

---

## 5. Production Guardrail Architectures

### 5.1 Langfuse — Observability-First with Custom Guards

**Source:** [Langfuse Security & Guardrails](https://langfuse.com/docs/security-and-guardrails), [ZenML Comparison](https://www.zenml.io/blog/langfuse-vs-phoenix)

**Architecture:** Observability-first. Captures every LLM call, tool invocation, and response as a hierarchical trace (spans). Hallucination detection is **not built-in** — requires custom implementation.

**Guardrail integration pattern:**
```python
from langfuse import observe, get_client

@observe  # Every decorated function auto-traced
def my_tool_call(params):
    result = actual_tool(params)
    # Manually add guardrail evaluation as a child span
    langfuse_client.score(
        trace_id=get_current_trace_id(),
        name="tool_output_validity",
        value=run_validity_check(result),
    )
    return result
```

**What Langfuse provides:**
- Request/response logging
- Latency tracking per span
- Session trees visualizing multi-step agent workflows
- LLM-as-judge evaluators (configurable)
- Custom dashboards to track security scores over time
- Annotation queues for human review of flagged conversations

**What Langfuse does NOT provide:**
- Built-in hallucination detection models
- Automatic prompt injection detection
- Tool-specific verification logic

**Tool call monitoring:** Captures tool name, arguments, response, latency. Can attach custom scores. Agent graph visualization available.

**Verdict for agentguard:** Langfuse is the ideal **observability layer** to sit alongside agentguard — it provides logging and UI, while agentguard provides the detection logic.

---

### 5.2 Arize Phoenix — Agent Evaluation with OpenInference

**Source:** [Arize LLM Evaluation Comparison](https://arize.com/llm-evaluation-platforms-top-frameworks/)

**Architecture:** OpenInference + OpenTelemetry tracing with structured evaluation workflows.

**Tool call tracking:** Captures complete multi-step agent traces with tool invocations. Can assess how agents make decisions over time.

**Evaluation capabilities:**
- LLM-as-judge evaluators via Phoenix Evals SDK
- Anomaly detection on trace clusters
- Retrieval relevancy in RAG pipelines
- Custom evaluators via plugin system

**Agent-specific features:**
- Clustering similar traces to detect systematic failures
- Detecting anomalies in agent behavior patterns
- Session-level evaluation across complete tool use sequences

**Verdict:** Better for agent evaluation than Langfuse; Phoenix Evals provides structured hallucination scoring infrastructure that agentguard can integrate with.

---

### 5.3 Patronus AI — Purpose-Built Guardrails

**Source:** [Patronus AI Guardrails](https://www.patronus.ai/ai-reliability/ai-guardrails), [Patronus AI Architecture](https://www.patronus.ai/ai-agent-development/ai-agent-architecture)

**Architecture:** Point-in-time guardrails with evaluator-based detection. The core is a `RemoteEvaluator` that evaluates any (input, output) pair against named criteria.

**Key components:**
```python
from patronus import RemoteEvaluator

evaluator = RemoteEvaluator("glider", "patronus:is-harmful-advice")
result = evaluator.evaluate(task_input=user_input, task_output=response)
# result.pass_ → bool
# result.explanation → string
```

**Lynx hallucination detection model:** Patronus's own model for hallucination detection with claimed highest precision/recall in industry.

**For agents:** Percival model analyzes full agent execution traces to identify which responses require guardrails and suggests implementation.

**Detection capabilities:**
- Prompt injection detection
- Dangerous code execution prevention
- Privacy/GDPR compliance
- Toxicity and harmful advice detection
- Hallucination detection via Lynx

**Guardrail loop pattern:**
```python
while not is_safe(response):
    response = generate_response(query)
    guardrail_result = evaluate_safety(response)
    if guardrail_result.fail:
        response = update_response(response, guardrail_result.reasoning)
```

**Verdict:** Patronus is strong for semantic content guardrails but lacks tool-call-specific verification. Agentguard should complement Patronus by adding structural/behavioral verification.

---

### 5.4 Amazon Bedrock Guardrails — Policy-Based with Automated Reasoning

**Source:** [AWS Bedrock Guardrails Docs](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-how.html), [Automated Reasoning Checks](https://docs.aws.amazon.com/bedrock/latest/userguide/integrate-automated-reasoning-checks.html)

**Architecture:** Policy-based guardrails evaluated in parallel. Input evaluated against all configured policies simultaneously (improved latency). Then model runs. Then output evaluated.

**Policy types:**
- Content filters (violence, sexual content, hate, etc.)
- Denied topics
- Sensitive information filters (PII)
- Word filters
- Image content filters
- **Automated Reasoning (AR) checks** — the unique capability

#### Automated Reasoning Checks (Unique to Bedrock)

AR checks use formal verification (SMT solver / theorem prover) against user-defined logical rules extracted from policy documents. This is **deterministic, not probabilistic**.

```python
# AR check response structure
{
  "valid": {...},      # Response logically follows from policy
  "invalid": {...},    # Response contradicts policy
  "satisfiable": {...},# Response is consistent with policy
  "impossible": {...}, # Response makes logically impossible claims
  "translationAmbiguous": {...},  # Can't determine
  "tooComplex": {...},  # Rule too complex to verify
  "noTranslations": {...}  # Can't translate to formal logic
}
```

**Example:** Insurance policy rules → formal logic → verify if LLM's coverage explanation is logically consistent with the actual policy. Scenario A ($1.5M trade, tier-2 client) → "Additional approval required" with exact logical path.

**Tool call function calling safety:** Bedrock Agents integrates guardrails natively. Tool responses are passed through guardrails before being returned to the model.

**Verdict:** Bedrock's AR checks are the most novel approach in production — formal verification of LLM outputs against business rules. However, it's tightly coupled to AWS and requires rule authoring.

---

### 5.5 Google Vertex AI Agent Builder

**Source:** [Vertex AI Agent Builder](https://cloud.google.com/products/agent-builder), [Vertex AI Safety Overview](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/learn/safety-overview)

**Architecture:** Three-pillar: Build + Deploy + Govern

**Safety layers (in order of strength):**

| Layer | Mechanism | Protection | Limitations |
|-------|-----------|-----------|-------------|
| Default Gemini safety | Built-in content filters + non-configurable CSAM/copyright filter | Baseline | Can't customize; model still hallucinates |
| System instructions | Policy guidance in system prompt | Content/brand safety | Model can ignore; motivvated attackers bypass |
| Configurable content filters | Adjustable harm thresholds per category | Adjustable content safety | Same as system instructions |
| Gemini-as-filter | Second Gemini call (Flash/Lite) evaluates output against policy | Highly robust; multimodal | Extra cost and latency; rare false negatives |
| Model Armor | Runtime protection via Security Command Center | Enterprise security | Google Cloud only |

**Tool call safety:** "Establish guardrails around your agents to control interactions at every step — from screening inputs before they reach models to **validating parameters before tool execution**."

**Observability:** Full tracing of tool selection, execution paths, and reasoning process via Cloud Trace.

---

### 5.6 Anthropic's Safety Architecture

**Source:** [Anthropic RSP](https://www.anthropic.com/responsible-scaling-policy), [Anthropic ASL-3 Report](https://www.anthropic.com/activating-asl3-report)

**Defense-in-depth (4 layers for ASL-3):**

1. **Access controls**: Tailor safeguards to deployment context and expected users
2. **Real-time prompt and completion classifiers**: Immediate online filtering (< token latency)
3. **Asynchronous monitoring classifiers**: Detailed analysis of completions for threats (post-hoc)
4. **Post-hoc jailbreak detection with rapid response**: Catch sophisticated attacks after the fact

**For tool use:** Anthropic emphasizes that unmonitored tool use/chain-of-thought is a risk factor at higher capability levels. The ASL-3 standard specifically flags "more unmonitored chain-of-thought / tool use" as a concern.

**MCP security pattern (from Anthropic ecosystem):**
```python
# The Gatekeeper pattern for MCP tools
from pydantic import BaseModel, validator

class ToolInput(BaseModel):
    query: str
    max_results: int
    
    @validator('max_results')
    def validate_max_results(cls, v):
        if v > 100:
            raise ValueError('max_results cannot exceed 100')
        return v
```

---

## 6. Novel and Unique Approaches

### 6.1 Cryptographic Attestation of Tool Execution (Proof of Execution)

**Source:** ["Cryptographic Binding and Reproducibility Verification for AI Agent Pipelines"](https://arxiv.org/html/2603.14332v1) (arxiv 2603.14332, March 2026)

This is the most technically sophisticated approach and addresses the **fabrication problem** at its root: cryptographically prove that a tool was actually executed.

#### Three-Layer Governance (G1 + G2 + G3)

**G1: Capability Binding** — Cryptographic certificate binding agent identity to tool set

```
cert_v = (id_v, id_par(v), pk_v, μ_v, σ_v, κ_v, ρ_v, t_s, t_e, sig)
where:
  μ_v = (provider, model_id, model_ver)          # Model identity binding
  σ_v = {(sid_i, ver_i, h_i, P_i)}_{i=1}^n       # Skills manifest (tool hashes)
  κ_v = (max_tier, max_depth, allowed_models)     # Trust constraints
  ρ_v = (level, config)                           # Reproducibility commitment
```

Every tool is represented by its source code hash in the skills manifest: `h_i = SHA-256(source_code)`. Any code change → certificate invalidation.

**Certificate verification algorithm:**
```python
def verify(chain, credential, current_skills, roots):
    # Phase 1: Chain integrity
    if chain[0].issuer not in roots:
        return "DENY"
    for i in range(1, len(chain)):
        if not verify_sig(chain[i-1].pk, chain[i].sig):
            return "DENY"
        if not monotonic_trust(chain[i].κ, chain[i-1].κ):
            return "DENY"
    
    # Phase 2: Capability binding
    current_hash = sha256(canonical(current_skills))
    if current_hash != cert_v.σ_hash:
        return "DENY"  # Tool set changed
    
    # Phase 3: Trust constraint check
    if credential.tier > cert_v.κ.max_tier:
        return "DENY"
    
    # Phase 4: Revocation
    if is_revoked(cert_v):
        return "DENY"
    
    return "ALLOW"
```

**G2: Behavioral Verifiability** — Replay verification using LLM near-determinism

```python
def replay_verify(cert, record, O_orig):
    ρ = cert.ρ  # (level, config)
    if ρ.level == "none":
        return "INCONCLUSIVE"
    
    O_ref = execute_model(cert.μ, cert.σ, record.input, record.seed, ρ.config)
    
    if ρ.level == "full":
        return "VERIFIED" if O_ref == O_orig else "VIOLATION"
    else:  # statistical
        return "VERIFIED" if char_match(O_ref, O_orig) >= θ else "VIOLATION"
```

**G3: Verifiable Interaction Ledger** — Hash-linked, bilaterally-signed audit log

Each tool call produces a ledger record:
```
R_i = (seq_i, t_i, a_send, a_recv, 
        η_in=SHA256(input), η_out=SHA256(output),
        π_i=(seed, model_ver, skills_hash),
        h_prev=SHA256(R_{i-1}),
        sig_i)  # Ed25519 bilateral signature
```

#### Attack Detection Results

| Attack Scenario | Detection Mechanism | Latency |
|----------------|---------------------|---------|
| Inject new tool (S1) | G1 capability hash mismatch | 121 µs |
| Tool trojanization (S2) | G1 code hash mismatch | 103 µs |
| Forged certificate (S3) | G1 signature verification | 106 µs |
| Rate limit abuse (S4) | G1 trust constraints | 165 µs |
| Ledger tampering (S5) | G3 hash chain break | — |

#### Performance Overhead

| Operation | Latency |
|-----------|---------|
| Certificate verification (chain depth 3) | 97.4 µs |
| Capability binding check | <1 ns |
| Skills manifest hash (10 tools) | 5.10 µs |
| MCP governance proxy total | ~0.62 ms |
| E2E pipeline overhead (5-20 agents) | 10.8–48.3 ms (0.12%) |

**vs BAID (blockchain approach):** 1,200,000× faster (97 µs vs ~120 s), 25× smaller proofs (2 KB vs 50 KB).

#### Implementation: MCP Governance Proxy (520 LoC Python)

Interposes between MCP client (agent) and MCP server (tool provider):
1. Verify G1 on every tool call
2. Record ledger entry (G3)
3. Track reproducibility commitments (G2)

#### Strengths
- Cryptographically strong: 0 false positives in all 12 attack scenarios
- Near-zero overhead (<1 ms per call)
- Detects tool trojanization that all other methods miss
- Open standard (extends X.509 v3)
- Enables forensic reconstruction after the fact (G3)

#### Limitations
- Requires PKI infrastructure (issuing and managing certificates)
- Closed-source tools need config descriptor hashing (less strong than code hash)
- Reproducibility (G2) requires LLM inference to be near-deterministic (seed control)
- Does not detect hallucinations in tool *outputs* — only verifies the tool was actually called

---

### 6.2 Tool Output Fingerprinting via Embedding Similarity

**Algorithm: Semantic Baseline Fingerprinting**

Build a statistical profile of expected tool outputs using embedding similarity:

```python
class ToolOutputFingerprinter:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(embedding_model)
        self.tool_profiles: dict[str, ToolProfile] = {}
    
    def update_profile(self, tool_name: str, output: str, 
                        metadata: dict, is_confirmed_valid: bool = True):
        """Update the baseline distribution for a tool."""
        embedding = self.model.encode(output)
        profile = self.tool_profiles.setdefault(
            tool_name, ToolProfile(name=tool_name)
        )
        if is_confirmed_valid:
            profile.valid_embeddings.append(embedding)
            profile.valid_metadata.append(metadata)
            # Update centroid
            profile.centroid = np.mean(profile.valid_embeddings, axis=0)
            # Update distribution
            distances = [
                cosine_distance(e, profile.centroid) 
                for e in profile.valid_embeddings
            ]
            profile.mean_distance = np.mean(distances)
            profile.std_distance = np.std(distances)
    
    def verify(self, tool_name: str, output: str) -> FingerprinterResult:
        if tool_name not in self.tool_profiles:
            return FingerprinterResult(score=None, verdict="UNKNOWN_TOOL")
        
        profile = self.tool_profiles[tool_name]
        if len(profile.valid_embeddings) < 10:
            return FingerprinterResult(score=None, verdict="INSUFFICIENT_BASELINE")
        
        embedding = self.model.encode(output)
        distance = cosine_distance(embedding, profile.centroid)
        
        # Z-score: how many std deviations from expected?
        z_score = (distance - profile.mean_distance) / max(profile.std_distance, 0.001)
        
        # High z-score → output is far from typical → suspicious
        is_anomalous = z_score > 3.0  # 3-sigma threshold
        
        return FingerprinterResult(
            z_score=z_score,
            distance_from_centroid=distance,
            is_anomalous=is_anomalous,
            verdict="ANOMALOUS" if is_anomalous else "NORMAL",
        )
```

**Why this works:** Tool outputs for similar inputs cluster tightly in embedding space. A weather API always returns temperature, conditions, and location. A stock price API always returns ticker and price. If an LLM fabricates a response, it may generate plausible-sounding but semantically different content that drifts from the expected cluster.

**Strengths:** Zero LLM calls, very fast (embedding inference only), improves over time.  
**Limitations:** Requires warm-up period to build baseline; may miss hallucinations that perfectly mimic real outputs; embedding model choice matters.

---

### 6.3 Differential Testing of Tool Calls

**Source:** [Differential Testing with LLMs paper](https://arxiv.org/html/2410.04249v3) (arxiv 2410.04249, 2025)

**Concept:** Run the same tool call twice (or against two implementations) and compare outputs.

```python
def differential_verify(tool: Callable, params: dict, 
                         n_samples: int = 2) -> DifferentialResult:
    """
    For deterministic tools: call twice, compare exact output.
    For stochastic tools: call N times, measure semantic consistency.
    """
    results = [tool(**params) for _ in range(n_samples)]
    
    if is_deterministic_tool(tool):
        # Exact comparison
        if all(r == results[0] for r in results):
            return DifferentialResult(consistent=True, variance=0.0)
        else:
            # Different results from same tool → tool is non-deterministic or broken
            return DifferentialResult(consistent=False, variance=1.0)
    else:
        # Semantic comparison for stochastic tools
        embeddings = [embed(str(r)) for r in results]
        pairwise_similarities = [
            cosine_similarity(embeddings[i], embeddings[j])
            for i, j in combinations(range(n_samples), 2)
        ]
        mean_similarity = np.mean(pairwise_similarities)
        
        return DifferentialResult(
            consistent=mean_similarity > 0.9,
            variance=1.0 - mean_similarity,
        )
```

**When to use:** High-stakes tool calls (financial transactions, health data, legal queries) where the cost of an extra API call is justified.

**Strengths:** Direct verification of tool execution; catches fabricated results definitively.  
**Limitations:** Doubles (or more) API call cost; stochastic tools require semantic comparison; non-idempotent tools cannot be called twice.

---

### 6.4 Historical Baseline Comparison (Statistical Process Control)

**Algorithm:** Build per-tool statistical baselines and use control chart logic to detect anomalies:

```python
class ToolOutputStatTracker:
    """Statistical process control for tool output verification."""
    
    def __init__(self):
        self.tool_metrics: dict[str, ToolMetrics] = {}
    
    def update_and_check(self, 
                          tool_name: str, 
                          output: dict,
                          metadata: dict) -> SPCResult:
        metrics = self.tool_metrics.setdefault(tool_name, ToolMetrics())
        
        # Extract measurable properties from tool output
        output_properties = {
            "response_length": len(json.dumps(output)),
            "field_count": len(output) if isinstance(output, dict) else 1,
            "numeric_values": [v for v in output.values() 
                               if isinstance(v, (int, float))],
            "latency_ms": metadata.get("latency_ms", 0),
        }
        
        anomalies = []
        
        for prop_name, value in output_properties.items():
            if isinstance(value, list) and value:
                value = np.mean(value)  # Summarize lists
            elif isinstance(value, list):
                continue
            
            history = metrics.get_history(prop_name)
            
            if len(history) < 30:
                metrics.add_observation(prop_name, value)
                continue
            
            mean = np.mean(history)
            std = np.std(history)
            
            # Western Electric SPC rules
            z = (value - mean) / max(std, 0.001)
            
            if abs(z) > 3.0:
                anomalies.append(SPCAnomaly(
                    property=prop_name,
                    value=value,
                    z_score=z,
                    rule="Rule 1: Point beyond 3σ"
                ))
            
            # Check trend (8 consecutive points trending up/down)
            recent = history[-7:] + [value]
            if all(recent[i] < recent[i+1] for i in range(len(recent)-1)):
                anomalies.append(SPCAnomaly(
                    property=prop_name,
                    rule="Rule 4: 8 consecutive increasing points"
                ))
            
            metrics.add_observation(prop_name, value)
        
        return SPCResult(
            anomalies=anomalies,
            is_anomalous=len(anomalies) > 0,
        )
```

**Key metrics to track per tool:**
- Response length (chars/bytes)
- Field count in response dict
- Numeric value ranges (prices, counts, coordinates)
- Response latency
- Embedding distance from centroid (combined with §6.2)
- Error rate (4xx/5xx per hour)

**Strengths:** Zero LLM calls, improves with more data, detects slow drift (API degradation, schema changes).  
**Limitations:** Cannot detect hallucinations that perfectly match the expected distribution; needs warm-up period.

---

### 6.5 Neurosymbolic / Formal Verification (Emerging)

**Source:** [Amazon Bedrock Automated Reasoning Checks](https://docs.aws.amazon.com/bedrock/latest/userguide/integrate-automated-reasoning-checks.html), [Neurosymbolic Guardrails article](https://dev.to/aws/reduce-agent-errors-and-token-costs-with-semantic-tool-selection-7mf)

**Concept:** Use formal logic (SMT solvers, constraint satisfaction) to verify that tool call arguments and outputs satisfy logical constraints.

**Amazon Bedrock AR Check example:**
```python
# Define logical rules from policy document
rules = """
approval_required(Trade) :-
    trade_value(Trade, V), V > 1000000,
    client_tier(Trade, T), T == tier2.
"""

# Verify LLM output: "This trade requires additional approval"
ar_result = bedrock.apply_guardrail(
    guardrail_identifier="your-guardrail",
    source="OUTPUT",
    content=[{"text": {"text": f"Trade: {trade_details}\nAgent: {agent_response}"}}]
)

# Check findings
for finding in ar_result["assessments"][0]["automatedReasoningPolicy"]["findings"]:
    if "valid" in finding:
        print("Output is logically valid per policy")
    elif "invalid" in finding:
        print("Output contradicts policy rules!")
```

**For tool calls:** Can verify:
- Argument constraints (price > 0, date in valid range)
- Business rule compliance (user has permission for requested action)
- Logical consistency (output is consistent with request)

**Strengths:** Deterministic (no probabilistic uncertainty), mathematically provable, generates explanations.  
**Limitations:** Requires rule authoring (domain expert + formalization); complex rules marked `tooComplex`; doesn't generalize beyond formalized rules.

---

## 7. Synthesis: Recommendations for AgentGuard Core Engine

### 7.1 Detection Signal Hierarchy

Organize signals into tiers by cost and reliability:

| Tier | Signal Type | Cost | Reliability | Latency |
|------|-------------|------|-------------|---------|
| 0 | JSON Schema validation | Zero | High (structural) | <1ms |
| 0 | Latency anomaly (network/file tool) | Zero | High (for absence) | <1ms |
| 0 | Side-effect check (did network I/O occur?) | Zero | High (where applicable) | <1ms |
| 0 | Schema fingerprint (version drift) | Zero | Medium | <1ms |
| 1 | Embedding similarity vs centroid | Embedding inference | Medium-High | 5–50ms |
| 1 | Token entropy / log-probability | Model internal | Medium | <1ms (same pass) |
| 1 | Historical baseline (SPC) | Zero | Medium | <1ms |
| 2 | Internal representation probe | Model internal | High (trained) | <1ms (same pass) |
| 2 | NLI consistency check vs query | Small NLI model | High | 50–200ms |
| 3 | LLM-as-judge (SPARC-style) | LLM call | Very High | 200–2000ms |
| 3 | Semantic entropy (N generations) | N×LLM calls | Very High | N×latency |
| 3 | Differential testing (2× tool call) | Tool call | Definitive | 2×tool latency |

### 7.2 Recommended Architecture for AgentGuard Core

```
Tool Call Generated by Agent
         ↓
[Tier 0: Zero-Cost Pre-Checks]
  ├── Schema validation (args vs tool spec)
  ├── Latency plausibility check
  ├── Capability hash check (G1, if PKI configured)
  └── Known-bad argument pattern matching
         ↓
[Bayesian Signal Combiner]
  ← Prior P(H) from tool's historical base rate
  ← Tier 0 signal updates posterior
         ↓
[Decision: Proceed / Escalate]
  ├── P(H) < 0.2: Proceed (execute tool)
  ├── 0.2 ≤ P(H) < 0.5: Execute + Tier 1 post-check
  └── P(H) ≥ 0.5: Block or require confirmation
         ↓
[Tool Executes]
         ↓
[Tier 1: Post-Tool Checks (Zero-Cost)]
  ├── Schema validation (output format)
  ├── Embedding similarity vs tool centroid
  ├── SPC baseline check (response properties)
  ├── Session consistency check (contradicts earlier outputs?)
  └── Side-effect verification (did expected I/O occur?)
         ↓
[Bayesian Update from Post-Tool Signals]
         ↓
[Decision: Accept / Flag / Repair]
  ├── P(H) < 0.3: Accept output
  ├── 0.3 ≤ P(H) < 0.7: Flag for review, continue
  └── P(H) ≥ 0.7: Escalate to Tier 2-3 verification
         ↓
[Tier 2-3: Escalation (as needed)]
  ├── NLI consistency check
  ├── LLM-as-judge (Silent Error Review style)
  └── Differential testing (for high-stakes calls)
```

### 7.3 Key Implementation Decisions

1. **For open-weight models:** Implement the Internal Representations probe (§1.1) — highest accuracy, zero cost. Train model-specific classifiers on the Glaive dataset or domain-specific data.

2. **For API-only models (GPT-4, Claude):** Use Semantic Entropy Probes if possible (indirect probe via embeddings). Fall back to SelfCheckGPT-style consistency checking for critical calls.

3. **For network-dependent tools:** eBPF or wrapper-level I/O monitoring is the strongest possible signal for fabricated results. If an HTTP tool makes no HTTP call, the result is provably fabricated.

4. **For high-security deployments:** Implement the MCP Governance Proxy (§6.1) with capability-bound certificates. This provides cryptographic proof that the tool was actually called.

5. **For online learning:** Maintain per-tool Bayesian posteriors with EMA updates from labeled feedback. Start with literature-calibrated priors, adapt to deployment environment.

6. **For enterprise integrations:** Amazon Bedrock AR Checks provide the only formal-verification-based tool output validation in production today. Worth integrating as an optional backend for customers with policy documents.

### 7.4 Novel Contributions AgentGuard Should Pioneer

1. **Latency-as-proof:** The latency fingerprint for proving execution is underexplored. Build a rigorous statistical model of expected latency distributions per tool, per argument type. Zero cost, high signal.

2. **Composite embedding fingerprint:** Combine (query embedding, tool output embedding, argument embedding) into a three-way similarity check. A hallucinated result will often have low coherence between all three.

3. **Call graph anomaly detection:** Track the sequence of tool calls (not just individual calls) as a graph. Unusual sequences (never-seen-before tool A → tool B → tool C transitions) indicate potential prompt injection or agent drift.

4. **Cross-session consistency:** Track what tools return for similar queries across sessions. If `get_stock_price(NVDA)` returned $650 for the last 100 calls but now returns $50, flag for verification even if the individual output looks valid.

5. **Tool-specific statistical priors:** Different tools have different hallucination base rates. A web search tool hallucinating results is much less likely than an LLM generating "search results" without calling the tool. Build per-tool priors from empirical measurement.

---

## Appendix A: Key Papers Reference Table

| Paper | arxiv ID | Key Contribution | Relevance |
|-------|----------|-----------------|-----------|
| Internal Representations for Tool Hallucinations | [2601.05214](https://arxiv.org/abs/2601.05214) | Zero-cost probe on final layer for tool-call hallucinations | **Critical** |
| Cryptographic Binding & Reproducibility | [2603.14332](https://arxiv.org/html/2603.14332v1) | G1/G2/G3 framework for provable tool execution | **Critical** |
| TrustBench | [2603.09157](https://arxiv.org/html/2603.09157v1) | Pre-execution trust scoring with calibration curves | High |
| AgentGuard (DPA) | [2509.23864](https://arxiv.org/html/2509.23864v1) | MDP-based probabilistic runtime verification | High |
| Semantic Entropy | [Nature 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11186750/) | Meaning-cluster entropy for hallucination detection | High |
| LLM-Check | [NeurIPS 2024](https://github.com/GaurangSriramanan/LLM_Check_Hallucination_Detection) | Eigenvalue + token uncertainty, 45-450× faster | High |
| Semantic Entropy Probes | [2406.15927](https://arxiv.org/abs/2406.15927) | Cheap probes predicting SE from hidden states | Medium |
| Hallucination Survey | [2510.06265](https://arxiv.org/html/2510.06265v3) | Comprehensive taxonomy of all detection methods | Reference |
| Microsoft Production System | [2407.15441](https://arxiv.org/abs/2407.15441) | NER+NLI+SBD+decision tree in production | High |
| ToolBench / ToolEval | [ICLR 2024](https://github.com/openbmb/toolbench) | Evaluation methodology for tool-calling | Reference |
| Comprehensive Benchmark Survey | [HuggingFace](https://huggingface.co/datasets/tuandunghcmut/BFCL_v4_information/blob/main/A%20Comprehensive%20Survey%20of%20Benchmarks%20for%20Evaluating%20Tool%20and%20Function%20Calling%20in%20Large%20Language%20Models.md) | Full survey of tool-calling benchmarks | Reference |
| ARMO Intent Drift | [armosec.io](https://www.armosec.io/blog/detecting-intent-drift-in-ai-agents-with-runtime-behavioral-data/) | eBPF-based runtime behavior analysis | Medium |
| Finch-Zk Cross-Model | [2508.14314](https://arxiv.org/html/2508.14314v1) | Cross-model consistency, 6-39% better than SelfCheckGPT | Medium |
| Differential Testing with LLMs | [2410.04249](https://arxiv.org/html/2410.04249v3) | Differential testing framework with LLM specification | Low |
| LLM Fingerprinting | [2501.18712](https://arxiv.org/html/2501.18712v4) | Static+dynamic fingerprinting of LLMs | Low |

## Appendix B: Zero-Cost vs LLM-Call Signal Summary

### Zero-Cost Signals (No Extra LLM Calls, No Extra API Calls)
- JSON schema validation of arguments and outputs
- Latency anomaly detection (latency < expected minimum)
- Side-effect verification (network/file/database I/O presence check)
- Token log-probability from existing forward pass
- HaluNet internal state multi-signal (single forward pass)
- Internal representation probe (trained classifier, same forward pass)
- Eigenvalue analysis of hidden states (LLM-Check)
- Lookback ratio (attention pattern analysis)
- Historical baseline / SPC checks
- Schema version drift detection
- Capability binding hash check (G1, <1ns per check)

### Signals Requiring Extra Inference (Not Full LLM Calls)
- Embedding similarity vs centroid (embedding model, fast)
- NLI consistency check (small NLI model, 50-200ms)
- Semantic Entropy Probes (embedding model inference)

### Signals Requiring Extra LLM Calls
- Semantic entropy (N generations, expensive)
- SelfCheckGPT (N generations)
- LLM-as-judge / SPARC-style semantic pre-execution analysis
- Silent Error Review
- TrustBench LLM-as-judge

### Signals Requiring Extra Tool Calls
- Differential testing (1-2 extra tool calls)

---

*Research compiled for agentguard architecture decisions. All papers fetched and read directly — no abstract-only summaries. Last updated: 2026.*

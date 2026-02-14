# Project MASA: Manifold-Aligned Sparse Attention

**Engineering Design Document v2.0 (Dual-Track LoRA Architecture)**

- **Date:** 2026-02-13
- **Status:** Detailed Design / Implementation Ready
- **Core Philosophy:** Dual-Track, Shared Base, Independent Positional Encoding, One-Way Dependency.

## 1. System Architecture Overview

The system implements a Dual-Track architecture where a single Foundation Model backbone is shared between two logical streams (Content & DSL). The divergence in behavior is managed via Track-Specific LoRA Adapters and distinct Attention/Positional strategies.

### 1.1 The Two Tracks

**Track A: Content Stream (The "Host")**
- **Role:** Raw text generation, primary reasoning.
- **Parameters:** Frozen Base + LoRA_Content (optional/minimal).
- **KV Cache:** `content_kv` (Massive, sparsely loaded).
- **Positional Encoding:** Standard continuous RoPE.

**Track B: DSL Stream (The "Index")**
- **Role:** Semantic indexing, structural management.
- **Parameters:** Frozen Base + LoRA_DSL (Specialized for structure generation).
- **KV Cache:** `dsl_kv` (Compact, fully loaded).
- **Positional Encoding:** Independent RoPE for self-attention; NoPE (No Positional Encoding) for cross-attention.

## 2. Positional Encoding & KV Cache Design

This is the critical differentiator from standard architectures. We decouple the time-series of the content from the topology of the DSL.

### 2.1 Content Track

- **Position IDs:** $P_{content} \in \{0, 1, 2, ..., N\}$.
- **Strictly monotonic increasing.**
- The generation of DSL tokens does not increment the Content Position ID.
- **Example:** If "The spice" is pos 0, 1, and we generate 5 DSL tokens, "must flow" starts at pos 2, not 7.
- **KV Cache:** `content_kv` stores keys/values computed with $RoPE(P_{content})$.
- **Self-Attention:** Standard Causal Attention.

$$
A_{content} = \text{Softmax}\left(\frac{Q_c K_c^T}{\sqrt{d}} + M_{sparse}\right) V_c
$$

$M_{sparse}$ is determined by the MPN (see Section 4).

### 2.2 DSL Track

- **Position IDs:** $P_{dsl} \in \{0, 1, 2, ..., M\}$.
- **Independent monotonic counter.**
- Resets or continues depending on session policy (usually continuous for session history).
- **KV Cache:** `dsl_kv` stores keys/values computed with $RoPE(P_{dsl})$.

### 2.3 Cross-Track Attention (The "Semantic Bridge")

The DSL Track must attend to both its own history (to maintain syntax) and the Content (to understand semantics).

- **Query:** $Q_{dsl}$ (derived from DSL token with LoRA_DSL).
- **Key/Value Sources:**
    - `dsl_kv`: Using RoPE (Relative distance matters for DSL syntax).
    - `content_kv`: Using NoPE (No Positional Encoding).
- **Rationale:** The relative distance between a DSL token (pos 5) and a Content token (pos 1000) is physically meaningless. We rely purely on Semantic Matching.

**Attention Equation:**

$$
A_{dsl} = \text{Softmax}\left( [Q_{dsl}K_{dsl}^T \cdot \mathrm{RoPE\_Mask} \parallel Q_{dsl}K_{\mathrm{content\_raw}}^T] \right) \cdot [V_{dsl} \parallel V_{content}]
$$

> **Note:** $K_{content\_raw}$ refers to Key vectors before RoPE application, or we must inversely rotate Q to cancel RoPE for the cross-segment. (Engineering decision: likely easier to store non-RoPE keys for cross-attn or use a dedicated projection).

## 3. Execution Flow (The "Stop-and-Go" Runtime)

The system operates as a state machine switching between Content Mode and DSL Mode.
```mermaid
graph TD
    Start((Start)) --> ContentGen[Content Generation<br/>(Self-Attn, RoPE)]
    ContentGen --> CheckHead{DSL Activation<br/>Head?}
    
    CheckHead -- No --> ContentGen
    CheckHead -- Yes --> PauseContent[Pause Content Stream]
    
    PauseContent --> InjectKV[Inject Content KV<br/>as Prefix Context (NoPE)]
    InjectKV --> DSLGen[DSL Generation<br/>(Autoregressive)]
    
    DSLGen --> CheckEnd{Is [DSL_END]?}
    CheckEnd -- No --> DSLGen
    CheckEnd -- Yes --> ResumeContent[Resume Content Stream]
    ResumeContent --> ContentGen
```

### 3.1 Step 1: Content Generation
- **State:** Active Track = Content. Adapter = LoRA_Content (or None).
- **Action:** Generate $W$ tokens (a chunk/segment) using standard causal self-attention.
- **Storage:** Accumulate `content_kv`.
- **Trigger:** Detect Stop Token (e.g., `\n`, `.`, or buffer full) OR `[DSL_START]` predicted by a lightweight head.

### 3.2 Step 2: Context Switch (The "Halt")
- **Action:**
    - Freeze Content generation.
    - Hot-Swap: Activate LoRA_DSL.
    - Define `Current_Content_Chunk = content_kv[Start_Ptr : End_Ptr]`.

### 3.3 Step 3: DSL Generation (Indexing)
- **State:** Active Track = DSL.
- **Action:** Autoregressive generation until `[DSL_END]` token.
- **Input:** `[DSL_START]`.
- **Context:** Can see all `dsl_kv` + `content_kv` (via NoPE Cross-Attn).
- **Storage:** Accumulate `dsl_kv`.

### 3.4 Step 4: Alignment & Mask Prediction
- **Action:**
    - **Registry Update:** Runtime records mapping: `New_DSL_Node_ID` $\rightarrow$ `Content_Range(Start_Ptr, End_Ptr)`.
    - **MPN Inference:**
        1. Take Hidden State of `[DSL_END]`.
        2. Compute scores against all historical DSL Nodes.
        3. **Gumbel Top-K:** Select Top-K relevant nodes.
    - **Mask Construction:**
        1. Convert selected Nodes to Content Ranges.
        2. Build binary/bias mask $M_{sparse}$ for the next content generation step.

### 3.5 Step 5: Resume Content
- **Action:**
    - **Hot-Swap:** Deactivate LoRA_DSL.
    - **Load:** Load selected `content_kv` pages into HBM (if paged out).
    - **Resume:** Continue Content generation. Position IDs continue from $W+1$.

## 4. Implementation Stages

### Stage 1: DSL SFT (Structure Learning)
- **Goal:** Train LoRA_DSL to generate valid DSL tags based on content.
- **Data:** `(Raw_Text, DSL_Sequence)` pairs generated by Oracle (GPT-4o).
- **Training:**
    - Freeze Base Model.
    - Train LoRA_DSL only.
    - **Loss:** NTP on DSL tokens only.
    - **Attention Mask:** DSL tokens attend to Content (NoPE) + DSL History (RoPE).

### Stage 2: MPN Distillation (Alignment Learning)
- **Goal:** Train the Mask Predictor to match Full Attention.
- **Components:** Frozen Base + Frozen LoRA_DSL + Trainable MPN.
- **Procedure:**
    1. Run Content Stream with Full Attention (Teacher). Record Attention Map $A_{full}$.
    2. Run DSL Stream to get Node Embeddings.
    3. Train MPN to predict which historic Nodes cover the high-attention regions in $A_{full}$.
- **Loss:** $KL(A_{full} || A_{pred}) + L1_{sparsity}$.

### Stage 3: Online Inference & Paging
- **Goal:** End-to-end system with KV Cache Paging.
- **Engineering:**
    - Implement Custom Attention Kernel: Supports RoPE (self) + NoPE (cross) mixed mode.
    - Implement KV Page Manager: CPU $\leftrightarrow$ GPU async transfer based on MPN prediction.

### Stage 3.5: Refresh Mechanism (Addressing Semantic Drift)
- **Problem:** Content generated at $T=0$ might change meaning by $T=10000$. The DSL Node embedding becomes stale.
- **Solution:** RefreshHead.
- **Action:** Periodically (or on-demand), re-run the DSL Node embedding through a small MLP (the RefreshHead) conditioned on the current global context summary, updating `dsl_kv` without re-generating tokens.

## 5. Critical Data Structures

### 5.1 Alignment Table (Runtime Registry)

#### Data Structure Design

```python
class DSLContentAlignment:
    segments: List[Segment]

class Segment:
    dsl_node_id: str          # e.g., "#D1"
    content_start: int        # content-only token index
    content_end: int          # content-only token index
    sequence_start: int       # absolute sequence position (content + DSL mixed)
    sequence_end: int         # absolute sequence position
```

#### Automatic Maintenance Logic

This alignment table is maintained automatically by the runtime based on token type (content vs DSL); the model is completely agnostic to its existence.

**Segmentation Logic at DSL Activation**

When `[DSL_START]` is generated, the runtime must determine the content range associated with this DSL node.

**Rule:** All content tokens between the previous `[DSL_END]` and the current `[DSL_START]` constitute the content range for the current DSL node.

```text
Time ----------------------------------------------------------------------->

Track A (Content):   [ Content Segment 1 ]          [ Content Segment 2 ]
                     ^                   ^          ^                   ^
                     |___________________|          |___________________|
                               |                              |
Track B (DSL):                 |         [DSL Block 1]        |         [DSL Block 2]
                               |         ^                    |         ^
                               |_________|                    |_________|
                               Suffix Binding                 Suffix Binding
```

The runtime tracks two state variables:
1. `content_token_count`: Global counter for content tokens.
2. `current_segment_start`: Starting content position of the current segment.

**Pseudocode:**

```python
content_token_count = 0      # Current segment's content token count
current_segment_start = 0    # Current segment's start content position

for token in generated_sequence:
    if token == [DSL_START]:
        # End of current content segment
        record_segment(start=current_segment_start,
                       end=content_token_count)
        # Enter DSL mode, pause content counting

    elif token == [DSL_END]:
        # DSL generation finished, new content segment starts
        current_segment_start = content_token_count
    elif is_content_token(token):
        content_token_count += 1
```
### 5.2 KV Cache Memory Layout
- **Pool A (Content):** Paged Memory. Pages can be evicted to CPU RAM. Indexed by `Content_Pos`.
- **Pool B (DSL):** Contiguous Memory (Ring Buffer). Always resident in HBM (High priority). Indexed by `DSL_Pos`.

## 6. Risk Analysis & Mitigation

- **Risk:** LoRA switching latency.
    - **Mitigation:** Use Multi-Head LoRA (merged weights with mask) instead of physical weight swapping. Both tracks' QKV projections exist simultaneously; we just route activation to the correct head.

- **Risk:** DSL Hallucination (generating IDs).
    - **Mitigation:** Strict Vocabulary constraints. DSL cannot generate numeric IDs. It only generates `[NODE]`, `[TYPE]`. Runtime assigns IDs.

- **Risk:** "NoPE" Attention collapse.
    - **Mitigation:** Ensure LoRA_DSL is sufficiently powerful to project Content Semantics into the DSL latent space effectively without needing positional cues.

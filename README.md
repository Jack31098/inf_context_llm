# Infinite-Context Memory OS (ICMOS)

> 💡 Research prototype: building an **infinite-context memory system** for an 8B LLM on a **small 4× MI100 node + large host RAM**.

This repo is an experiment in **treating KV cache as a first-class memory system**, not just a byproduct of attention.

The long-term goal is to make a mid-size model (e.g. Qwen3-8B) able to:

- Hold **100K–1M+ tokens of dialogue/history** on a **4× MI100 box + host RAM**  
- Decide **what to remember / what to forget** at the **token level**  
- Organize past context into **semantic memory blocks**  
- Retrieve the right blocks later via **neural, differentiable retrieval** instead of brute-force scanning

This is **not** another “we trained a 1M context model on a huge TPU/GPUs cluster” project.  
This is a **memory OS** on top of an existing LLM, targeting a **small 4-GPU node** rather than a hyperscale setup.

---

## Why this project exists

Most “long context” solutions today fall into one of these buckets:

1. **Bigger context windows**  
   - Train or finetune a model to 128K / 1M context  
   - Still essentially a **sliding window** with smarter attention  
   - Memory is **implicit**: the model just hopes everything it needs is still in the window

2. **RAG / Vector DB**  
   - Documents / chunks go into a vector store  
   - At query time, you embed → retrieve → stuff into prompt  
   - Good for static knowledge, but:
     - Long **conversational history** and its latent state is hard to manage
     - Retrieval is **hard, discrete**, not integrated into the model’s inference loop

3. **Host KV / StreamingLLM**  
   - Offload old KV to CPU, keep a recent window on GPU  
   - Maybe pin a few “anchor tokens” at the beginning  
   - Still mostly **position-based eviction** with light heuristics

These approaches are very useful, but they don’t really answer:

> “If an LLM had a *brain-like, layered memory system* on a small 4-GPU node,  
> what would that system look like?”

This repo is my attempt to **prototype** such a system, end-to-end, in a way that is:

- **Compute-realistic** (4× MI100 + large host RAM)
- **Modular** (token-level, block-level, graph-level)
- **Differentiable** where it matters (so the LLM can *learn how to recall*)

---

## What’s different from existing work?

This project is **not** introducing one new trick; it’s about **how the pieces fit together**:

### 1. Token-level learned “forgetting” (Trimmer)

Instead of:

- “Keep first 4 tokens + a rolling window”
- Or “evict tokens purely based on position”

We train a **Trimmer module** that:

- Looks at **summary token hidden state + all previous token hidden states**
- Uses **Gumbel-Softmax + soft-to-hard Top-K** to decide which tokens’ KV to keep
- Is trained with **NTP / logits KL loss** so that:
  - Under a fixed KV budget, it learns to **keep what actually matters**
- Optionally, a light **LoRA** adapts the base LLM to “living with” trimmed KV + injected prefixes

This is **token-level, learned KV eviction**, not just sliding windows.

---

### 2. Interpretation-driven dual-phase encoding for block memory

Instead of:

- Only storing raw KV
- Or only storing bag-of-words text summaries
- Or training retrieval vectors directly

We adopt an aggressive **“interpret first, distill later”** dual-phase architecture:

1. **Phase 1 – Semantic Compressor (The Cornerstone)**  
   Train a Q-Former to produce **reasoning-aware Detail Tokens** `D` for each 4K block, using the LLM’s own **causal reasoning** as a regularizer.

2. **Phase 2 – Index Distiller (Alignment & Indexing)**  
   On top of frozen Detail Tokens, train a lightweight distiller + query projector to map into a **metric retrieval space** suitable for ANN / HNSW, without destroying the semantics learned in Phase 1.

This way we get:

- A **Tier-2 payload** (`D`): high-fidelity, interpretable, “knows why this block matters”.
- A **Tier-1 index** (`H`): compact, ANN-friendly embeddings for fast retrieval.

See below for details.

---

### 2.1 Phase 1: Semantic Compressor (Detail Tokens)

**Role:** Semantic backbone of the memory system.  
**Goal:** Use the frozen LLM’s reasoning ability as a **regularizer**, to force a Q-Former to compress a 4K-token block into a set of **reasoning-aware Detail Tokens** `D` (Tier-2 payload).  
**Key design:** Prevent information loss and shortcut learning during compression.

**Model structure**

- **Writer:** Q-Former (BERT/ViT-style) + MLP  
- **Output:** `N = 32` **Detail Tokens** (`D`) per block  
  - Not just textual summaries  
  - They are **carriers of logical information** (what version? which entity? which condition?)

**Training strategy – Explanation SFT**

For each `(Block, Query)` pair:

1. Encode the block with the Q-Former to obtain Detail Tokens `D`.
2. Inject `D` as a **soft prompt** into the frozen LLM.
3. Ask the LLM to generate a **Retrieval Rationale**:

   - **Type A (Positive):**  
     > “I retrieve this because it defines `var_x` needed by the query.”  
     Forces `D` to encode semantic associations.

   - **Type B (Negative – Anti-Shortcut):**  
     > “I do **not** retrieve this because the query asks for v2.0 features,  
     > but this block describes v1.0.”  

**Curriculum (crucial):**

- At least **50% hard negatives**:
  - Blocks that are *semantically similar* but *logically incompatible* with the query
- To generate good rejection rationales, `D` must preserve **sharp details** (e.g., IDs, versions), not just fuzzy “topic vectors”.

**Control tokens**

- Jointly train special tokens:
  - `<MEM_START>`
  - `<MEM_END>`
- So the LLM learns **semantic isolation** of the memory segment injected via `D`.

**Result**

- After Phase 1, the Q-Former is **frozen**.  
- Detail Tokens `D` form a **high-fidelity, causally-informed intermediate representation** of each memory block.
- `D` are stored in **Host RAM** as the Tier-2 payload for that block.

---

### 2.2 Phase 2: Index Distiller (Retrieval Alignment)

**Role:** Retrieval adapter for the memory system.  
**Goal:** On top of frozen `D`, train a light-weight distiller to resolve the tension between **semantic completeness** and **retrieval geometry**, producing a Tier-1 **index embedding** `H`.

**Asymmetric two-tower structure**

- **Writer tower (Index Distiller):**
  - **Input:** frozen Detail Tokens `D` from Phase 1
  - **Architecture:** Tiny Transformer (2–4 layers)
  - **Trick – Random Token Dropout (input masking):**
    - During training, randomly mask out **30–50%** of tokens in `D`
    - Forces the distiller to use **global** information, not overfit to one “golden token”
  - **Output:** single **Index Embedding** `H` for the block

- **Reader tower (Query Projector):**
  - **Input:** main LLM hidden state (e.g., when generating a `<RETRIEVE>` token or at a special retrieval step)
  - **Architecture:** MLP / Mixer
  - **Output:** single **Query Embedding** `Q`

**Training – Contrastive alignment**

- **Task:** map the **interpretive space** (`D`) to a **metric retrieval space** (`H`, `Q`) suitable for ANN / DNII.
- **Loss:**  
  - **InfoNCE** (listwise contrastive loss)  
  - **No BCE loss** – we want a **well-shaped embedding topology**, not just a “yes/no” classifier.

- **Hard negative mining:**
  - In each batch, mix blocks that:
    - Look **semantically close** (same topic / entity)
    - But are **logically incompatible** with the query (different version, wrong ID, wrong condition)
  - These hard negatives are constructed using Phase-1 signals (rationales, tags).
  - This forces the distiller to preserve the **sharp features** encoded in `D`.

**Result**

- `H` lives in a retrieval-friendly space (HNSW / DNII), while `D` remains a rich, interpretable representation living in Host RAM.
- At runtime, retrieval is:
  - `Q` → ANN over `H` → block IDs → load `D` for those blocks.

---

### 2.3 Inference workflow: decoupling speed & understanding

At inference time, the system reuses both phases to decouple **fast retrieval** from **deep understanding**.

**Write path (indexing)**

1. `Block  ->(Frozen Q-Former)->  Detail Tokens D   ->(Save)-> Host RAM (Tier 2)`
2. `Detail Tokens D  ->(Tiny Transformer)->  Index Embedding H  ->(Save)-> HNSW / DNII (Tier 1)`

- Phase 1 guarantees the **quality & faithfulness** of `D`.
- Phase 2 guarantees the **retrieval directionality** of `H`.

**Read path (retrieval)**

1. `LLM context  ->(Projector)->  Query embedding Q`
2. **Search:**  
   `Q` vs HNSW / DNII over `H` → Top-K block IDs
3. **Inject:**  
   Load corresponding Detail Tokens `D` from Host RAM and inject them as a **soft prompt / memory segment** into the current LLM context.
4. **Generate:**  
   The LLM generates the final answer, conditioned on these **reasoning-aware Detail Tokens**.

---

## Very high-level roadmap

This repo is being built in **phases**, with the explicit goal that **each phase is independently useful**.

### ✅ Phase 0: Plumbing & Baseline

- [ ] Hook Qwen3-8B (or similar 7–8B model) into a **custom inference loop** on a 4× MI100 node
- [ ] Implement:
  - Basic **StreamingLLM-style sliding window + first-4 anchors** (baseline)
  - Simple **Host KV offload** (CPU-side KV store, GPU-side rolling window)
- [ ] Construct small **long-context toy benchmarks**:
  - Needle-in-a-Haystack / passkey tasks (64K–128K)
  - Long-dialogue QA / summarization

> Output: a **reproducible baseline** for long-context on a 4× MI100 machine.

---

### 🚧 Phase 1: Token-level Trimmer (Tech Report A)

- [ ] Design & implement **Trimmer**:
  - Input: summary token hidden + block hidden states
  - Output: importance scores → soft/hard Top-K mask
- [ ] Train Trimmer:
  - Fixed KV budget K (function of block size)
  - NTP + logits KL loss under masking  
  - Base LLM frozen
- [ ] Train light **LoRA** to adapt LLM to:
  - KV trimming
  - (Optionally) local KV prefix injection
- [ ] Evaluate vs baselines:
  - PPL / QA / retrieval success
  - Under equal KV budget

> Output: **Tech Report A – “Learning to Forget: Token-level KV Trimming for Host-Resident Long-Context Inference”**

---

### 🚧 Phase 2: Block-level Detail Tokens & Index Distiller (Tech Report B)

- [ ] Train **Q-Former** to produce **Detail Tokens** `D`:
  - Input: block hidden states
  - Bottleneck: fixed number of latent queries (e.g. 32)
  - Training: **Explanation SFT** + auxiliary classification / contrastive losses, so `D` preserves **task-relevant logical features**
- [ ] Train **Index Distiller** on top of `D`:
  - Tiny Transformer + random token dropout
  - Output a single index embedding `H` per block
- [ ] Build **DNII** on top of `H`:
  - Online clustering / EMA centroids
  - Posting lists, splitting/merging overloaded centroids
  - Two-hop retrieval (Query → Centroids → Blocks)
- [ ] Train **Query Projector**:
  - Input: retrieval token hidden state (or special retrieval step)
  - Loss: **InfoNCE** (positive block vs hard negatives)
- [ ] Evaluate:
  - Block recall@K
  - QA accuracy with block-level retrieval
  - Latency vs number of blocks

> Output: **Tech Report B – “Interpretation-Driven Dual-Phase Encoding and DNII for Long-Term LLM Memory”**

---

### 🔭 Phase 3: System integration & cognitive view

- [ ] Wire Phase 1 + Phase 2 into a single **Memory OS**:
  - Token-level Trimmer decides what survives in local KV
  - Past blocks get summarized → `D` stored as host KV + DNII index `H`
  - Retrieval tokens query DNII → fetch blocks back via `D` into KV
- [ ] Add **tools & visualizations**:
  - Memory timeline
  - Block clustering view (DNII heads, centroids, posting lists)
  - Retrieval traces (“why did we recall this block?”)
- [ ] Write an **overview blog post**:
  - The story, design, and lessons learned
  - Demo scenarios

> Output: a **working demo** + a blog:  
> *“Building an Infinite-Context Memory OS for 8B LLMs on a 4× MI100 Node”*

---

## Status

This is **early-stage research code**:

- Expect broken pieces, TODOs, and experiments
- The focus is on **clarity of ideas + modular design**, not production polish
- Contributions (design discussion, issues, PRs) are very welcome

If you are interested in:

- Long-context inference **beyond** just bigger windows  
- Treating KV as **structured, trainable memory**  
- Or just want to see how far a small **4× MI100** node can go

…then this project is for you 🙂

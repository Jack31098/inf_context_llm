# Infinite-Context Memory OS (ICMOS)

> 💡 Research prototype: building an **infinite-context memory system** for 8B LLMs on a single consumer GPU + large host RAM.

This repo is an experiment in **treating KV cache as a first-class memory system**, not just a byproduct of attention.

The long-term goal is to make a mid-size model (e.g. Qwen3-8B) able to:

- Hold **100K–1M+ tokens of dialogue/history** on **one GPU (e.g. 3090/4090)** + host RAM  
- Decide **what to remember / what to forget** at the **token level**  
- Organize past context into **semantic memory blocks**  
- Retrieve the right blocks later via **neural, differentiable retrieval** instead of brute-force scanning

This is **not** another “we trained a 1M context model on a big cluster” project.  
This is a **memory OS** on top of an existing LLM.

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

> “If an LLM had a *brain-like, layered memory system* on a single GPU,  
> what would that system look like?”

This repo is my attempt to **prototype** such a system, end-to-end, in a way that is:

- **Compute-realistic** (single consumer GPU + host RAM)
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

### 2. Block-level semantic memory via Q-Former

Instead of:

- Only storing KV
- Or only storing text summaries

We:

- Cut the past into **Memory Blocks**  
- For each block, use a **Q-Former** to produce a **small set of summary embeddings** (information bottleneck)
- These embeddings are **not** used as online prefix-KV; they are:
  - **Block tags / handlers** for later retrieval
  - A semantic “name” for a huge KV block stored on host

All layers’ KV for that block can be stored off-GPU and later fetched **as a unit** via that tag.

---

### 3. Differentiable Neural Inverted Index (DNII) for blocks

Instead of:

- Pure HNSW / FAISS search over block embeddings (fast but non-differentiable)
- Or a giant static graph RAG that’s expensive to maintain

We design a **DNII**:

- Many **Index Heads**, each with:
  - A learnable **centroid matrix** `C` (virtual nodes / topics)
  - **Posting lists** `L[i]` storing block IDs attached to centroid `i`
- **Write path** (online clustering):
  - For a new block summary `s_new`:
    - Find nearest centroid(s)
    - Append block ID to corresponding list(s)
    - Update centroid via EMA
- **Read path** (two-hop, soft-to-hard):
  - Hop 1: Query `q` → Softmax / STE over centroids → select Top-M virtual nodes
  - Hop 2: Scan only their posting lists, re-score candidate blocks → Top-K blocks

Crucially:

- The **query projector** is trained with **InfoNCE**  
- The routing is **soft / STE**, so retrieval is **end-to-end trainable** w.r.t. the LLM’s retrieval token hidden state

This sits in between “vector DB” and “DSI-style neural index”:  
fast, dynamic, and **purpose-built for LLM memory blocks**.

---

## Very high-level roadmap

This repo is being built in **phases**, with the explicit goal that **each phase is independently useful**.

### ✅ Phase 0: Plumbing & Baseline

- [ ] Hook Qwen3-8B (or similar 7–8B model) into a **custom inference loop**
- [ ] Implement:
  - Basic **StreamingLLM-style sliding window + first-4 anchors** (baseline)
  - Simple **Host KV offload** (CPU-side KV store, GPU-side rolling window)
- [ ] Construct small **long-context toy benchmarks**:
  - Needle-in-a-Haystack / passkey tasks (64K–128K)
  - Long-dialogue QA / summarization

> Output: a **reproducible baseline** for long-context on a single GPU.

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

### 🚧 Phase 2: Block-level Summaries & DNII (Tech Report B)

- [ ] Train **Q-Former** to produce block summary embeddings:
  - Input: block hidden states
  - Bottleneck: fixed number of latent queries
  - Training: decode textual summary via a small decoder LoRA (to force non-trivial compression)
- [ ] Build **DNII** on top of block summaries:
  - Online clustering / EMA centroids
  - Posting lists, splitting/merging overloaded centroids
  - Two-hop retrieval (Query → Centroids → Blocks)
- [ ] Train **Query Projector**:
  - Input: retrieval token hidden state
  - Loss: InfoNCE (positive block vs negatives)
- [ ] Evaluate:
  - Block recall@K
  - QA accuracy with block-level retrieval
  - Latency vs number of blocks

> Output: **Tech Report B – “Differentiable Neural Inverted Index for Long-Term LLM Memory”**

---

### 🔭 Phase 3: System integration & cognitive view

- [ ] Wire Phase 1 + Phase 2 into a single **Memory OS**:
  - Token-level trimmer decides what survives in local KV
  - Past blocks get summarized → stored as host KV + DNII entry
  - Retrieval tokens query DNII → fetch blocks back into KV
- [ ] Add **tools & visualizations**:
  - Memory timeline
  - Block clustering view
  - Retrieval traces (“why did we recall this block?”)
- [ ] Write an **overview blog post**:
  - The story, design, and lessons learned
  - Demo scenarios

> Output: a **working demo** + a blog:  
> *“Building an Infinite-Context Memory OS for 8B LLMs on a Single GPU”*

---

## Status

This is **early-stage research code**:

- Expect broken pieces, TODOs, and experiments
- The focus is on **clarity of ideas + modular design**, not production polish
- Contributions (design discussion, issues, PRs) are very welcome

If you are interested in:

- Long-context inference **beyond** just bigger windows  
- Treating KV as **structured, trainable memory**  
- Or just want to see how far a single GPU can go

…then this project is for you 🙂

---

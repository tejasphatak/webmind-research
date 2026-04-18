# Mindmap — How Everything Connects

> **Note:** This is a raw dump of ideas, not a scientific paper. It's a map of how the pieces *might* fit together. Some connections are proven. Some are speculative. Some might be wrong. The point is to see the whole picture and ask: does this hold up?

## The Big Picture

```mermaid
graph TB
    subgraph UNDERSTAND ["1. UNDERSTAND (Language)"]
        E[Sentence Transformer<br/>22M params<br/>Turns text → meaning vectors]
    end

    subgraph THINK ["2. THINK (Reasoning)"]
        CL[Convergence Loop<br/>Fixed-point iteration<br/>in embedding space]
        BR[Bi-embedding Re-ranking<br/>Q-sim + A-sim scoring]
    end

    subgraph REMEMBER ["3. REMEMBER (Knowledge)"]
        KB[Growing Knowledge Base<br/>Q&A pairs + embeddings<br/>Self-evolving]
        WS[Web Search Fallback<br/>Wikipedia + DuckDuckGo<br/>Source agreement]
    end

    subgraph DISTRIBUTE ["4. DISTRIBUTE (Scale)"]
        SY[Synapse<br/>WebGPU + WebRTC<br/>Split across devices]
        CP[Carrier-Payload<br/>36-70x compression<br/>Activation transfer]
    end

    subgraph PROTECT ["5. PROTECT (Safety)"]
        ET[Ethics Through Data<br/>High-weight KB pairs<br/>PII sanitization]
    end

    E --> CL
    CL --> BR
    BR --> KB
    KB --> WS
    WS --> KB
    KB --> SY
    SY --> CP
    ET --> KB

    style UNDERSTAND fill:#e1f5fe
    style THINK fill:#f3e5f5
    style REMEMBER fill:#e8f5e9
    style DISTRIBUTE fill:#fff3e0
    style PROTECT fill:#ffebee
```

## The Thesis In One Sentence

**An LLM is a lossy database compressed into weights. We built a lossless one that grows.**

## How The Papers Connect

```mermaid
graph LR
    P1[Self-Evolving Retrieval<br/>THE CORE PAPER<br/>0.7% → 25.3% on held-out data] 

    P2[SFCA<br/>Credit assignment for<br/>multi-agent cognition]

    P3[Activation Speculation<br/>is Dead<br/>Negative result - bootstrap<br/>deadlock in speculative decode]

    P4[Carrier-Payload<br/>36-70x activation<br/>compression for wire transfer]

    P5[MoeMoe<br/>Mixture-of-experts<br/>resilience under node failure]

    P6[Synapse v2<br/>Distributed specialists<br/>across browser devices]

    P7[SAQT Ethics<br/>Safety through data<br/>not code]

    P1 --> |"self-evolution<br/>needs distribution"| P6
    P6 --> |"activations need<br/>compression"| P4
    P6 --> |"nodes fail,<br/>need resilience"| P5
    P1 --> |"who gets credit<br/>for the answer?"| P2
    P1 --> |"safety can't be<br/>hardcoded"| P7
    P3 --> |"speculation doesn't work,<br/>retrieval does"| P1

    style P1 fill:#e8f5e9,stroke:#2e7d32,stroke-width:3px
    style P3 fill:#ffebee
```

### What each paper actually says:

**1. Self-Evolving Retrieval** (the main paper)
- All you need: encoder (understand) + convergence loop (think) + database (remember)
- 175B params is overkill — 22M for understanding, rest goes in a database
- System teaches itself: 0.7% → 25.3% on held-out data, zero human intervention
- Runs in browser at 214MB
- *Status: benchmarked, results proven, paper drafted*

**2. SFCA — Shapley-Fair Credit Assignment**
- When multiple sources contribute to an answer, who gets credit?
- Relevant because: self-evolving system learns from KB + web + user feedback
- Each source's contribution should be tracked and credited
- *Status: pre-registered on Zenodo, code working*

**3. Activation Speculation is Dead**
- Tried speculative decoding for Synapse distributed inference
- Found a bootstrap deadlock: warmup needs verify(), verify needs pending, pending needs enabled
- Negative result — but it pushed us toward retrieval instead of generation
- *Status: published as negative result*

**4. Carrier-Payload Compression**
- 36-70x compression on activation tensors using PCA/VQ
- A 384-dim embedding (1.5KB) → ~40 bytes
- Directly relevant: if we distribute the KB across devices, embeddings need compression for transfer
- *Status: math proven, implementation pending*

**5. MoeMoe Resilience**
- What happens when a node in the distributed mesh goes down?
- Mixture-of-experts approach: route around failure
- Relevant for: school scenario with 30 tablets, kids disconnect randomly
- *Status: preliminary results on Zenodo*

**6. Synapse v2 — Distributed Specialists**
- Each device in the mesh specializes in a subset of the KB
- Query broadcasts → each device searches its shard → merge results
- This is how you scale from 300K pairs to 30M without any single device holding everything
- *Status: architecture designed, not yet validated for retrieval (only for inference)*

**7. SAQT Ethics**
- Safety taught through data, not hardcoded rules
- High-weight ethics pairs in the KB act as gravitational attractors
- PII sanitization strips personal data from learned content
- *Status: implemented, adversarial eval shows 48% — needs improvement*

## The Evolution of the Idea

```mermaid
timeline
    title How we got here (April 14-18, 2026)
    
    section Day 1 (Apr 14)
        Synapse distributed LLM inference : WebGPU across phones
        SFCA credit assignment : Who contributed what?
        
    section Day 2 (Apr 15)  
        First live mobile inference : 0.82 tok/s on 2 phones
        Speculation is dead : Bootstrap deadlock found
        Offline schools idea : 30 tablets, no internet
        
    section Day 3 (Apr 16)
        Carrier-Payload compression : 36-70x on activations
        Tim Dettmers validation : "No deep prior art"
        
    section Day 4 (Apr 17)
        MoeMoe resilience : Handle node failures
        Async stale summaries : Distributed inference optimization
        
    section Day 5 (Apr 18)
        SAQT browser engine : 305K Q&A pairs in browser
        Self-evolution proven : 0.7% → 25.3% on held-out
        New architecture thesis : Understand + Think + Remember
        Paper published : webmind-research public
```

## Connections That Might Work

### 1. Self-Evolving Retrieval + Synapse Distribution
- Shard the KB across 30 tablets
- Each holds 10K pairs (~7MB)
- Query broadcasts via WebRTC
- Each device searches its shard in parallel
- Carrier-Payload compresses embedding vectors for transfer
- **Question:** Does sharded retrieval lose quality vs centralized?

### 2. Convergence Loop + Distributed Search
- The convergence loop iterates: search → check → refine
- What if each iteration hits a DIFFERENT shard?
- Hop 1: device A finds a partial answer
- Hop 2: device B refines it with its shard
- Hop 3: device C converges
- **Question:** Does distributed convergence actually converge? Or does shard partitioning break it?

### 3. Self-Evolving Encoder + Carrier-Payload
- The encoder fine-tunes weekly from retrieval feedback
- Carrier-Payload compresses embeddings during transfer
- If the encoder shifts, the compression basis vectors shift too
- **Question:** Do we need to retrain compression after encoder fine-tune?

### 4. Ethics Through Data + Self-Evolution
- The system learns from the web automatically
- Ethics pairs try to block harmful content
- But the adversarial eval shows 48% pass rate
- **Question:** Can the ethics system evolve too? Can it learn new safety boundaries from user rejections?

### 5. SFCA + Self-Evolution Feedback
- When an answer is accepted, who gets credit?
- The original KB pair? The web search? The convergence loop?
- SFCA could attribute credit → boost the right sources
- **Question:** Is Shapley value computation fast enough for real-time retrieval?

## What's Speculative (Unproven)

- "Third architecture for AI" — reviewers say overclaimed
- Distributed retrieval across phones — P2P never connected in testing
- Self-evolving encoder — planned but not implemented
- Ethics through data alone — 48% adversarial pass rate is not enough
- School scenario — requires offline infra that doesn't exist yet
- Scale to 30M pairs — tested only at 300K

## What's Proven

- Self-evolution works: 0.7% → 25.3% on held-out data (HotPotQA: 0→72%)
- Bi-embedding re-ranking catches false positives (PCA→cupcakes bug)
- Convergence loop stabilizes answers (embedding delta < epsilon)
- Browser-native at 214MB — demo live at webmind.sh
- Multilingual: Hindi/Marathi at 92-97% similarity
- Web search + learn loop grows KB automatically (305K → 306K+ in one session)
- All code verified against claims (10/10 features have working code)

## The One Question

If you strip away all the grand claims and just look at what we built:

**A system that understands questions, searches a database, thinks iteratively, learns from mistakes, and gets better with every query — all in 214MB on a phone.**

Is that useful? Is that worth scaling? Is that the future, or a dead end?

We don't know yet. But the math works, the code runs, and the numbers go up.

---

*This mindmap will evolve as the research does. It's a living document.*

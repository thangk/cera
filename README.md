# CERA: Context-Engineered Reviews Architecture

<p align="center">
  <img src="assets/cera-logo.png" alt="CERA Logo" width="300">
</p>

<p align="center">
  <b>A training-free framework for generating realistic, controllable synthetic review datasets for Aspect-Based Sentiment Analysis (ABSA).</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/status-in%20development-yellow" alt="Status">
  <img src="https://img.shields.io/badge/python-3.10+-blue" alt="Python">
  <img src="https://img.shields.io/badge/license-research%20only-lightgrey" alt="License">
</p>

---

> **Note:** This repository is currently under active development as part of an MSc thesis at the University of Windsor. Code will be released upon completion.

---

## Quick Start (Docker)

### Prerequisites
- Docker and Docker Compose
- Node.js (for Convex deployment)

### First-Time Setup

1. **Clone and create `.env`**
   ```bash
   git clone https://github.com/thangk/cera.git
   cd cera
   cp .env.example .env
   ```

2. **Start Convex backend**
   ```bash
   docker-compose up convex -d
   ```

3. **Generate admin key** (wait ~10 seconds for Convex to start)
   ```bash
   docker exec convex ./generate_admin_key.sh
   ```
   Copy the output (starts with `convex_...`)

4. **Add key to `.env`**
   ```
   CONVEX_ADMIN_KEY=convex_your_key_here
   ```

5. **Deploy Convex schema**
   ```bash
   cd gui
   npm install
   npx convex deploy --url http://localhost:3210 --admin-key YOUR_KEY
   cd ..
   ```

6. **Start all services**
   ```bash
   docker-compose up --build
   ```

### Access Points
| Service | URL |
|---------|-----|
| Web GUI | http://localhost:3001 |
| Python API | http://localhost:8000 |
| Convex Dashboard | http://localhost:6791 |

### Optional: Add OpenRouter API Key
For real LLM generation (instead of placeholder mode), add your [OpenRouter API key](https://openrouter.ai/keys) to `.env`:
```
OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

### Optional: GPU Acceleration (MDQA Metrics)

GPU acceleration significantly speeds up semantic metrics (BERTScore, MoverScore) during evaluation.

**Prerequisites:**
- NVIDIA GPU with CUDA support
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed

**1. Verify GPU access in Docker:**
```bash
docker run --rm --gpus all nvidia/cuda:12.1.1-runtime-ubuntu22.04 nvidia-smi
```

If this fails, install NVIDIA Container Toolkit:
```bash
# Ubuntu/Debian (including WSL2)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

**2. Rebuild the CLI container with GPU support:**
```bash
docker-compose build cli
docker-compose up -d
```

**3. Verify GPU is detected:**
```bash
docker exec cli python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

> **Note:** The GPU-enabled image is larger (~5GB vs ~1GB). To use CPU-only, comment out the `deploy` section in `docker-compose.yml` and change the base image in `cli/Dockerfile` to `python:3.11-slim`.

---

## Overview

CERA addresses critical challenges in ABSA research:

- **Data scarcity** - Benchmark datasets like SemEval contain only ~3K sentences
- **Class imbalance** - Real reviews skew ~65% positive, hurting minority class performance
- **Domain sparsity** - Niche domains lack sufficient annotated data

Unlike existing approaches that require model fine-tuning, CERA generates high-quality synthetic ABSA data using only **prompt engineering** and **multi-agent verification**.

---

## Key Features

| Component | Description |
|-----------|-------------|
| **Subject Intelligence Layer (SIL)** | Agentic web search for factual grounding - reduces hallucination |
| **Multi-Agent Verification (MAV)** | Cross-model consensus (2/3 majority voting) for fact verification |
| **Authenticity Modeling Layer (AML)** | Configurable polarity control with diversity enforcement and authentic writing patterns |
| **Negative Example Buffer (NEB)** | Rolling buffer of previously generated reviews injected as "what to avoid" — enforces cross-batch diversity |
| **Opening Directives** | Per-review assigned opening strategies ensuring within-batch structural diversity (15 distinct patterns) |
| **Authentic Imperfections** | Prompt-driven writing imperfections: capitalization variation, run-on sentences, informal grammar, regional colloquialisms |
| **Multi-Dimensional Quality Assessment (MDQA)** | Comprehensive evaluation: lexical, semantic, and diversity metrics |

---

## Pipeline Architecture

<p align="center">
  <img src="assets/cera-pipeline.jpg" alt="CERA Pipeline Architecture">
</p>

---

## Authenticity & Diversity Mechanisms

CERA employs several complementary mechanisms to ensure generated reviews are diverse and linguistically authentic, avoiding the "robotic uniformity" common in LLM-generated text.

### Negative Example Buffer (NEB)

Reviews are generated in sequential batches of `request_size`. After each batch completes, its reviews are added to a rolling FIFO buffer. Subsequent batches receive these previous reviews in their system prompt as explicit negative examples — the LLM is instructed to produce text that differs in opening phrases, vocabulary, writing style, structure, and specific details.

```
Batch 1 → generates reviews [1..N]    (no NEB context)
Batch 2 → generates reviews [N+1..2N] (NEB contains batch 1 reviews)
Batch 3 → generates reviews [2N+1..3N](NEB contains batch 1+2 reviews)
...
```

The buffer depth is configurable via `neb_depth` (default: 2). Buffer size = `neb_depth × request_size`, with FIFO eviction when full.

### Opening Directives

Since all reviews within a batch are generated in parallel (sharing the same NEB snapshot), NEB alone cannot enforce within-batch diversity. Opening Directives solve this: each review in a batch is randomly assigned a unique opening strategy from a pool of 15 patterns:

- Specific product detail or measurement
- Mid-thought continuation ("So I finally...", "Three weeks in...")
- Rhetorical question
- Raw emotional reaction
- Acquisition context (when/where/how purchased)
- Comparison to competitor or previous experience
- Casual filler word ("Ok so...", "Man,", "Look,")
- Direct complaint or frustration
- Enthusiastic praise or recommendation
- Warning or caveat to other buyers
- Direct reader address ("If you're looking for...")
- Time reference ("After two months...")
- Contradictory/nuanced take ("I wanted to love this but...")
- Factual usage statement
- Story or anecdote

Within a batch, `random.sample()` guarantees no duplicate directives (when batch size ≤ 15). This is especially critical for the first batch where NEB is empty.

### Capitalization Style Directives

LLMs exhibit a strong bias toward grammatically correct capitalization — even when instructed generally to "vary capitalization," generated reviews almost universally start every sentence with a capital letter. This is a detectable artifact: real product reviews on platforms like Amazon and Reddit show significant capitalization inconsistency, with many users writing entirely in lowercase or mixing styles.

To overcome this, CERA assigns each review a mandatory **capitalization style directive** using weighted random sampling, following the same per-review injection approach as Opening Directives:

| Style | Weight | Description |
|-------|--------|-------------|
| **Standard** | 55% | Proper capitalization throughout |
| **Mostly lowercase** | 20% | Lowercase sentence starts, lowercase "i", lowercase brands |
| **Casual/mixed** | 15% | Inconsistent — some sentences capitalized, others not |
| **Emphasis caps** | 10% | Normal capitalization with occasional ALL CAPS for emphasis |

The directive is injected as a mandatory instruction (`"You MUST follow this capitalization style"`) rather than a soft guideline, which we found necessary because LLMs reliably ignore soft capitalization suggestions. The weighted distribution approximates patterns observed in real user-generated content, where the majority of reviews do use proper capitalization but a meaningful minority (roughly 35–45%) deviate.

### Authentic Imperfections

The AML system prompt also instructs the LLM to introduce natural writing imperfections:

| Category | Examples |
|----------|----------|
| **Grammar imperfections** | Run-on sentences, sentence fragments, "me and my wife" vs "my wife and I" |
| **Informal punctuation** | Extra periods..., multiple exclamation marks!!, missing commas |
| **Measurement inconsistency** | "16GB" vs "16 gigs" vs "16gb", "$300" vs "300 bucks" |
| **Brand name variation** | "MacBook Pro" vs "macbook pro" vs "MBP" vs "mbp" |
| **Regional colloquialisms** | US: "awesome", "kinda" / UK: "brilliant", "rubbish" / AU: "heaps", "reckon" |

These imperfections are guided by the reviewer persona and writing temperature — casual reviewers exhibit more irregularities, while formal reviewers write more conventionally.

### Per-Review Injection vs. Batch-Level Instructions

A key architectural distinction in CERA is that Opening Directives and Capitalization Style Directives are **injected per-review** — each individual review receives its own dedicated system prompt with a specific, mandatory directive. This contrasts with the heuristic baseline approach where diversity instructions are given once at the batch level (e.g., "vary your opening lines" or "assign each review a different capitalization style").

| Approach | Mechanism | Enforcement |
|----------|-----------|-------------|
| **CERA (per-review)** | Each review gets a unique system prompt with `"You MUST follow this assigned opening strategy"` and `"You MUST follow this capitalization style"` | Mandatory — the LLM has no choice but to comply since it sees only one directive |
| **Heuristic (batch-level)** | All reviews share a single prompt containing the full list of strategies and a general instruction to vary | Advisory — the LLM tends to favor a subset of patterns and often defaults to proper capitalization |

In practice, we observed that batch-level instructions for capitalization variation are largely ignored by LLMs — even with explicit percentage targets (e.g., "~20% should be lowercase"), generated reviews consistently default to proper capitalization. Per-review mandatory injection overcomes this by removing the LLM's discretion: it receives exactly one style directive and is instructed to follow it without deviation.

This per-review injection is made possible by CERA's architecture where each review is generated as an independent API call with its own composed system prompt, rather than generating multiple reviews in a single batch call.

---

## Example Configuration

CERA uses a single JSON configuration file to control all aspects of generation:

```json
{
  "$schema": "./schema.json",
  "subject_profile": {
    "query": "The Keg Steakhouse",
    "region": "canada",
    "category": "restaurant",
    "feature_count": "5-10",
    "sentiment_depth": "praise and complain",
    "context_scope": "typical dining experiences, food quality, service, ambiance, value"
  },
  "reviewer_profile": {
    "age_range": [18, 65],
    "sex_distribution": {
      "male": 0.45,
      "female": 0.45,
      "unspecified": 0.10
    },
    "audience_context": ["regular diners", "food enthusiasts", "business lunch customers"],
    "additional_context": "The Keg is an upscale steakhouse..."
  },
  "attributes_profile": {
    "polarity": {
      "positive": 0.65,
      "neutral": 0.15,
      "negative": 0.20
    },
    "noise": {
      "typo_rate": 0.01,
      "colloquialism": true,
      "grammar_errors": true
    },
    "length_range": [2, 5],
    "temp_range": [0.7, 0.9]
  },
  "generation": {
    "count": 2000,
    "count_mode": "reviews",
    "batch_size": 50,
    "request_size": 5,
    "mode": "single-provider",
    "provider": "anthropic",
    "model": "claude-sonnet-4-20250514",
    "dataset_mode": "explicit",
    "neb_enabled": true,
    "neb_depth": 2
  },
  "output": {
    "formats": ["jsonl", "csv", "semeval_xml"],
    "directory": "./output/restaurant",
    "include_metadata": true
  }
}
```

See [configs/example.json](configs/example.json) for the complete example.

---

## Output Formats

CERA produces datasets in three formats:

| Format | Use Case | Description |
|--------|----------|-------------|
| **SemEval XML** | ABSA benchmarks | Compatible with SemEval-2014/2015/2016 |
| **JSONL** | Streaming/ML pipelines | One JSON object per line with metadata |
| **CSV** | Analysis | Tabular format, pandas-ready |

---

## Evaluation Metrics (MDQA)

| Category | Metrics |
|----------|---------|
| **Lexical Quality** | BLEU, ROUGE-L |
| **Semantic Similarity** | BERTScore, MoverScore |
| **Corpus Diversity** | Distinct-1/2, Self-BLEU |
| **Downstream Task** | Macro-F1, Per-class F1, APG |

---

## Job Directory Structure

When a generation job runs, CERA creates a structured directory for all job-related files:

```
./jobs/{jobId}-{sanitized-job-name}/
├── amls/                    # AML prompt files (one per review)
│   ├── aml-0001.md
│   ├── aml-0002.md
│   └── ...
├── mavs/                    # MAV raw data (for verification)
│   ├── perplexity-sonar-pro/
│   │   ├── understanding.md # Model's interpretation of the subject
│   │   ├── query.md         # Search queries generated
│   │   └── response.md      # Search results returned
│   ├── anthropic-claude-35-sonnet/
│   │   ├── understanding.md
│   │   ├── query.md
│   │   └── response.md
│   └── google-gemini-25-flash/
│       ├── understanding.md
│       ├── query.md
│       └── response.md
├── metrics/                 # Evaluation metrics (MDQA)
│   ├── mdqa-results.json
│   └── mdqa-results.csv
├── reports/                 # Analysis reports (JSON + CSV)
│   ├── mav-report.json      # MAV verification results
│   ├── mav-summary.csv      # For paper tables
│   ├── mav-facts.csv        # Per-fact verification data
│   ├── conformity-report.json
│   ├── conformity-summary.csv
│   └── conformity-details.csv
└── dataset/                 # Final dataset
    ├── reviews.jsonl
    └── reviews.csv
```

### MAV Raw Data Files

Each MAV model gets its own subfolder with three files:

**understanding.md** - How the model interpreted the subject:
```markdown
# MAV Understanding - perplexity/sonar-pro

## Subject: iPhone 15 Pro

## Subject Type
Consumer electronics (smartphone)

## Relevant Aspects
- Technical specifications
- Camera quality
- Battery life
- Build quality and materials
- Pricing and value

## Timestamp
2026-01-21T10:30:00Z
```

**query.md** - The search queries the model generated:
```markdown
# MAV Query - perplexity/sonar-pro

## Generated Search Queries
1. "iPhone 15 Pro specifications A17 Pro chip"
2. "iPhone 15 Pro camera review 48MP"
3. "iPhone 15 Pro battery life test"
4. "iPhone 15 Pro titanium build quality"
5. "iPhone 15 Pro price comparison"
```

**response.md** - The search results and extracted facts:
```markdown
# MAV Response - perplexity/sonar-pro

## Extracted Facts
- characteristics: ["A17 Pro chip", "48MP main camera", "Titanium frame"]
- positives: ["Premium build quality", "Excellent camera system"]
- negatives: ["High price", "Heavy weight"]
- use_cases: ["Professional photography", "Mobile gaming"]
```

Comparing these files across models allows verification of MAV independence - each model should have different queries reflecting their unique understanding.

---

## Project Structure

```
cera/
├── cli/                     # Python CLI & API
│   └── cera/
│       ├── pipeline/
│       │   ├── composition/ # Phase 1: Context composition
│       │   │   ├── sil.py   # Subject Intelligence Layer
│       │   │   ├── rgm.py   # Reviewer Generation Module
│       │   │   └── acm.py   # Attributes Composition Module
│       │   ├── generation/  # Phase 2: Review generation
│       │   │   ├── aml.py   # Authenticity Modeling Layer
│       │   │   ├── neb.py   # Negative Example Buffer
│       │   │   ├── batch_engine.py
│       │   │   └── noise.py # Noise injection
│       │   └── evaluation/  # Phase 3: Quality assessment
│       │       └── mdqa.py  # Multi-Dimensional Quality Assessment
│       ├── llm/             # LLM provider abstraction
│       ├── prompts/         # Prompt templates
│       ├── models/          # Pydantic models
│       └── utils/           # Utility functions
├── gui/                     # React frontend (Vite + Convex)
├── configs/                 # Configuration files
├── jobs/                    # Generated job output directories
├── output/                  # Dataset outputs
└── assets/                  # Logo and diagrams
```

---

## Preliminary Results

Using [LADy-kap](https://github.com/thangk/LADy-kap) for implicit aspect detection:

| Synthetic Dataset | APG vs Real Data |
|-------------------|------------------|
| Claude Sonnet 4 (2000 reviews) | **-6.8%** |

Synthetic LLM-generated data achieves **up to 93.2%** of real human-annotated dataset performance.

---

## Roadmap

- [ ] Core pipeline implementation
- [ ] Subject Intelligence Layer (SIL)
- [ ] Multi-Agent Verification (MAV)
- [ ] Authenticity Modeling Layer (AML)
- [ ] MDQA evaluation suite
- [ ] Experiments & ablation studies
- [ ] Paper submission
- [ ] Public release

---

## Citation

```bibtex
@mastersthesis{thang2026cera,
  title     = {CERA: Context-Engineered Reviews Architecture for
               Synthetic ABSA Dataset Generation},
  author    = {Thang, Kap},
  school    = {University of Windsor},
  year      = {2026},
  type      = {Master's Thesis}
}
```

---

## Related Projects

- [LADy-kap](https://github.com/thangk/LADy-kap) - Latent Aspect Detection toolkit
- [LADy](https://github.com/fani-lab/LADy) - Original LADy framework

---

## License

This project is for **research purposes only**.

---

## Acknowledgments

- Supervisor: Dr. Luis Rueda *(School of Computer Science)*
- Internal reader: Dr. Arunita Jaekel *(School of Computer Science)*
- External reader: Dr. Mahsa Hosseini *(Odette School of Business)*
- University of Windsor

---

<p align="center">
  <i>Last Updated: February 2026</i>
</p>

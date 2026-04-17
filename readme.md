
# BEAM-Net: Bayesian Event-Driven Attentional Memory Networks

**A Principled Framework for Spike-Based Causal Attention with Uncertainty Quantification**

*Author: Dhouha Meliane — ESPRIT School of Engineering, Tunisia*
*Target: CIFRE partnership (France) in neuromorphic AI for healthcare, autonomous systems, and industrial perception*

---

## Table of Contents

1. [Scientific Motivation](https://claude.ai/chat/72ec1847-3d49-4617-938b-110e1a5725f0#1-scientific-motivation)
2. [Architecture Overview](https://claude.ai/chat/72ec1847-3d49-4617-938b-110e1a5725f0#2-architecture-overview)
3. [Project Structure](https://claude.ai/chat/72ec1847-3d49-4617-938b-110e1a5725f0#3-project-structure)
4. [Industrial Standards Adopted](https://claude.ai/chat/72ec1847-3d49-4617-938b-110e1a5725f0#4-industrial-standards-adopted)
5. [Scientific Rigour Adopted](https://claude.ai/chat/72ec1847-3d49-4617-938b-110e1a5725f0#5-scientific-rigour-adopted)
6. [Data Platform Design (Lakehouse Pattern)](https://claude.ai/chat/72ec1847-3d49-4617-938b-110e1a5725f0#6-data-platform-design-lakehouse-pattern)
7. [Infrastructure Services](https://claude.ai/chat/72ec1847-3d49-4617-938b-110e1a5725f0#7-infrastructure-services)
8. [Database Schema](https://claude.ai/chat/72ec1847-3d49-4617-938b-110e1a5725f0#8-database-schema)
9. [Quick Start](https://claude.ai/chat/72ec1847-3d49-4617-938b-110e1a5725f0#9-quick-start)
10. [Running Experiments](https://claude.ai/chat/72ec1847-3d49-4617-938b-110e1a5725f0#10-running-experiments)
11. [Expected Results](https://claude.ai/chat/72ec1847-3d49-4617-938b-110e1a5725f0#11-expected-results)
12. [CIFRE Positioning](https://claude.ai/chat/72ec1847-3d49-4617-938b-110e1a5725f0#12-cifre-positioning)
13. [References](https://claude.ai/chat/72ec1847-3d49-4617-938b-110e1a5725f0#13-references)

---

## 1. Scientific Motivation

BEAM-Net unifies three foundational paradigms that have developed largely in isolation:

| Paradigm                      | Source Discipline                          | BEAM-Net Role                                                  |
| ----------------------------- | ------------------------------------------ | -------------------------------------------------------------- |
| Modern Hopfield Networks      | Machine Learning (Ramsauer et al., 2020)   | Exponential-capacity associative memory via log-sum-exp energy |
| Free Energy Principle         | Computational Neuroscience (Friston, 2010) | Variational inference for uncertainty quantification           |
| Event-Driven Spiking Networks | Neuromorphic Engineering (Maass, 1997)     | Energy-efficient asynchronous computation                      |

The convergence matters practically: transformer attention consumes  **~150 nJ per retrieval on GPU** , while biological cortex performs equivalent attentional selection at **~0.77 nJ** — a 195× gap. Existing spiking attention models (SpikFormer, Yao et al. 2023) improve efficiency but lack principled uncertainty quantification. Existing Bayesian SNNs (Jang et al., 2019) quantify uncertainty but have no connection to associative memory theory.

BEAM-Net bridges these gaps with a mathematically unified framework deriving its dynamics from first principles rather than heuristic adaptation.

### Building on W-TCRL (Fois & Girau, 2023)

BEAM-Net explicitly extends ideas from the W-TCRL paper ( *Enhanced representation learning with temporal coding in sparsely spiking neural networks* , Front. Comput. Neurosci. 17:1250908):

| W-TCRL Concept                      | BEAM-Net Extension                                     |
| ----------------------------------- | ------------------------------------------------------ |
| Population latency coding (§2.2)   | Preserved as encoding option, extended to DVS events   |
| Deterministic LIF neurons (§2.1.1) | Generalized to stochastic Bayesian LIF (Eq. 9)         |
| STDP learning rule (§2.3)          | Compatible STDP (Eq. 26) + gradient-based optimization |
| Winner-Take-All circuit (§2.3.3)   | Replaced by Dirichlet multi-cause attention (Def. 3.4) |
| Sparsity metric (§3.1.2)           | Preserved, extended with uncertainty decomposition     |
| RMS reconstruction error            | Extended with ECE calibration metric                   |

---

## 2. Architecture Overview

```
Input (DVS events or images)
        │
        ▼
┌─────────────────────┐
│  Temporal Encoder    │  Component 1 (§3.1): Rank-order / population latency coding
│  temporal_encoder.py │  Eq. 7: t_i(ξ) = T_ref − τ · ξ_i
└─────────┬───────────┘
          │ spike times
          ▼
┌─────────────────────┐
│  Coincidence         │  Component 2a (§4.2): Temporal kernel κ(Δt)
│  Detector            │  Eq. 21–22: S_j(ξ) = Σ w_i · κ(t_i(ξ) − t_i(x_j))
│  bayesian_lif.py     │
└─────────┬───────────┘
          │ similarity scores S_j
          ▼
┌─────────────────────┐
│  Bayesian LIF        │  Component 2b (§3.2): Stochastic threshold firing
│  Neurons             │  Eq. 8–9: P(spike | V) = σ((V − θ) / Δ)
│  bayesian_lif.py     │  Prop. 3.3: firing rates → P(cause_j | ξ)
└─────────┬───────────┘
          │ firing rates
          ▼
┌─────────────────────┐
│  Dirichlet           │  Component 3 (§3.3): Multi-cause attention
│  Attention           │  Eq. 12: π ~ Dir(α), α_j = α_0 + η · S_j
│  dirichlet_attn.py   │  Eqs. 14–15: epistemic + aleatoric uncertainty
└─────────┬───────────┘
          │ posterior π
          ▼
┌─────────────────────┐
│  Bidirectional       │  Component 4 (§3.4): Predictive coding loop
│  Inference Loop      │  Eq. 16: top-down modulation
│  bidir_inference.py  │  Thm. 3.6: convergence in ≤ ⌈3τ_m/τ_s⌉ iterations
└─────────┬───────────┘
          │
          ▼
    Classification / Reconstruction + Calibrated Uncertainty
```

---

## 3. Project Structure

```
beam-net/
│
├── .env                              # Credentials (gitignored)
├── .env.example                      # Credentials template (committed)
├── .gitignore                        # Protects secrets
├── docker-compose.yml                # Full infrastructure orchestration
├── Dockerfile                        # Container image definition
├── requirements.txt                  # Pinned Python dependencies
│
├── configs/
│   └── experiment.yaml               # Hyperparameters with paper equation refs
│
├── sql/                              # Database schema & analytics
│   ├── init/                         # Auto-runs on first Postgres startup
│   │   ├── 01_create_databases.sql
│   │   └── 02_grant_permissions.sql
│   ├── migrations/                   # Versioned schema changes (Flyway-style)
│   │   ├── V001__beam_metrics_schema.sql
│   │   ├── V002__add_indexes.sql
│   │   └── V003__add_energy_tracking.sql
│   ├── queries/                      # Reusable analytics queries
│   │   ├── experiment_comparison.sql
│   │   ├── sparsity_evolution.sql
│   │   ├── reliability_diagram.sql
│   │   └── energy_scaling_report.sql
│   └── README.md
│
├── src/
│   ├── config.py                     # Config loader + theoretical validation
│   ├── temporal_encoder.py           # Component 1: encoding
│   ├── bayesian_lif.py               # Component 2: stochastic LIF + coincidence
│   ├── dirichlet_attention.py        # Component 3: multi-cause attention
│   ├── bidirectional_inference.py    # Component 4: predictive coding loop
│   ├── beam_net.py                   # Full model assembly
│   ├── data_loader.py                # DVS datasets via tonic
│   ├── train.py                      # Training + MLflow + Parquet logging
│   ├── evaluate.py                   # Benchmarking vs baselines
│   ├── energy_profiler.py            # Theoretical energy model (Eq. 25)
│   ├── parquet_logger.py             # Lakehouse writer (MinIO Parquet)
│   ├── parquet_analyzer.py           # DuckDB analytics on Parquet
│   ├── report_generator.py           # Scientific PDF report
│   └── utils.py                      # Reproducibility utilities
│
├── dags/
│   └── beam_net_pipeline.py          # Airflow DAG: data → train → eval → report
│
├── data/                             # Dataset cache (gitignored)
└── results/                          # Generated outputs (gitignored)
    ├── BEAM_Net_Report.pdf
    ├── best_model.pt
    ├── training_history.json
    └── *.png
```

---

## 4. Industrial Standards Adopted

### 4.1 Infrastructure as Code

* Entire stack declared in `docker-compose.yml` — reproducible across Windows/Linux/macOS
* Pinned base images (`postgres:15-alpine`, `python:3.10-slim`) prevent version drift
* Service healthchecks enforce startup ordering (e.g., Postgres must be `healthy` before Airflow starts)
* Named volumes (`beam-net-postgres-data`) for easy identification and backup

### 4.2 Twelve-Factor App Compliance

The project follows the [Twelve-Factor App](https://12factor.net/) methodology:

| Factor            | Implementation                                          |
| ----------------- | ------------------------------------------------------- |
| Codebase          | Single Git repo, multiple deploys                       |
| Dependencies      | Explicit `requirements.txt`with pinned versions       |
| Config            | All credentials in `.env`, never in code              |
| Backing services  | Postgres/MinIO attached via URL (swap-in-place)         |
| Build/release/run | Docker multi-stage pattern                              |
| Processes         | Each service is a stateless process                     |
| Port binding      | Every service exposes exactly one port                  |
| Concurrency       | Airflow scales via separate scheduler/worker processes  |
| Disposability     | Containers start <30s, shut down gracefully             |
| Dev/prod parity   | Same Docker images for dev and production               |
| Logs              | Structured `[Module] message`format, stdout streaming |
| Admin processes   | Migrations as separate `db-migrate`service            |

### 4.3 Secrets Management

* `.env` contains all credentials, gitignored
* `.env.example` provides safe-to-commit template with `CHANGE_ME_*` placeholders
* Variable composition: change `POSTGRES_PASSWORD` once, it propagates everywhere via `${VAR}` interpolation
* Airflow Fernet key for encrypting connection secrets in metadata DB
* Production upgrade path: swap `.env` for HashiCorp Vault or AWS Secrets Manager

### 4.4 Database Engineering

* **Versioned migrations** following Flyway naming (`V###__description.sql`)
* **Idempotent scripts** : `CREATE TABLE IF NOT EXISTS`, `INSERT ON CONFLICT DO NOTHING`
* **Referential integrity** : foreign keys with `ON DELETE CASCADE`
* **Generated columns** : `is_correct`, `total_pj` computed at storage time
* **Triggers** : auto-populate `completed_at` timestamps
* **Partial indexes** : `WHERE status = 'running'` for common dashboard queries
* **Views for convenience** : `experiment_summary`, `energy_comparison` denormalize common joins
* **Inline documentation** : every table has `COMMENT ON TABLE` for CIFRE audit trails

### 4.5 Experiment Tracking

* **MLflow** captures params, metrics, artifacts per run
* **PostgreSQL** stores operational data (experiment status, aggregates)
* **Parquet on MinIO** stores analytical data (per-sample predictions, energy time-series)
* Every run produces a **run_id** that joins all three systems
* Git commit hash logged alongside each experiment for full reproducibility

### 4.6 Pipeline Orchestration

Airflow DAG with enforced task dependencies:

```
download_data → train_model → evaluate_model → generate_report
```

* Idempotent tasks (safe to re-run)
* Retry policy: 1 retry, 5-minute backoff
* Task documentation via `doc` attributes (visible in Airflow UI)
* Manual trigger only — no accidental scheduled runs

### 4.7 Object Storage Organization

MinIO buckets follow the  **lakehouse partition convention** :

```
s3://beam-net-results/
├── predictions/
│   └── dataset=nmnist/model=beam_net/experiment_id=42/
│       └── predictions_20260416T130000.parquet
├── energy_timeseries/
│   └── dataset=nmnist/platform=loihi2/experiment_id=42/epoch=15/
│       └── energy_20260416T131200.parquet
└── spike_traces/
    └── experiment_id=42/batch=0/
        └── trace_20260416T132400.parquet
```

This **Hive-style partitioning** is read natively by DuckDB, Spark, Athena, and pandas without a catalog service.

### 4.8 Code Quality

* Type hints throughout: `def forward(self, x: torch.Tensor) -> Tuple[...]`
* NumPy-style docstrings with Parameters/Returns sections
* Single-responsibility modules
* No magic numbers — every constant references a paper equation or peer-reviewed source
* Runtime configuration validation enforcing theoretical constraints (`w_inh > 2.0` per Proposition 3.3)

### 4.9 Observability

Three web dashboards out of the box:

| Tool          | URL                   | Purpose                             |
| ------------- | --------------------- | ----------------------------------- |
| MLflow        | http://localhost:5000 | Metric dashboards, run comparison   |
| Airflow       | http://localhost:8080 | Pipeline status, task logs, retries |
| MinIO Console | http://localhost:9001 | Artifact browser, bucket management |

### 4.10 Network Isolation

All services on a private Docker bridge network (`beam-net-network`). Inter-service communication uses service names (e.g., `http://mlflow:5000`), not localhost. Only documented ports are exposed to the host.

---

## 5. Scientific Rigour Adopted

### 5.1 Traceability to Theory

Every hyperparameter in `configs/experiment.yaml` carries a paper reference:

```yaml
neuron:
  tau_m: 10.0                # Eq. 8
  theta_0: 1.0               # Base threshold
  delta: 0.5                 # Eq. 9: stochasticity parameter
  w_inh: 2.5                 # Prop. 3.3 (must be >2)
```

The config loader (`src/config.py`) validates constraints at load time and raises errors with paper citations if violated:

```python
assert w_inh > 2.0, (
    f"Lateral inhibition w_inh={w_inh} must be > 2.0 "
    f"for posterior convergence (Proposition 3.3, BEAM-Net paper)"
)
```

### 5.2 Biological Plausibility Validation

Hyperparameters constrained to biologically observed ranges with peer-reviewed citations:

| Parameter               | Value     | Biological Source     |
| ----------------------- | --------- | --------------------- |
| Membrane τ_m           | 10 ms     | Gerstner et al., 2014 |
| NMDA coincidence window | 5 ms      | Markram et al., 1997  |
| Sparsity target         | 1–5%     | Lennie, 2003          |
| Gamma oscillation       | 30–80 Hz | Buzsáki & Wang, 2012 |
| Inhibitory τ_IPSP      | 8–12 ms  | Cardin et al., 2009   |

### 5.3 Theoretical Claims → Experimental Validation

Every theorem in the paper has a corresponding empirical test:

| Theoretical Claim                                | Empirical Validation                           |
| ------------------------------------------------ | ---------------------------------------------- |
| Theorem 4.1 (monotonic energy descent)           | Training loss curves (should be monotonic)     |
| Proposition 3.3 (posterior convergence)          | Firing-rate-to-probability calibration (ECE)   |
| Theorem 3.6 (convergence in L ≤ ⌈3τ_m/τ_s⌉) | Iteration count logging per batch              |
| Eq. 25 (energy model)                            | Scaling analysis plot across N                 |
| Eqs. 14–15 (uncertainty decomposition)          | Epistemic > Aleatoric on incorrect predictions |

### 5.4 Controlled Baselines

Three models evaluated on identical data, splits, and training budgets:

* **BEAM-Net** — the proposed architecture
* **ANN-MLP** — deterministic baseline matching GPU attention (Table 1 row)
* **Rate-coded SNN** — temporal coding control (inspired by Tavanaei et al., 2018)

This controls for:

* Network capacity (parameter count matched within ±20%)
* Training data (same seed-controlled splits)
* Preprocessing (same normalization and encoding)
* Evaluation metrics (identical test-set predictions)

### 5.5 Multi-Metric Evaluation

Beyond accuracy alone, each model is scored on:

| Metric                           | What It Measures                      | Why It Matters                     |
| -------------------------------- | ------------------------------------- | ---------------------------------- |
| Accuracy                         | Classification correctness            | Primary task performance           |
| ECE (Expected Calibration Error) | Uncertainty calibration               | Safety-critical decisions          |
| Sparsity                         | Fraction active neurons               | Energy proxy, biological realism   |
| Energy (nJ)                      | Theoretical consumption per inference | Deployment feasibility             |
| Epistemic uncertainty            | Model's knowledge gap                 | Flags "out-of-distribution" inputs |
| Aleatoric uncertainty            | Irreducible data ambiguity            | Informs data quality decisions     |
| Convergence iterations           | Bidirectional loop steps              | Validates Theorem 3.6              |

### 5.6 Reproducibility Discipline

* **Deterministic seeding** across `random`, `numpy`, `torch`, CUDA (via `utils.set_seed()`)
* **Fixed train/val/test split** via seeded `random_split`
* **Git commit hash** stored with each experiment in the database
* **Full config snapshot** stored as YAML in `experiments.config_yaml`
* **Pinned dependencies** prevent version drift (`requirements.txt` + Docker base image)
* **Parquet immutability** — each experiment's predictions are a versioned snapshot

### 5.7 Statistical Honesty

* All baselines report the **same metrics** — no selective metric hiding
* If BEAM-Net loses to a baseline on accuracy, it's reported (we expect this; BEAM-Net's advantage is calibration + energy, not raw accuracy)
* Energy figures are reported as  **theoretical projections** , not measured (until Loihi 2 access is obtained)
* Single-seed runs for initial validation; multi-seed harness ready for final reporting

### 5.8 Scientific Report Standards

The auto-generated PDF follows CNRS/INRIA documentation structure:

1. Abstract
2. Experimental Setup (dataset, config, theoretical constraints)
3. Training Dynamics (energy descent validation)
4. Comparative Results (baseline benchmarks)
5. Energy Efficiency (Eq. 25 validation)
6. Uncertainty Calibration (reliability diagram, decomposition)
7. Conclusions & CIFRE Relevance
8. References (peer-reviewed only)

Every figure is captioned with the equation it validates.

### 5.9 Connection to Prior Work

No silent borrowing — every theoretical influence documented in both code comments and module docstrings:

```python
"""
Component 1: Native Event Encoding (BEAM-Net §3.1)
=====================================================
...
Connection to W-TCRL (Fois & Girau, 2023):
  Population-based latency coding from W-TCRL §2.2 uses Gaussian 
  receptive fields to distribute each input dimension across 
  l neurons, encoding values in relative spike latencies.
"""
```

---

## 6. Data Platform Design (Lakehouse Pattern)

BEAM-Net uses a **hybrid lakehouse architecture** — the pattern deployed at Criteo, Dataiku, and Datadog for production ML systems.

### 6.1 Why Not One Database?

PostgreSQL excels at transactional queries on small-to-medium data but struggles beyond ~10M rows. Parquet excels at analytical scans on large data but lacks transaction support. Neither alone is sufficient.

### 6.2 Data Placement Strategy

| Data Type                     | Expected Volume | Storage                 | Access Pattern       |
| ----------------------------- | --------------- | ----------------------- | -------------------- |
| Experiment metadata           | ~100 rows/month | PostgreSQL              | Transactional, joins |
| Epoch-level metrics           | ~2.5k rows/exp  | PostgreSQL              | Dashboard queries    |
| Per-sample test predictions   | 10k–10M rows   | **Parquet/MinIO** | Analytical scans     |
| Per-batch energy measurements | ~50k rows/exp   | **Parquet/MinIO** | Time-series analysis |
| Spike rasters                 | 10M–1B rows    | **Parquet/MinIO** | Rare, heavy queries  |
| Model weights (.pt)           | Binary blobs    | MinIO (via MLflow)      | Reload for inference |
| PDF reports                   | Binary blobs    | MinIO (via MLflow)      | Download for review  |

### 6.3 Data Flow

```
Training / Evaluation
         │
         ├──► MLflow (runs, params, final metrics) ──► PostgreSQL
         │
         ├──► ParquetMinIOWriter ──► MinIO (Hive-partitioned)
         │                           │
         │                           ├── predictions/
         │                           ├── energy_timeseries/
         │                           └── spike_traces/
         │
         └──► PostgreSQL (custom beam_metrics schema)

Analysis & Reporting
         │
         └──► DuckDB ──► Queries Parquet directly ──► pandas DataFrame ──► plots
```

### 6.4 Compression Wins

With **ZSTD compression** on numerical scientific data:

* CSV format: ~850 MB per 10M prediction rows
* PostgreSQL: ~600 MB per 10M rows
* **Parquet + ZSTD: ~65 MB per 10M rows** (10× reduction)

For spike rasters (high sparsity), storing only non-zero events in sparse long format gives an additional ~100× reduction.

### 6.5 Query Engine: DuckDB

DuckDB is the ideal companion for this architecture:

* **Zero-server** : no separate service to manage
* **Native S3/MinIO support** via the `httpfs` extension
* **Predicate pushdown** : reads only the Parquet columns/rows actually needed
* **Hive partition awareness** : prunes irrelevant files automatically
* **SQL interface** familiar to both engineers and scientists

Example query reading directly from MinIO:

```python
analyzer = BeamNetAnalyzer()
df = analyzer.compare_models(dataset="nmnist")  # Scans all experiments
```

Under the hood, this executes:

```sql
SELECT model, AVG(CAST(is_correct AS DOUBLE)) AS accuracy, ...
FROM read_parquet('s3://beam-net-results/predictions/dataset=nmnist/**/*.parquet',
                  hive_partitioning=1)
GROUP BY model
```

No data loaded into memory until the aggregation completes.

---

## 7. Infrastructure Services

Nine Docker Compose services orchestrate the platform:

```
┌─────────────────────────────────────────────────────────────┐
│                   BEAM-Net Infrastructure                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  postgres ──────► airflow-init ─► airflow-webserver         │
│      │                          └─► airflow-scheduler       │
│      │                                                       │
│      ├────────► db-migrate                                   │
│      │                                                       │
│      └────────► mlflow ────────► (S3 artifacts)            │
│                                                              │
│  minio ──────► minio-init                                    │
│                                                              │
│  beam-net (compute container, uses all above)               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

| Service               | Image                    | Purpose                              |
| --------------------- | ------------------------ | ------------------------------------ |
| `postgres`          | postgres:15-alpine       | Metadata + analytics DB              |
| `minio`             | minio/minio:latest       | S3-compatible object store           |
| `minio-init`        | minio/mc:latest          | Bucket creation (one-shot)           |
| `db-migrate`        | postgres:15-alpine       | Schema migrations (one-shot)         |
| `mlflow`            | Custom (from Dockerfile) | Experiment tracking server           |
| `airflow-init`      | Custom                   | DB bootstrap + admin user (one-shot) |
| `airflow-webserver` | Custom                   | DAG monitoring UI                    |
| `airflow-scheduler` | Custom                   | DAG execution engine                 |
| `beam-net`          | Custom                   | Training/evaluation compute          |

---

## 8. Database Schema

Three logical databases hosted by the single Postgres instance:

### 8.1 `airflow` (managed by Airflow)

Standard Airflow metadata schema. Do not modify.

### 8.2 `mlflow` (managed by MLflow)

Standard MLflow backend schema. Do not modify.

### 8.3 `beam_metrics` (custom, maintained via migrations)

**Core tables:**

| Table                   | Purpose                                             | Paper Reference          |
| ----------------------- | --------------------------------------------------- | ------------------------ |
| `experiments`         | Run registry, joins to MLflow via `mlflow_run_id` | —                       |
| `epoch_metrics`       | Per-epoch training dynamics                         | Theorem 4.1              |
| `test_predictions`    | Per-sample test-set results with uncertainty        | §5                      |
| `energy_measurements` | Energy projections per Eq. 25                       | Table 1                  |
| `sparsity_by_layer`   | Encoding vs representation layer sparsity           | W-TCRL §3.1.2           |
| `hardware_platforms`  | Reference constants (Loihi 2, GPU, bio)             | Davies 2021, Lennie 2003 |
| `energy_timeseries`   | Per-batch energy during training                    | —                       |
| `energy_scaling`      | Dedicated Eq. 25 sweep results                      | §5.3                    |

**Views:**

| View                   | Purpose                                            |
| ---------------------- | -------------------------------------------------- |
| `experiment_summary` | Denormalized latest-epoch metrics per experiment   |
| `energy_comparison`  | Table 1 reproduction with reduction factors vs GPU |

 **Migrations are idempotent and versioned** :

* `V001__beam_metrics_schema.sql` — core tables
* `V002__add_indexes.sql` — performance indexes at scale
* `V003__add_energy_tracking.sql` — enhanced energy schema with hardware reference

---

## 9. Quick Start

### 9.1 Prerequisites

* Docker Desktop (Windows/macOS) or Docker Engine (Linux)
* Docker Compose v2
* 8 GB RAM minimum (16 GB recommended)
* 20 GB free disk space

### 9.2 First-Time Setup

```bash
# 1. Clone or extract project
cd beam-net/

# 2. Copy environment template and customize
cp .env.example .env
# Edit .env to set your preferred credentials

# 3. Build images and launch stack
docker compose up -d --build

# 4. Wait ~90 seconds for all services to initialize
docker compose ps
# All services should show `running` or `healthy`
```

### 9.3 Verify Services

```bash
# PostgreSQL
docker compose exec postgres psql -U beam -d beam_metrics -c "\dt"
# Should list 8 tables

# MinIO
curl http://localhost:9000/minio/health/live
# Returns HTTP 200

# MLflow
curl http://localhost:5000
# Returns MLflow UI HTML

# Airflow
curl http://localhost:8080/health
# Returns JSON health status
```

### 9.4 Access Dashboards

| Service       | URL                   | Credentials                   |
| ------------- | --------------------- | ----------------------------- |
| MLflow        | http://localhost:5000 | —                            |
| Airflow       | http://localhost:8080 | admin / admin (from `.env`) |
| MinIO Console | http://localhost:9001 | minioadmin / minioadmin       |

---

## 10. Running Experiments

### 10.1 Interactive Mode (Recommended for Development)

```bash
# Enter compute container
docker compose exec beam-net bash

# Run training
python -m src.train

# Run evaluation + baselines
python -m src.evaluate

# Generate scientific report
python -m src.report_generator

# Query Parquet data via DuckDB
python
>>> from src.parquet_analyzer import BeamNetAnalyzer
>>> analyzer = BeamNetAnalyzer()
>>> print(analyzer.compare_models(dataset="nmnist"))
```

### 10.2 Orchestrated Mode (Airflow)

1. Open http://localhost:8080
2. Log in as `admin` / `admin`
3. Find the `beam_net_pipeline` DAG
4. Toggle it ON
5. Click "Trigger DAG"
6. Monitor task progression in the Graph view

Each task logs to MLflow and updates the Postgres metadata. Failed tasks retry automatically once after 5 minutes.

### 10.3 Inspecting Results

**MLflow UI** — experiment tracking, run comparison:

* Filter by parameter ranges
* Compare metric curves across runs
* Download model artifacts

**MinIO Console** — artifact storage:

* Browse partitioned Parquet data
* Download reports

**DuckDB on Parquet** — ad-hoc analytics:

```python
from src.parquet_analyzer import BeamNetAnalyzer
analyzer = BeamNetAnalyzer()

# Compare models across all experiments
analyzer.compare_models(dataset="nmnist")

# Reliability diagram data
analyzer.reliability_diagram_data(experiment_id=42)

# Scientific validation: do errors have higher epistemic uncertainty?
analyzer.uncertainty_vs_correctness(experiment_id=42)

# Find most selective neurons (from spike traces)
analyzer.neuron_selectivity(experiment_id=42, top_k=20)
```

**SQL queries on Postgres** — structured analytics:

```bash
docker compose exec postgres psql -U beam -d beam_metrics \
  -v dataset="'nmnist'" -f /sql/queries/experiment_comparison.sql
```

---

## 11. Expected Results

Based on the theoretical claims in the paper and the W-TCRL baseline:

### 11.1 Accuracy

| Model          | N-MNIST Accuracy                                 |
| -------------- | ------------------------------------------------ |
| BEAM-Net       | 85–92%                                          |
| ANN-MLP        | 90–95% (likely slightly higher on raw accuracy) |
| Rate-coded SNN | 80–88%                                          |

**BEAM-Net is not designed to maximize raw accuracy.** Its value proposition is calibrated uncertainty + energy efficiency.

### 11.2 Uncertainty Calibration (ECE)

| Model          | ECE                                      |
| -------------- | ---------------------------------------- |
| BEAM-Net       | **0.03–0.08**(target: calibrated) |
| ANN-MLP        | 0.10–0.15 (typical overconfidence)      |
| Rate-coded SNN | 0.08–0.12                               |

### 11.3 Sparsity

| Model          | Fraction Active        |
| -------------- | ---------------------- |
| BEAM-Net       | 1–5% (matches cortex) |
| Rate-coded SNN | 5–10%                 |

### 11.4 Energy (Theoretical)

| Platform                      | Energy per Inference | Reduction vs GPU   |
| ----------------------------- | -------------------- | ------------------ |
| GPU A100 (ANN attention)      | 150 nJ               | 1×                |
| SpikFormer                    | 25 nJ                | 6×                |
| **BEAM-Net on Loihi 2** | **5–7 nJ**    | **22–30×** |
| Biological cortex             | 0.77 nJ              | 195×              |

### 11.5 Scientific Validation Results

The following claims should be empirically supported:

1. **Monotonic energy descent** (Theorem 4.1) — training loss decreases steadily
2. **Epistemic uncertainty > Aleatoric for errors** — DuckDB query `uncertainty_vs_correctness()` should show epistemic uncertainty is 1.5–3× higher on incorrectly classified samples
3. **Convergence within ⌈3τ_m/τ_s⌉** (Theorem 3.6) — `mean_iterations` in `epoch_metrics` should stay ≤ 10
4. **Sparsity-accuracy Pareto frontier** — BEAM-Net should dominate rate-SNN on the (sparsity, accuracy) plane

---

## 12. CIFRE Positioning

### 12.1 Why This Stack Matters for CIFRE

CIFRE (Conventions Industrielles de Formation par la Recherche) partnerships evaluate proposals on two axes: **industrial relevance** and  **scientific merit** . This project addresses both.

**Industrial evaluators (ANRT)** recognize the stack immediately — it's what's deployed at:

* **Thales** — edge AI for aerospace/defense
* **Renault / Stellantis** — autonomous driving perception
* **Dassault Systèmes** — industrial simulation
* **Atos / Eviden** — ML platform engineering
* **Criteo** — large-scale lakehouse for ML
* **Dataiku** — data science platform

**Academic evaluators (university supervisors)** recognize the scientific discipline — it matches expectations at:

* **CNRS** (Centre National de la Recherche Scientifique)
* **INRIA** (National Institute for Computer Science)
* **Université de Lorraine** (home of W-TCRL authors)
* **ESPCI** (École Supérieure de Physique et de Chimie Industrielles)

### 12.2 Target Application Domains

| Domain                     | BEAM-Net Value                                      | Potential Partners            |
| -------------------------- | --------------------------------------------------- | ----------------------------- |
| Automotive perception      | <5 ms latency, uncertainty-aware handoff            | Valeo, Stellantis, Renault    |
| Medical devices            | Calibrated confidence for safety-critical decisions | GE Healthcare, Philips France |
| Industrial quality control | Event-driven inspection >1000 parts/sec             | Schneider Electric, Siemens   |
| Defense / aerospace        | Edge neuromorphic processing                        | Thales, Dassault, Airbus      |
| Neuromorphic hardware      | Native Loihi 2 deployment pathway                   | Prophesee, GrAI Matter Labs   |

### 12.3 Research Novelty

BEAM-Net is the **first framework** to:

1. Unify Hopfield energy, variational free energy, and spiking dynamics with formal convergence guarantees
2. Provide epistemic/aleatoric uncertainty decomposition in an SNN architecture
3. Propose a concrete Loihi 2 deployment pathway with traceable energy projections
4. Connect modern attention theory to biologically plausible cortical microcircuits

---

## 13. References

### 13.1 Core BEAM-Net References

1. Meliane D (2026).  *BEAM-Net: Bayesian Event-Driven Attentional Memory Networks* . Research proposal.

### 13.2 Foundational Prior Work

2. Fois A, Girau B (2023). Enhanced representation learning with temporal coding in sparsely spiking neural networks. *Front. Comput. Neurosci.* 17:1250908.
3. Ramsauer H et al. (2020). Hopfield networks is all you need.  *arXiv:2008.02217* .
4. Friston K (2010). The free-energy principle: A unified brain theory? *Nat. Rev. Neurosci.* 11(2):127–138.
5. Vaswani A et al. (2017). Attention is all you need. *NeurIPS* 30.
6. Maass W (1997). Networks of spiking neurons: The third generation of neural network models. *Neural Networks* 10(9):1659–1671.

### 13.3 Neuromorphic Hardware

7. Davies M et al. (2021). Advancing neuromorphic computing with Loihi. *Proc. IEEE* 109(5):911–934.
8. Lichtsteiner P et al. (2008). A 128×128 120 dB 15 μs latency asynchronous temporal contrast vision sensor. *IEEE J. Solid-State Circuits* 43:566–576.

### 13.4 Computational Neuroscience

9. Gerstner W et al. (2014).  *Neuronal Dynamics* . Cambridge University Press.
10. Rao RPN, Ballard DH (1999). Predictive coding in the visual cortex. *Nat. Neurosci.* 2(1):79–87.
11. Larkum M (2013). A cellular mechanism for cortical associations. *Trends Neurosci.* 36(3):141–151.
12. Lennie P (2003). The cost of cortical computation. *Curr. Biol.* 13(6):493–497.
13. Knill DC, Pouget A (2004). The Bayesian brain. *Trends Neurosci.* 27(12):712–719.

### 13.5 Datasets

14. Orchard G et al. (2015). Converting static image datasets to spiking neuromorphic datasets using saccades. *Front. Neurosci.* 9:437. *(N-MNIST)*
15. Amir A et al. (2017). A low power, fully event-based gesture recognition system.  *CVPR* . *(DVSGesture)*

### 13.6 Engineering Standards

16. Wiggins A (2017).  *The Twelve-Factor App* . https://12factor.net
17. Apache Parquet Format Specification. https://parquet.apache.org
18. Armbrust M et al. (2020). Delta Lake: High-performance ACID table storage over cloud object stores.  *VLDB* .

---

## License & Contact

This work is licensed for research use. Commercial use requires permission.

 **Author** : Dhouha Meliane
 **Email** : dhouha.meliane@esprit.tn
 **Affiliation** : ESPRIT School of Engineering, Tunisia
 **Project repository** : (to be added upon publication)

For CIFRE partnership inquiries, please include in your message:

* Target application domain
* Available neuromorphic hardware (if any)
* Estimated project timeline
* French academic partner (if already identified)

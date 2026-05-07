# Federated and Hybrid Distributed Learning for Edge Devices

## Problem Statement

Design a federated learning system for edge or mobile devices where multiple clients
collaboratively train a shared model while preserving data locality. Demonstrate hybrid
parallelism and adaptive serving.

---

## Project Structure

```
federated_edge_learning/
├── model.py                  # Neural network with hybrid parallelism (device+server split)
├── data_utils.py             # Non-IID Dirichlet partitioning for realistic edge simulation
├── client.py                 # Flower FL client (edge device simulation)
├── server.py                 # Flower FL server with custom FedAvg strategy + round logger
├── fl_runner.py              # Lightweight in-process FL simulation (no Ray overhead)
├── communication_analysis.py # Per-round metrics, convergence plots, bandwidth analysis
├── adaptive_serving.py       # Adaptive inference deployment across device tiers
├── run_simulation.py         # Full end-to-end simulation entry point
├── requirements.txt          # Python dependencies
└── results/                  # Generated outputs (plots, logs, reports)
    ├── round_log.json
    ├── convergence_and_cost.png
    ├── client_participation.png
    ├── bandwidth_efficiency.png
    └── analysis_report.txt
```

---

## Scope Coverage

### 1. Federated Learning using Flower (flwr)

**File:** `client.py`, `server.py`, `fl_runner.py`

The system uses the [Flower (flwr)](https://flower.ai/) framework to simulate
federated learning across 5 edge clients.

**How it works:**
```
Round N:
  Server  ──── global weights ────►  Client 0 (phone A)
                                      Client 1 (phone B)   ← each trains locally
                                      Client 2 (IoT device)
                                      Client 3 (tablet)
                                      Client 4 (laptop)
          ◄─── updated weights ────  (each client sends back)
  Server: FedAvg aggregation → new global model
```

**Key design choices:**
- `EdgeClient` (Flower `NumPyClient`) handles local training, weight upload
- `EdgeFedAvg` (custom `FedAvg` strategy) aggregates weights and logs rounds
- Each round: server sends weights → clients train → clients send weights back
- **No raw data ever leaves a device** — only model weights are exchanged

**Non-IID Dirichlet partitioning** (`data_utils.py`):
- Simulates realistic edge scenario where devices collect class-skewed data
- Parameter α controls heterogeneity:  α→0 (extreme non-IID), α→∞ (IID)
- Default α=0.5 gives a moderately heterogeneous, realistic distribution

Client data distribution (α=0.5):
```
Client    C0    C1    C2    C3    C4    C5    C6    C7    C8    C9   Total
Client 0  3330  1383    52   877   495   100   963   455  3915   217   11787
Client 1   318   859   272    54  1029  1940   770  1867   177  5046   12332
Client 2     0   232  4510  1675   596   610  2560    63   352   238   10836
Client 3   447  3289   504    16   440  3021  1486  3591  1249   280   14323
Client 4  1905   237   662  3378  3440   329   221    24   307   219   10722
```
Notice: Client 2 has almost no class-0 data; Client 4 dominates class-3 and class-4.
This is exactly the kind of data heterogeneity seen in real edge deployments.

---

### 2. Hybrid Parallelism (Data + Model)

**File:** `model.py` (`EdgeCNN_DevicePart`, `EdgeCNN_ServerPart`, `EdgeCNN_Full`)

**Data Parallelism:**
- Each of the 5 clients trains on its own private data shard (different subset of the dataset)
- Training happens concurrently (or sequentially in simulation) on each device
- Only model weights — never data — are sent to the server
- Enables training on distributed, private edge data

**Model Parallelism (Split Inference):**
The `EdgeCNN_Full` model is split at a natural boundary:

```
[EDGE DEVICE]                          [SERVER]
─────────────────────────────          ─────────────────
Input: 1×28×28                         Feature: 128-d
  ↓ Conv2d(1,16, k=3)                    ↓ Linear(128, 64)
  ↓ MaxPool2d(2)                          ↓ ReLU
  ↓ Conv2d(16,32, k=3)                   ↓ Linear(64, 10)
  ↓ MaxPool2d(2)                          ↓ Logits
  ↓ Flatten
  ↓ Linear(1568, 128)
  ↓ Feature: 128-d  ──── NETWORK ────►
```

**Bandwidth saving with model parallelism:**
| What is transmitted | Size | Savings |
|--------------------|------|---------|
| Raw input (1×28×28) | 3,136 bytes | — |
| Feature vector (128-d) | 512 bytes | **83.7% reduction** |

**Parameter breakdown:**
- Device part: 205,632 parameters (runs on-device)
- Server part:   8,906 parameters (runs on server)

---

### 3. Communication Round Analysis

**File:** `communication_analysis.py`

Generates 3 plots and a detailed text report for every FL run:

**Plot 1: Convergence & Communication Cost**
- Left axis: Server loss and average training loss vs. round
- Right axis: Server accuracy (%) vs. round
- Right chart: Per-round and cumulative upload KB

**Plot 2: Client Participation**
- Bar chart showing how many clients participated in each round
- Mean participation line

**Plot 3: Bandwidth Efficiency**
- Accuracy % gained per MB of data uploaded
- Higher is better; typically peaks mid-training

**Convergence Metrics (example output):**
```
Total rounds         : 10
Total data uploaded  : 15.92 MB
Final server accuracy: 91.23%

Convergence Speed:
  70% accuracy → Round 6
  80% accuracy → Round 7
  85% accuracy → Round 8
  90% accuracy → Round 10
```

**Key insight:** In federated learning, communication cost (MB uploaded) is
often the bottleneck, not compute. The bandwidth efficiency chart shows
you get diminishing accuracy returns per MB after round ~6, which tells
you when to stop training.

---

### 4. Adaptive Inference Deployment

**File:** `adaptive_serving.py`

Edge devices have wildly different capabilities. Serving the same large model
to every device wastes battery on constrained devices and creates latency.

**Three model variants:**

| Tier | Device Examples | Model | Parameters | Mean Latency |
|------|----------------|-------|-----------|--------------|
| HIGH | Laptop, flagship phone | `EdgeCNN_Full` (large) | 214,538 | ~3.7 ms |
| MEDIUM | Mid-range phone, RPi 4 | `EdgeCNN_Medium` | 101,146 | ~0.2 ms |
| LOW | Microcontroller, old phone | `EdgeCNN_Small` | 50,890 | ~0.07 ms |

**Device Profiling → Model Selection pipeline:**
```
DeviceProfiler.profile()
      ↓ (cpu_cores, ram_mb, cpu_freq_mhz)
_assign_tier()   →  "high" | "medium" | "low"
      ↓
ModelSelector.select()
      ↓
Load model + optionally inject federated weights
      ↓
LatencyBenchmarker.benchmark()  →  mean/p95/p99 latency, FPS
      ↓
AdaptiveInferenceEngine.predict()
```

**In production** the medium/small models would be trained via knowledge
distillation from the large teacher model (teacher-student distillation),
ensuring they retain as much accuracy as possible.

---

### 5. Real-World Edge ML Scenario

**Files:** All of the above, tied together

The full system demonstrates a realistic federated learning scenario:

1. **5 heterogeneous edge clients** — each with different amounts of data,
   different class distributions (non-IID), different compute capabilities

2. **Data locality preserved** — the Dirichlet partitioning ensures no
   client ever sees another client's data; only model weights are shared

3. **Communication-aware training** — per-round upload tracking shows
   exactly how much data crosses the network per round

4. **Straggler simulation** — `EdgeClient` can inject random delays to
   simulate slow devices in the network (configurable via `straggler_prob`)

5. **Bandwidth simulation** — configurable upload bandwidth cap
   (`bandwidth_mbps`) to simulate constrained mobile/IoT connectivity

6. **Adaptive deployment** — the trained global model is automatically
   adapted to 3 device tiers at inference time

---

## Quick Start

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run the full simulation
```bash
python fl_runner.py
```

### Run with custom parameters
```bash
python run_simulation.py \
  --rounds 20 \
  --clients 10 \
  --alpha 0.3 \      # more non-IID
  --epochs 5 \       # more local training
  --variant large \  # large | medium | small
  --device cpu
```

### Run individual components
```bash
# Data distribution analysis
python data_utils.py

# Model architecture + hybrid parallelism
python model.py

# Adaptive serving demo
python adaptive_serving.py
```

---

## Convergence Results

Example run (10 rounds, 5 clients, α=0.5, large model):

| Round | Server Loss | Server Accuracy | Upload (KB) |
|-------|------------|----------------|-------------|
| 1     | 2.259      | 14.23%          | 1,630       |
| 3     | 1.632      | 42.37%          | 4,890       |
| 5     | 1.023      | 67.04%          | 8,151       |
| 7     | 0.642      | 82.13%          | 11,411      |
| 10    | 0.310      | 91.23%          | 16,302      |

**Total communication cost: 15.92 MB for 91.23% accuracy.**
This compares favourably to centralised training which would require
transmitting the entire 60,000-sample dataset (~47 MB).

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    FEDERATED LEARNING SERVER                     │
│                                                                  │
│   Global Model ←──── FedAvg Aggregation ◄──────────────────┐   │
│        │                                                    │   │
│        │  Broadcast global weights                          │   │
└────────┼────────────────────────────────────────────────────┼───┘
         │                                                    │
    ┌────▼────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌───┴─────┐
    │Client 0 │  │Client 1 │  │Client 2 │  │Client 3 │  │Client 4 │
    │ phone A │  │ phone B │  │IoT dev  │  │ tablet  │  │ laptop  │
    │         │  │         │  │         │  │         │  │         │
    │local    │  │local    │  │local    │  │local    │  │local    │
    │data     │  │data     │  │data     │  │data     │  │data     │
    │(private)│  │(private)│  │(private)│  │(private)│  │(private)│
    └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘
         │            │            │            │            │
         └────────────┴────────────┴────────────┴────────────┘
                         Send updated weights only
                         (no raw data ever transmitted)
```

---

## Key Concepts Implemented

| Concept | Where | Details |
|---------|-------|---------|
| FedAvg (McMahan 2017) | `server.py`, `fl_runner.py` | Weighted average of client weights ∝ local sample count |
| Non-IID data | `data_utils.py` | Dirichlet(α=0.5) distribution across classes |
| Data Parallelism | `fl_runner.py` | Each client trains on its own shard |
| Model Parallelism | `model.py` | EdgeCNN split into DevicePart + ServerPart |
| Split Inference | `model.py` | 128-d feature vector (83.7% bandwidth saving) |
| Adaptive Serving | `adaptive_serving.py` | 3 model tiers matched to device capability |
| Comm. Analysis | `communication_analysis.py` | Per-round loss, accuracy, upload KB tracked |
| Straggler mitigation | `client.py` | Random delay injection + async-ready design |



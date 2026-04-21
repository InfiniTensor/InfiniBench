# InfiniBench Dashboard User Guide
## 1. Dashboard Overview

InfiniBench Dashboard provides a unified interface to visualize benchmark and evaluation results of AI accelerators across the following scenarios:

- Communication (NCCL / Collective Communication)

- Training (Training / Distributed Training)

- Inference (Direct / Service Inference)

- Operator (Core Operator Performance)

The benchmark framework produces two types of outputs:

```
JSON  -> configuration / environment / scalar metrics
CSV   -> curves / time-series data
```
The Dashboard automatically loads test results and provides unified analysis capabilities, including:

- un ID fuzzy search: locate specific test runs using partial Run IDs

- General filters: filter results by framework, model, device count, etc.

- Multi-run comparison: select multiple runs to compare performance

- Performance visualization: display curves such as latency / throughput / loss

- Statistics and configuration view: inspect throughput statistics, runtime configuration, and environment details

For example, you can enter:
```
allreduce
service
```
to perform fuzzy matching on Run IDs

Example screenshot:
![Run_ID research](./images/runid_research.jpg)
## 2. Running the Dashboard
### 2.1 Environment Requirements

Before using the Dashboard, install the following dependencies:
```
streamlit
plotly
pandas
```
### 2.2 Start the Dashboard

Run the following command in the project root directory:
```
python -m streamlit run dashboard/app.py
```
Access URL after startup:
```
Local URL:    http://localhost:8501
Network URL:  http://<server-ip>:8501
```
Explanation:

Local URL: accessible only on the local machine

Network URL: accessible from other machines within the same network

## 3. Communication Test Analysis

Path:
```
Dashboard → Communication Performance Test
```
Supported features:
```
Bandwidth analysis curve - peak bandwidth

Latency analysis curve - average latency

Test duration

GPU memory usage

Communication configuration analysis
```
Example screenshot:
![Communication Test](./images/dashboard_communication.jpg)

## 4. Inference Test Analysis

Path:
```
Dashboard → Inference Performance Test
```
Modes:
```
Direct Inference
Service Inference
```
Displayed metrics:
```
TTFT

Latency

Throughput

GPU memory usage

Inference configuration analysis
```
Example screenshot:
![Inference Test](./images/dashboard_inference.jpg)

## 5. Training Test Analysis

Path:
```
Dashboard → Training Performance Test
```
Supported features:
```
Loss curve

Perplexity curve

Throughput curve

GPU memory usage

Training configuration analysis
```
Example screenshot:
![Training Test](./images/dashboard_training.jpg)

## 6. Operator Test Analysis

Path:
```
Dashboard → Operator Performance Test
```
Supported metrics:
```
latency

flops

bandwidth
```
Example screenshot:
![Operator Test](./images/dashboard_operators.jpg)

# Simulating TinyML-class Learning in Flower

Re-implementation of the federated learning setup from  
*“Optimizing Federated Learning on TinyML Devices for Privacy Protection and Energy Efficiency in IoT Networks”* using the **Flower** framework.

This project is designed to evaluate the strategies proposed in the paper within the Flower simulation environment. Client-side RAM usage is recorded using `psutil` to monitor the memory footprint of local training on resource-constrained devices.

---

## Citation

W. Villegas-Ch, R. Gutierrez, A. Maldonado Navarro, and A. Mera-Navarrete,  
“Optimizing Federated Learning on TinyML Devices for Privacy Protection and Energy Efficiency in IoT Networks,”  
*IEEE Access*, vol. 12, pp. 174354–174370, 2024.  
DOI: 10.1109/ACCESS.2024.3503516  

---

## Repository Structure

```bash
TinyFed-sim-FLOWER/
├── .gitignore
├── pyproject.toml
├── README.md
├── models/
│   ├── __init__.py
│   └── cnn.py
├── simulations/
│   ├── __init__.py
│   ├── client.py
│   ├── strategy.py
│   ├── server.py
│   └── task.py
```

## Overview
This project extends Flower’s `FedAvg` strategy to record per-client, per-round metrics for experimental analysis.

At the start of each simulation, a unique experiment ID (generated using a timestamp) is recorded at the top of the metrics JSON file:

```bash
    "experiment_id": 20251229_130428,
    "type": experiment_start,
    "timestamp": 1766991868.0635061
```

Server-side global evaluation metrics and client-side training and evaluation metrics are then appended to the same JSON file on a per-round basis.

### Client-side metrics
```bash
    "experiment_id": 20251229_130428,
    "round": FL training rounds,
    "client_id": unique 'id' assigned to the clients,
    "train_loss": recorded training loss in 'local' training epochs,
    "eval_loss": evaluation loss on client-side 'test' data after 
                'local' training in that round,
    "client_accuracy": accuracy on client-side 'test' data,
    "training_time": training time of 'local' epochs,
    "peak_ram_usage_MB": client-side RAM usage after dataset loading, 
                model loading, and 'local' training,
    "global_train_loss": aggregated training loss of all clients
```

### Server-side metrics
```bash
    "experiment_id": 20251229_130428,
    "round": FL training rounds,
    "client_id": GLOBAL,
    "train_loss": aggregated training loss of all clients,
    "eval_loss": evaluation loss on server-side 'test' data unseen by 
                clients on the aggregated model after 'local' training in 
                that round,
    "global_accuracy": accuracy of aggregated model on client-side 
                'test' data,
    "training_time": null,
    "peak_ram_usage_MB": null
```

## Architecture
The system consists of three main components:

* `ClientApp`
Performs local training and evaluation, and reports training metrics such as loss, runtime, and memory usage to the server.

* `ServerApp`
Orchestrates federated rounds, initializes the global model, and invokes the custom strategy. Saves the final trained model to disk.

* **Custom Federated Strategy**

    * Injects server-side round numbers into metrics

    * Aggregates client model updates

    * Merges client training and evaluation metrics

    * Collects and persists client-level and global metrics

    * Tracks experiment metadata across runs

Training-related configurations are defined in pyproject.toml, including:

        Number of clients (num-supernodes)
        Number of FL rounds (num-server-rounds)
        Fraction of clients participating per round (fraction-train)
        Local training epochs (local-epochs)
        Learning rate (lr)

### Dataset
The FEMNIST dataset `(flwrlabs/femnist)` from Hugging Face is used in task.py.

* Images: grayscale, 28 × 28

* Labels: 62 character classes

Dirichlet partitioning is applied using Flower’s `DirichletPartitioner`, partitioning data by `hsf_id` to simulate non-IID client data distributions.

## How to run
clone the git repository and in ubuntu, run:
```bash
pip install -e .
flwr run .
```


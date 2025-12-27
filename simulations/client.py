"""TinyFed-sim: A Flower / PyTorch app."""

import os
import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from simulations.task import Net, load_data
from simulations.task import test as test_fn
from simulations.task import train as train_fn
import psutil
import time

# Flower ClientApp
app = ClientApp()

def load_model(msg: Message) -> torch.nn.Module:
    """Load the model."""
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    return model

def load_data_from_context(context: Context):
    """Load the data."""
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, valloader = load_data(partition_id, num_partitions)
    return trainloader, valloader

@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""
    start_time = time.time()

    process = psutil.Process(os.getpid())
    ini_mem = process.memory_info().rss / (1024)
    print(f"Memory usage before model loading and training: {ini_mem:.2f} kB")

    # Load the model and initialize it with the received weights
    model = load_model(msg)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #model.to(device)

    # Load the data
    trainloader, _ = load_data_from_context(context)

    # Call the training function
    train_loss = train_fn(
        model,
        trainloader,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],
        device,
    )

    end_time = time.time()
    training_time = end_time - start_time

    fin_mem = process.memory_info().rss / (1024)
    print(f"Memory usage after training: {fin_mem:.2f} kB")
    peak_mem = max(ini_mem, fin_mem)    

    # Append to list in context or initialize if it doesn't exist
    if "ram_usage" not in context.state:
        # Initialize MetricRecord in state
        context.state["ram_usage"] = MetricRecord({
            "per_round": []
            })

    # Append to record
    context.state["ram_usage"]["per_round"].append(peak_mem)

    print(
        f"[Client {context.node_id}] "
        f"Round {len(context.state['ram_usage']['per_round'])} "
        f"RAM: {peak_mem:.0f} kB"
    )


    # Construct and return reply Message
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
        "client_id": context.node_id,  # New metric
        "training_time": training_time,  # New metric
        "peak_ram_usage_kb": peak_mem,  # New metric
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    # Load the model and initialize it with the received weights
    model = load_model(msg)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #model.to(device)

    # Load the data
    _, valloader = load_data_from_context(context)

    # Call the evaluation function
    eval_loss, eval_acc = test_fn(
        model,
        valloader,
        device,
    )

    # Construct and return reply Message
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)

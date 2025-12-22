"""TinyFed-sim: A Flower / PyTorch app."""

import os
import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from simulations.task import Net, load_data
from simulations.task import test as test_fn
from simulations.task import train as train_fn
import psutil

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    ini_mem = psutil.Process(os.getpid()).memory_info().rss / (1024)
    print(f"Memory usage before training: {ini_mem:.2f} kB")

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, _ = load_data(partition_id, num_partitions)

    # Call the training function
    train_loss = train_fn(
        model,
        trainloader,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],
        device,
    )

    fin_mem = psutil.Process(os.getpid()).memory_info().rss / (1024)
    print(f"Memory usage after training: {fin_mem:.2f} kB")

    # Append to list in context or initialize if it doesn't exist
    if "ram-usage" not in context.state:
        # Initialize MetricRecord in state
        context.state["ram-usage"] = MetricRecord({"ram-usage": []})
        #context.state["ram-usage"] = []

    # Append to record
    context.state["ram-usage"]["ram-usage"].append(max(ini_mem, fin_mem))
    #context.state["ram-usage"].append(max(ini_mem, fin_mem))


    # Print history
    print(context.state["ram-usage"])

    # Construct and return reply Message
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    _, valloader = load_data(partition_id, num_partitions)

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

"""TinyFed-sim: A Flower / PyTorch app."""

import torch
import numpy as np
from datasets import load_dataset
from models.cnn import Net

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

from datasets import load_dataset
from flwr.app import ArrayRecord, MetricRecord

fds = None
GLOBAL_TESTSET = None
INITIALIZED = False

TEST_WRITER_FRACTION = 0.2
SEED = 42


def build_global_test_writers():
    ds = load_dataset("flwrlabs/femnist", split="train")

    writers = np.unique(ds["hsf_id"])
    rng = np.random.default_rng(SEED)

    test_writers = set(
        rng.choice(
            writers,
            size=int(len(writers) * TEST_WRITER_FRACTION),
            replace=False,
        )
    )

    return test_writers

def build_global_test_datasets(test_writer_fraction=0.2, seed=SEED):
    ds = load_dataset("flwrlabs/femnist", split="train")

    test_writers = build_global_test_writers(
        test_writer_fraction=test_writer_fraction,
        seed=seed,
    )

    test_ds = ds.filter(lambda x: x["hsf_id"] in test_writers)

    return test_ds

def make_preprocessor(test_writers):
    def preprocessor(ds_dict):
        # Filter TRAIN split only
        ds_dict["train"] = ds_dict["train"].filter(
            lambda x: x["hsf_id"] not in test_writers
        )
        return ds_dict
    return preprocessor


def init_client_datasets(num_partitions: int):
    global fds
    if fds is not None:
        return

    test_writers = build_global_test_writers()

    partitioner = DirichletPartitioner(
        num_partitions=num_partitions,
        alpha=0.1,
        partition_by="hsf_id",
        shuffle=True,
        seed=SEED,
    )

    fds = FederatedDataset(
        dataset="flwrlabs/femnist",
        partitioners={"train": partitioner},
        preprocessor=lambda d: {
            "train": d["train"].filter(
                lambda x: x["hsf_id"] not in test_writers
            )
        },
    )



def init_server_testset():
    global GLOBAL_TESTSET
    if GLOBAL_TESTSET is not None:
        return

    test_writers = build_global_test_writers()
    full_ds = load_dataset("flwrlabs/femnist", split="train")

    GLOBAL_TESTSET = full_ds.filter(
        lambda x: x["hsf_id"] in test_writers
    )



pytorch_transforms = Compose([ToTensor(), 
                              Normalize((0.5, ), (0.5, ))
                              ])


def apply_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    batch["image"] = [pytorch_transforms(img) for img in batch["image"]]
    return batch


def load_data(partition_id: int, num_partitions: int):
    init_client_datasets(num_partitions)

    partition = fds.load_partition(partition_id)

    split = partition.train_test_split(test_size=0.2, seed=SEED)
    split = split.with_transform(apply_transforms)

    trainloader = DataLoader(split["train"], batch_size=32, shuffle=True)
    testloader = DataLoader(split["test"], batch_size=32)

    return trainloader, testloader




def train(net, trainloader, epochs, lr, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["image"].to(device)
            labels = batch["character"].to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["image"].to(device)
            labels = batch["character"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy

def central_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    init_server_testset()

    model = Net()
    model.load_state_dict(arrays.to_torch_state_dict())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    testset = GLOBAL_TESTSET.with_transform(apply_transforms)

    testloader = DataLoader(
        testset,
        batch_size=64,
        shuffle=False,
    )

    loss, accuracy = test(model, testloader, device)

    return MetricRecord({
        "accuracy": accuracy,
        "loss": loss,
    })

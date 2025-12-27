"""TinyFed-sim: A Flower / PyTorch app."""

import torch
from pathlib import Path
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from simulations.strategy import FedAvgWithClientLogging 

from simulations.task import Net, central_evaluate

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    fraction_train: float = context.run_config["fraction-train"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["lr"]

    # Load global model
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize custom FedAvg strategy
    metrics_path = PROJECT_ROOT / "client_metrics.json"

    strategy = FedAvgWithClientLogging(
        fraction_train=fraction_train,
        metrics_json_path=str(metrics_path),
    )


    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=central_evaluate,
    )

    # Save final model to disk
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")

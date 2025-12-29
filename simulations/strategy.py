import json
import io
from logging import INFO, log
import time
from typing import List, Tuple
import torch
from pathlib import Path
from typing import Callable, Iterable, Optional
from flwr.app import ArrayRecord, ConfigRecord, Message
from flwr.serverapp import Grid

from flwr.serverapp.strategy import FedAvg, Result
from flwr.common import MetricRecord
from flwr.server.client_proxy import ClientProxy

from datetime import datetime
from typing import Optional

PROJECT_NAME = "TinyFed-sim"

class FedAvgWithClientLogging(FedAvg):
    def __init__(self, *, metrics_json_path: str, experiment_id: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.metrics_json_path = metrics_json_path
        self.client_metrics = []  # in-memory buffer
        self.save_path = None
        self.latest_eval_metrics = {}  # store eval metrics per client

        # --- experiment id ---
        self.experiment_id = (
            experiment_id
            if experiment_id is not None
            else datetime.now().strftime("%Y%m%d_%H%M%S")
        )


    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Include lr decay when needed."""
        return super().configure_train(server_round, arrays, config, grid)
    
    def set_save_path(self, path: Path):
        """Set the path where wandb logs and model checkpoints will be saved."""
        self.save_path = path

    def _update_best_acc(self, current_round: int, accuracy: float, arrays: ArrayRecord) -> None:
        """Update the best accuracy and save the model if needed."""
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.best_round = current_round
            # Save the best model
            if self.save_path is not None:
                file_name = f"model_state_acc_{accuracy}_round_{current_round}.pth"
                best_model_path = self.save_path / file_name
                torch.save(arrays.to_torch_state_dict(), best_model_path)
                print(f"Best model saved with accuracy: {accuracy:.4f} at round {current_round}")

    def aggregate_train(
        self,
        server_round: int,
        results: List[Message],
    ) -> Tuple[Optional[ArrayRecord], Optional[MetricRecord]]:

        print(f"[DEBUG] aggregate_train called for round {server_round}")

        # Let FedAvg aggregate first
        agg_arrays, agg_metrics = super().aggregate_train(server_round, results)

        metrics_path = Path(self.metrics_json_path)

        # Load existing metrics
        if metrics_path.exists():
            try:
                with open(metrics_path, "r") as f:
                    self.client_metrics = json.load(f)
            except json.JSONDecodeError:
                self.client_metrics = []

        existing = {
            (m.get("round"), m.get("client_id"))
            for m in self.client_metrics
            if "round" in m and "client_id" in m
        }


        for message in results:
            metric_record = message.content.get("metrics")
            if metric_record is None:
                continue

            client_id = metric_record.get("client_id")  # <-- use this
            if client_id is None:  # skip entries without client_id (e.g., global metrics)
                continue
            key = (server_round, client_id)
            if key in existing:
                continue

            # Lookup latest eval metrics for this client
            eval_metrics = self.latest_eval_metrics.get(client_id, {})
            accuracy = eval_metrics.get("eval_acc")
            eval_loss = eval_metrics.get("eval_loss")

            self.client_metrics.append({
                "experiment_id": self.experiment_id,
                "round": server_round,
                "client_id": client_id,
                "train_loss": metric_record.get("train_loss"),
                "eval_loss": eval_loss,
                "client_accuracy": accuracy,
                "training_time": metric_record.get("training_time"),
                "peak_ram_usage_MB": metric_record.get("peak_ram_usage_MB"),
                "global_train_loss": agg_metrics.get("train_loss") if agg_metrics else None,
            })

        with open(metrics_path, "w") as f:
            json.dump(self.client_metrics, f, indent=2)

        return agg_arrays, agg_metrics


    def aggregate_evaluate(
        self, server_round: int, results: List[Message]
    ) -> Optional[MetricRecord]:
        agg_metrics = super().aggregate_evaluate(server_round, results)

        # Store per-client evaluation metrics for merging later
        for message in results:
            metrics = message.content.get("metrics")
            if metrics is None:
                continue
            client_id = metrics.get("client_id")
            if client_id is not None:
                self.latest_eval_metrics[client_id] = metrics

        return agg_metrics

    
    def start(
        self,
        grid: Grid,
        initial_arrays: ArrayRecord,
        num_rounds: int = 3,
        timeout: float = 3600,
        train_config: Optional[ConfigRecord] = None,
        evaluate_config: Optional[ConfigRecord] = None,
        evaluate_fn: Optional[
            Callable[[int, ArrayRecord], Optional[MetricRecord]]
        ] = None,
    ) -> Result:
        """Start the federated learning strategy."""

        metrics_path = Path(self.metrics_json_path)

        if metrics_path.exists():
            try:
                with open(metrics_path, "r") as f:
                    self.client_metrics = json.load(f)
            except json.JSONDecodeError:
                self.client_metrics = []

        self.client_metrics.append({
            "experiment_id": self.experiment_id,
            "type": "experiment_start",
            "timestamp": time.time(),
        })

        with open(metrics_path, "w") as f:
            json.dump(self.client_metrics, f, indent=2)


        # Keep track of best acc
        self.best_accuracy = 0.0

        # Initialize if None
        train_config = ConfigRecord() if train_config is None else train_config
        evaluate_config = ConfigRecord() if evaluate_config is None else evaluate_config
        result = Result()

        t_start = time.time()
        # Evaluate starting global parameters
        if evaluate_fn:
            res = evaluate_fn(0, initial_arrays)
            if res is not None:
                result.evaluate_metrics_serverapp[0] = res

        arrays = initial_arrays

        for current_round in range(1, num_rounds + 1):
            log(INFO, "")
            log(INFO, "[ROUND %s/%s]", current_round, num_rounds)

            # -----------------------------------------------------------------
            # --- TRAINING (CLIENTAPP-SIDE) -----------------------------------
            # -----------------------------------------------------------------

            # Call strategy to configure training round
            # Send messages and wait for replies
            train_replies = grid.send_and_receive(
                messages=self.configure_train(
                    current_round,
                    arrays,
                    train_config,
                    grid,
                ),
                timeout=timeout,
            )

            # Aggregate train
            agg_arrays, agg_train_metrics = self.aggregate_train(
                current_round,
                train_replies,
            )

            # Log training metrics and append to history
            if agg_arrays is not None:
                result.arrays = agg_arrays
                arrays = agg_arrays
            if agg_train_metrics is not None:
                log(INFO, "\t└──> Aggregated MetricRecord: %s", agg_train_metrics)
                result.train_metrics_clientapp[current_round] = agg_train_metrics

            # -----------------------------------------------------------------
            # --- EVALUATION (CLIENTAPP-SIDE) ---------------------------------
            # -----------------------------------------------------------------

            # Call strategy to configure evaluation round
            # Send messages and wait for replies
            evaluate_replies = grid.send_and_receive(
                messages=self.configure_evaluate(
                    current_round,
                    arrays,
                    evaluate_config,
                    grid,
                ),
                timeout=timeout,
            )

            # Aggregate evaluate
            agg_evaluate_metrics = self.aggregate_evaluate(
                current_round,
                evaluate_replies,
            )

            # Log training metrics and append to history
            if agg_evaluate_metrics is not None:
                log(INFO, "\t└──> Aggregated MetricRecord: %s", agg_evaluate_metrics)
                result.evaluate_metrics_clientapp[current_round] = agg_evaluate_metrics
                
            # -----------------------------------------------------------------
            # --- EVALUATION (SERVERAPP-SIDE) ---------------------------------
            # -----------------------------------------------------------------

            # Centralized evaluation
            # Centralized evaluation
            if evaluate_fn:
                log(INFO, "Global evaluation")
                res = evaluate_fn(current_round, arrays)
                log(INFO, "\t└──> MetricRecord: %s", res)
                if res is not None:
                    result.evaluate_metrics_serverapp[current_round] = res
                    # Maybe save to disk if new best is found
                    self._update_best_acc(current_round, res["accuracy"], arrays)

                    # --- Append global metrics to JSON ---
                    metrics_path = Path(self.metrics_json_path)
                    if metrics_path.exists():
                        try:
                            with open(metrics_path, "r") as f:
                                self.client_metrics = json.load(f)
                        except json.JSONDecodeError:
                            self.client_metrics = []

                    self.client_metrics.append({
                        "experiment_id": self.experiment_id,
                        "round": current_round,
                        "client_id": "GLOBAL",  # Mark as global
                        "train_loss": agg_train_metrics.get("train_loss") if agg_train_metrics else None,
                        "eval_loss": res.get("loss"),
                        "global_accuracy": res.get("accuracy"),
                        "training_time": None,
                        "peak_ram_usage_MB": None,
                    })

                    with open(metrics_path, "w") as f:
                        json.dump(self.client_metrics, f, indent=2)

                    


        log(INFO, "")
        log(INFO, "Strategy execution finished in %.2fs", time.time() - t_start)
        log(INFO, "")
        log(INFO, "Final results:")
        log(INFO, "")
        for line in io.StringIO(str(result)):
            log(INFO, "\t%s", line.strip("\n"))
        log(INFO, "")

        return result
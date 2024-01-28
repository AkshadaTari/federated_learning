import flwr as fl
from typing import List, Tuple
from flwr.common import Metrics
from report import generate_reports

from flwr.common import NDArrays, Scalar

from typing import Dict, Optional, Tuple
import torch

from model import ConvNeuralNet, load_data, train, test
from collections import OrderedDict

import flwr as fl

# for reproducibility
import torch, random, numpy as np
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

CLIENTS=2
ROUNDS=20 # 20 * 5 epochs = 100 total epochs

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = ConvNeuralNet().to(DEVICE)
# Get model weights as a list of NumPy ndarray's
weights = [val.cpu().numpy() for _, val in net.state_dict().items()]
# Serialize ndarrays to `Parameters`
parameters = fl.common.ndarrays_to_parameters(weights)

# process fit/training result data
# returns {"data": [(client_id, {"acc": float, "loss": float, "examples": int}), ...]}
def fit_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    ret_val = {"data": []}

    for client_id, m in metrics:
        ret_val["data"].append((client_id, m))

    return ret_val

# centralized evaluation of model, use test data from "datasets/evaluate"
# returns { "acc": [(round, accuracy), ...]}
def get_evaluate_fn(net):
    """Return an evaluation function for server-side evaluation."""

    # Load data 
    testloader, num_examples = load_data("datasets/evaluate")

    # The `evaluate` function will be called after every round
    def evaluate(
        server_round: int, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        # Update model with the latest parameters
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

        print(f"Evaluating on {num_examples} unseen images")
        loss, accuracy = test(net, testloader)
        return loss, {"acc": accuracy}

    return evaluate

# Define strategies
# https://flower.dev/docs/framework/ref-api/flwr.server.strategy.html
FedAvg = fl.server.strategy.FedAvg(
    evaluate_fn=get_evaluate_fn(net),
    fit_metrics_aggregation_fn=fit_metrics,
    initial_parameters=parameters,
    min_fit_clients=CLIENTS,
    min_evaluate_clients=CLIENTS,
    min_available_clients=CLIENTS,
)

FedYogi = fl.server.strategy.FedYogi(
    evaluate_fn=get_evaluate_fn(net),
    fit_metrics_aggregation_fn=fit_metrics,
    initial_parameters=parameters,
    min_fit_clients=CLIENTS,
    min_evaluate_clients=CLIENTS,
    min_available_clients=CLIENTS,
)

FedOpt = fl.server.strategy.FedOpt(
    evaluate_fn=get_evaluate_fn(net),
    fit_metrics_aggregation_fn=fit_metrics,
    initial_parameters=parameters,
    min_fit_clients=CLIENTS,
    min_evaluate_clients=CLIENTS,
    min_available_clients=CLIENTS,
)

FedMedian = fl.server.strategy.FedMedian(
    evaluate_fn=get_evaluate_fn(net),
    fit_metrics_aggregation_fn=fit_metrics,
    initial_parameters=parameters,
    min_fit_clients=CLIENTS,
    min_evaluate_clients=CLIENTS,
    min_available_clients=CLIENTS,
)

FedTrimmedAvg = fl.server.strategy.FedTrimmedAvg(
    evaluate_fn=get_evaluate_fn(net),
    fit_metrics_aggregation_fn=fit_metrics,
    initial_parameters=parameters,
    min_fit_clients=CLIENTS,
    min_evaluate_clients=CLIENTS,
    min_available_clients=CLIENTS,
) 

# select strategy
strategy = FedYogi

# run server
history = fl.server.start_server(
    server_address="0.0.0.0:9090",
    config=fl.server.ServerConfig(num_rounds=ROUNDS),
    strategy=strategy,
)

# generate & store reports in reports directory
generate_reports(history, strategy.__class__.__name__)

from model import ConvNeuralNet, load_data, train, test
import torch
import flwr as fl
from collections import OrderedDict

class FedClient(fl.client.NumPyClient):

    def __init__(self, data_dir, client_id):
        print("Initializing client.....")
        print(f"Loading data from {data_dir} directory.....")
        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = ConvNeuralNet().to(DEVICE)
        self.trainloader, self.num_examples = load_data(data_dir)
        self.num_of_epochs = 5
        self.client_id = client_id
        self.round_counter = 0
        
    def get_parameters(self, config):
        print("Getting new model parameters.....")
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        print("Setting new model parameters.....")
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.round_counter += 1
        print(f"Training Round {self.round_counter}")
        self.set_parameters(parameters)
        loss, accuracy  = train(self.net, self.trainloader, epochs=self.num_of_epochs)
        return self.get_parameters(config={}), self.client_id, {"acc": accuracy, "loss": loss, "examples": self.num_examples}

    # ignore evaluation on client, we perform central evaluation on server
    def evaluate(self, parameters, config):
        return float(0), self.client_id, {"acc": 0, "loss": 0, "examples": 0}
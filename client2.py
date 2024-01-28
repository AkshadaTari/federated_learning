from federated_client import FedClient
import flwr as fl

CLIENT_ID=2
DATASET="./datasets/client2/alzheimers"

# start the client with address of the server and path of data directory
fl.client.start_numpy_client(
    server_address="127.0.0.1:9090", 
    client=FedClient(data_dir=DATASET, client_id=CLIENT_ID)
)
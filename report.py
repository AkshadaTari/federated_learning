from flwr.server import History
import matplotlib.pyplot as plt
import json
import pickle 
from PIL import Image, ImageDraw, ImageFont
from prettytable import PrettyTable


# Sample history output
#History (metrics, distributed, fit):
#{'data': [(1, [(2, {'acc': 64.0625, 'loss': 0.9239413123847782, 'examples': 2177}), (1, {'acc': 57.53538995726496, 'loss': 0.9637091912042637, 'examples': 3305})]), (2, [(2, {'acc': 65.87409420289856, 'loss': 0.7096140153598094, 'examples': 2177}), (1, {'acc': 64.63675213675214, 'loss': 0.8055731549572486, 'examples': 3305})])]}
#History (metrics, centralized):
#{'acc': [(0, 1.5250544662309369), (1, 58.06100217864924), (2, 55.33769063180828)]}

# process data & create reports
def generate_reports(history: History, aggr_name: str):
    # process data
    train_loss, train_acc = process_history(history.metrics_distributed_fit["data"])
    test_acc = [acc[1] for acc in history.metrics_centralized["acc"]]

    # create graphs
    create_graph(train_loss, "train_loss", aggr_name)
    create_graph(train_acc, "train_accuracy", aggr_name)
    create_graph(test_acc, "test_accuracy", aggr_name)

    # store train & test data
    json.dump(history.metrics_distributed_fit["data"], open(f"reports/{aggr_name}_train_results.json", 'w' ))
    json.dump(history.metrics_centralized["acc"], open(f"reports/{aggr_name}_test_results.json", 'w' ))

    # calculate and save avg test accuracy
    total_train_loss = 0
    total_train_acc = 0
    total_test_acc = 0
    clients = 0
    for id in train_loss:
        clients += 1
        total_train_loss += train_loss[id][-1]
        total_train_acc += train_acc[id][-1]
        
    total_test_acc = test_acc[-1]

    aggr_results={}
    try: 
        with open('reports/aggr_results.pkl', 'rb') as f:
            aggr_results = pickle.load(f)
    except Exception as e:
        aggr_results={}

    with open('reports/aggr_results.pkl', 'wb') as f:
        aggr_results[aggr_name] = {
            "avg_train_loss": total_train_loss/clients,
            "avg_train_acc": total_train_acc/clients,
            "avg_test_acc": total_test_acc,
            }

        pickle.dump(aggr_results, f)

    table = PrettyTable()
    table.field_names = ["Aggr", "Test Acc %"]
    for k,v in aggr_results.items():
        table.add_row([k, "{:.2f}".format(v["avg_test_acc"])])

    print(table)
    with open('reports/aggr_results_table.txt', 'w') as f:
        f.write(table.get_string())

def create_graph(points, report_type, aggr_name):
    if report_type == "test_accuracy":
        plt.plot(points,'-*')
    else:
        for id, p in points.items():
            plt.plot(p,'-*',label=f"client {id}" )

    plt.suptitle(aggr_name, size=16)
    plt.title(f"{report_type} vs. rounds")
    plt.xlabel('rounds')
    plt.ylabel(report_type)
    plt.legend()

    plt.savefig(f"reports/{aggr_name}_{report_type}_graph.png", bbox_inches='tight')
    plt.close()

#loss: {1: [0.7863981469296941, 0.6805136848898494], 2: [0.6959222245979986, 0.539809363273789]}
#acc: {1: [64.35661764705883, 69.35355392156863], 2: [68.02536231884059, 75.12681159420289]}        
def process_history(data) -> (dict, dict): 
    loss = {}
    acc = {}
    for round, results in data:
        for id, result in results:
            if id not in loss:
                loss[id] = []
            if id not in acc:
                acc[id] = []
            loss[id].append(result['loss'])
            acc[id].append(result['acc'])

    return loss, acc

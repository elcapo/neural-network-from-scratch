import numpy as np
import json
from nn_from_scratch.network import Network


class Repository:
    def store(self, network: Network, file_path: str):
        weights_and_biases = {
            "hidden_layer": {
                "weights": network.hidden_layer.W.tolist(),
                "biases": network.hidden_layer.b.tolist(),
            },
            "output_layer": {
                "weights": network.output_layer.W.tolist(),
                "biases": network.output_layer.b.tolist(),
            },
        }

        with open(file_path, "w") as f:
            json.dump(weights_and_biases, f)

    def load(self, network: Network, file_path: str):
        with open(file_path, "r") as f:
            weights_and_biases = json.load(f)

        hidden_layer = weights_and_biases["hidden_layer"]
        output_layer = weights_and_biases["output_layer"]

        network.hidden_layer.W = np.asarray(hidden_layer["weights"])
        network.hidden_layer.b = np.asarray(hidden_layer["biases"])
        network.output_layer.W = np.asarray(output_layer["weights"])
        network.output_layer.b = np.asarray(output_layer["biases"])

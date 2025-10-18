import numpy as np
from nn_from_scratch.network import Network

class Repository:
    def store(self, network: Network, file_path: str):
        weights_and_biases = {
            "hidden_layer": {
                "weights": network.hidden_layer.W,
                "biases": network.hidden_layer.b,
            },
            "output_layer": {
                "weights": network.output_layer.W,
                "biases": network.output_layer.b,
            }
        }

        np.save(file_path, weights_and_biases)

    def load(self, network: Network, file_path: str):
        weights_and_biases = np.load(file_path, allow_pickle=True)

        hidden_layer = weights_and_biases.item().get("hidden_layer")
        output_layer = weights_and_biases.item().get("output_layer")

        network.hidden_layer.W = hidden_layer["weights"]
        network.hidden_layer.b = hidden_layer["biases"]
        network.output_layer.W = output_layer["weights"]
        network.output_layer.b = output_layer["biases"]
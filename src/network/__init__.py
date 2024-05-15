from .network import DensityNetwork, WireNetwork


def get_network(type):
    if type == "mlp":
        return DensityNetwork
    elif type == 'WIRE':
        return WireNetwork
    else:
        raise NotImplementedError("Unknown network typeß!")


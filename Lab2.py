import numpy as np
from Utilities.Parser import parse_data
from Utilities.Graph import draw_scatter


def run():
    data = parse_data("data/objects20_06.data")
    print(data)
    draw_scatter(data)


if "__main__" == __name__:
    run()

from pathlib import Path

import pyabf

import pandas as pd
import numpy as np
import click

from copy import deepcopy


@click.command()
@click.option("--csv", type=click.Path(exists=True), help="The csv to convert.")
@click.option("--abf", type=click.Path(exists=True), help="The original abf.")
def convert(csv, abf):
    csv_path = Path(csv)

    csv = Path(csv)
    abf = Path(abf)

    # reads the input abf
    abf = pyabf.ABF(abf)
    csv = pd.read_csv(csv)

    time_points = abf.sweepX
    csv_rows = csv.iterrows()
    i, (start, stop, value) = next(csv_rows)

    data = []
    for time_point in time_points:
        if time_point < stop:
            data.append((time_point, value))
        else:
            try:
                while time_point > stop:
                    i, (start, stop, value) = next(csv_rows)
            except StopIteration:
                data.append((time_point.value))

    new_csv = pd.DataFrame(data, columns=["time", "current"])
    new_csv.to_csv(csv_path.parent / f"{csv_path.name[:-4]}_discretized.csv")


if __name__ == "__main__":
    convert()

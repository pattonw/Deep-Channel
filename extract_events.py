from pathlib import Path

import pyabf

import pandas as pd
import numpy as np
import click

from copy import deepcopy


@click.command()
@click.option(
    "--csv", type=click.Path(exists=True), help="The csv from which to extract events."
)
@click.option("--event-conductance-min", type=float)
@click.option("--event-conductance-max", type=float)
def convert(csv, event_conductance_min, event_conductance_max):
    csv_path = Path(csv)

    csv = Path(csv)

    csv = pd.read_csv(csv)

    during_event = 0
    previous_value = None
    forwards_csv = []
    for i, (indx, time, value) in csv.iterrows():
        change = 0 if previous_value is None else value - previous_value
        if (
            during_event == 0
            and event_conductance_min <= change
            and event_conductance_max >= change
        ):
            print(f"Starting event at time {time}")
            during_event = 1
        elif during_event == 1 and change < 0:
            print(f"Ending event at time {time}")
            during_event = 0
        forwards_csv.append((time, during_event))
        previous_value = value

    forwards_csv = pd.DataFrame(forwards_csv, columns=["time", "event"])
    forwards_csv.to_csv(csv_path.parent / f"{csv_path.name[:-4]}_events_lr.csv")

    during_event = 0
    previous_value = None
    reverse_csv = []
    for i, (indx, time, value) in list(csv.iterrows()[::-1]):
        change = 0 if previous_value is None else value - previous_value
        if (
            during_event == 0
            and event_conductance_min <= change
            and event_conductance_max >= change
        ):
            print(f"Starting event at time {time}")
            during_event = 1
        elif during_event == 1 and change < 0:
            print(f"Ending event at time {time}")
            during_event = 0
        reverse_csv.append((time, during_event))
        previous_value = value
    reverse_csv = pd.DataFrame(reverse_csv[::-1], columns=["time", "event"])
    reverse_csv.to_csv(csv_path.parent / f"{csv_path.name[:-4]}_events_rl.csv")


if __name__ == "__main__":
    convert()

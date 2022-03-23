from pathlib import Path

import pyabf

import pandas as pd
import numpy as np
import click

from copy import deepcopy


@click.command()
@click.option("--csv", type=click.Path(exists=True), help="The csv from which to extract events.")
@click.option("--event-conductance-min", type=float)
@click.option("--event-conductance-max", type=float)
def convert(csv, event_conductance_min, event_conductance_max):
    csv_path = Path(csv)

    csv = Path(csv)

    csv = pd.read_csv(csv)

    number_of_events = 0
    previous_value = None
    new_csv = []
    for i, (indx, time, value) in csv.iterrows():
        change = 0 if previous_value is None else value - previous_value

        # Start event
        if event_conductance_min <= change and event_conductance_max >= change:
            print(f"Starting event at time {time}")
            number_of_events = number_of_events + 1

        # End event
        elif number_of_events > 0 and change < 0 and abs(change) > event_conductance_min:
            print(f"Ending event at time {time}")
            number_of_events = number_of_events - 1

        new_csv.append((time, number_of_events))
        previous_value = value

    new_csv = pd.DataFrame(new_csv, columns=["time", "event"])
    new_csv.to_csv(csv_path.parent / f"{csv_path.name[:-4]}_events.csv")


if __name__ == "__main__":
    convert()

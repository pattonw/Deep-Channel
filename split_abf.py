from pathlib import Path

import pyabf

from copy import deepcopy

import numpy as np
import click


@click.command()
@click.option("--abf", type=click.Path(exists=True), help="The abf to split.")
def split(input_file):

    input_file = Path(input_file)
    output_path = input_file.parent

    # reads the input abf
    abf = pyabf.ABF(input_file)

    num_channels = abf.channelCount
    sweep_data = []
    for sweep in abf.sweepList:
        channel_data = []
        for channel in range(num_channels):
            abf.setSweep(sweep, channel=channel)
            y = abf.sweepY  # data
            channel_data.append(y)
        sweep_data.append(channel_data)

    abf.sweepCount = 1
    abf.sweepList = [0]
    for i, channel_data in enumerate(sweep_data):
        for j, y in enumerate(channel_data):
            sweepData = np.empty((1, abf.sweepPointCount))
            sweepData[0] = y
            pyabf.abfWriter.writeABF1(
                sweepData,
                output_path / f"{input_file.name[-4]}_sweep{i}_channel{j}.abf",
                abf.dataRate,
            )


if __name__ == "__main__":
    split()

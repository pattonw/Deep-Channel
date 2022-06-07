from pathlib import Path

import pandas as pd
import click


@click.command()
@click.option(
    "--csv", type=click.Path(exists=True), help="The csv from which to extract events."
)
@click.option("-t", "--threshold", type=float, default=None)
def convert_mean(csv, threshold):
    csv_path = Path(csv)

    csv = Path(csv)

    csv = pd.read_csv(csv)

    if threshold is None:
        threshold = csv["conductancepS"].mean()
    thresholded = (csv["conductancepS"] > threshold).astype(int)

    new_csv = pd.DataFrame(
        list(zip(csv["time"], thresholded)), columns=["time", "event"]
    )
    new_csv.to_csv(csv_path.parent / f"{csv_path.name[:-4]}_events.csv")

    print(new_csv)

if __name__ == "__main__":
    convert_mean()

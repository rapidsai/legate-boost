import argparse
import glob

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

sns.set()
plt.rcParams["font.family"] = "serif"

parser = argparse.ArgumentParser()
parser.add_argument("directory")
parser.add_argument("--log", action="store_true", help="log scale on y-axis")
parser.add_argument("-o", "--output", default="scaling.png")
args = parser.parse_args()


def human_format(num):
    num = float("{:.3g}".format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return "{}{}".format(
        "{:f}".format(num).rstrip("0").rstrip("."), ["", "K", "M", "B", "T"][magnitude]
    )


dfs = [pd.read_csv(f) for f in glob.glob(args.directory + "/*.csv")]
df = pd.concat(dfs, ignore_index=True)
if args.log:
    plt.yscale("log")
    plt.ylabel("Time (s) (log scale)")
    plt.ylim(top=10**4.5)
sns.lineplot(x="n_processors", y="time", data=df, hue="Model type")
scaling = "Strong" if (df["nrows"][0] == df["nrows"]).all() else "Weak"
scaling = "Weak"
nrows_per_proc = df["nrows"][0] / df["n_processors"][0]
plt.title(
    scaling
    + " scaling "
    + human_format(nrows_per_proc)
    + " rows per processor, "
    + human_format(df["ncols"][0])
    + " columns"
)
plt.xscale("log", base=2)
plt.xlabel("Number of Processors")
plt.ylabel("Time (s)")
plt.ylim(ymin=0)
plt.savefig(args.output)

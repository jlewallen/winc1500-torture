#!/usr/bin/python3

import re
import sys
import itertools
import logging
import argparse
import jsonpickle
import jsonpickle.ext.pandas as jsonpickle_pandas
import dataclasses
import typing as t
import matplotlib.pyplot as plt
import numpy as np
import PIL as pil
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

log = logging.getLogger("dma-find")

jsonpickle_pandas.register_handlers()


class Map:
    def __init__(self):
        self.data = []

    def fill(self, offset: int, length: int):
        end = offset + length
        if len(self.data) < end:
            self.data += [0] * (end - len(self.data))
        pass


def generate_png_raw():
    image = pil.Image.new("RGB", [1024, 1024], 255)
    data = image.load()

    for x in range(image.size[0]):
        for y in range(image.size[1]):
            data[x, y] = (
                x % 256,
                y % 256,
                (x ** 2 - y ** 2) % 255,
            )

    image.save("image.png")


def display_map(arr: np.ndarray):
    plt.imshow(arr, cmap="viridis")
    plt.colorbar()
    plt.show()


strip_ansi_re = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def strip_ansi_escaped(line: str) -> str:
    return strip_ansi_re.sub("", line.strip())


data_size_re = re.compile(r"(data-size|size)=(\d+)")
dma_addr_re = re.compile(r"(dma|address)=(\d+)")


DmaEntry = dataclasses.make_dataclass(
    "DmaEntry",
    [
        ("conn", int),
        ("accesses", int),
        ("time", int),
        ("addr", int),
        ("size", int),
        ("kind", int),
    ],
)


class DmaSummary:
    start = 0
    end = 0
    rows = 0

    def __init__(self, entries):
        self.rows = len(entries)
        if self.rows > 0:
            self.start = min([e.addr for e in entries])
            self.end = max([e.addr + e.size for e in entries])

    def __str__(self):
        return "DmaSummary<start=%d, end=%d, length=%d, rows=%d>" % (
            self.start,
            self.end,
            self.end - self.start,
            self.rows,
        )


def generate_graphs(base_name, df):
    sns.set_theme(style="ticks", color_codes=True)

    plt.clf()
    fn = base_name + "-cp_1.png"
    log.info("generating graph {0}".format(fn))
    sns.catplot(x="addr", y="time", height=10, data=df)
    plt.savefig(fn)

    plt.clf()
    fn = base_name + "-cp_2.png"
    log.info("generating graph {0}".format(fn))
    sns.catplot(x="addr", y="time", height=10, data=df, hue="kind")
    plt.savefig(fn)

    log.info("{0} graphs ready".format(base_name))


KindWrite = 1
KindISR = 2


def is_new_connection(line: str) -> bool:
    return "new connection" in line


def get_kind(line: str) -> int:
    if "hif-isr" in line:
        return KindISR
    return KindWrite


def parse_logs(name: str):
    connections: int = 0
    time: int = 0
    accesses: int = 0
    entries: t.List[DmaEntry] = []
    for raw in sys.stdin:
        line = strip_ansi_escaped(raw)
        parts = line.split(" ", 1)

        try:
            time = int(parts[0])
        except ValueError:
            pass

        if len(parts) >= 2:
            if is_new_connection(line):
                connections += 1
                continue

            kind = get_kind(line)

            data_size: int = None
            m = data_size_re.search(parts[1])
            if m:
                data_size = int(m[2])

            dma_addr: int = None
            m = dma_addr_re.search(parts[1])
            if m:
                dma_addr = int(m[2])

            if dma_addr and data_size:
                entry = DmaEntry(connections, accesses, time, dma_addr, data_size, kind)
                entries.append(entry)
                accesses += 1

    return entries


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
        ],
    )

    parser = argparse.ArgumentParser(description="vr tool")
    parser.add_argument("--parse", dest="parse", action="store_true", help="parse data")
    parser.add_argument(
        "--chart", dest="chart", action="store_true", help="chart already parsed data"
    )
    parser.add_argument(
        "--name", dest="name", default=None, help="scenario name", type=str
    )
    args, nargs = parser.parse_known_args()

    log.info("args: {0}".format(args))

    if args.parse:
        entries: t.List[DmaEntry] = parse_logs(args.name)

        summary = DmaSummary(entries)
        log.info("summary: {0}".format(summary))

        fn = args.name + ".pd"
        log.info("writing {0}".format(fn))
        df = pd.DataFrame(data=entries)
        df.sort_values(by=["addr", "time"], inplace=True)
        with open(fn, "w") as f:
            f.write(jsonpickle.encode(df))

    df: pd.DataFrame = None
    fn = args.name + ".pd"
    log.info("opening {0}".format(fn))
    with open(fn, "r") as f:
        df = jsonpickle.decode(f.read())

    log.info("{0}".format(df.describe()))

    if args.chart:
        generate_graphs(args.name, df)

        sns.set_theme(style="ticks", color_codes=True)
        plt.clf()
        fn = args.name + "-sizes.png"
        log.info("generating graph {0}".format(fn))
        g = sns.histplot(x="size", data=df)
        g.set_yscale("log")
        plt.savefig(fn)


if __name__ == "__main__":
    main()

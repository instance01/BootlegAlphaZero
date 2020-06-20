import sys
import os
from collections import defaultdict

import numpy as np

from tensorflow.core.util import event_pb2
from tensorflow.python.lib.io import tf_record


def my_summary_iterator(path):
    for r in tf_record.tf_record_iterator(path):
        yield event_pb2.Event.FromString(r)


def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def aggregate(path):
    keys = [
        'Eval/Reward',
        'Eval/Length',
        'Eval/MCTS_Confidence',
        'Train/AvgLoss'
    ]

    agg_runs = defaultdict(lambda: defaultdict(list))
    agg_keys = {}

    path = os.path.join(path, "event.tfevents")
    for event in my_summary_iterator(path):
        tag = event.summary.value[0].tag
        for key in keys:
            if tag.startswith(key):
                run = tag[tag.rfind('/')+1:]
                agg_runs[key][run].append(event.summary.value[0].simple_value)

    for key in agg_runs:
        aggregated = []
        for run in agg_runs[key]:
            aggregated.append(agg_runs[key][run])
        max_len = max(len(x) for x in aggregated)
        aggregated_ = []
        for x in aggregated:
            aggregated_.append(
                np.pad(
                    x,
                    (0, max_len - len(x)),
                    mode='constant',
                    constant_values=(0, x[-1])
                )
            )
        agg_keys[key] = np.array(aggregated_)

    return agg_runs, agg_keys


def gen_tex(agg_runs, agg_keys):
    gen = """\\documentclass{standalone}
        \\usepackage{tikz,pgfplots}

        \\pgfplotsset{compat=1.10}
        \\usepgfplotslibrary{fillbetween}

        \\begin{document}
        """
    for key in agg_runs:
        data = agg_keys[key]
        data_q50 = smooth(np.percentile(data, 50, axis=0), 20)[10:-9]
        data_q10 = smooth(np.percentile(data, 10, axis=0), 20)[10:-9]
        data_q25 = smooth(np.percentile(data, 25, axis=0), 20)[10:-9]
        data_q75 = smooth(np.percentile(data, 75, axis=0), 20)[10:-9]
        data_q90 = smooth(np.percentile(data, 90, axis=0), 20)[10:-9]
        q50_coords = ""
        q10_coords = ""
        q25_coords = ""
        q75_coords = ""
        q90_coords = ""
        for i, (q50, q10, q25, q75, q90) in enumerate(
                zip(data_q50, data_q10, data_q25, data_q75, data_q90)):
            q50_coords += str((i, q50))
            q10_coords += str((i, q10))
            q25_coords += str((i, q25))
            q75_coords += str((i, q75))
            q90_coords += str((i, q90))

        gen += """
            \\begin{tikzpicture}
                \\begin{axis}[
                    thick,smooth,no markers,
                    grid=both,
                    grid style={line width=.1pt, draw=gray!10},
                    xlabel={Steps},
                    ylabel={%s}]
                ]
                \\addplot+[name path=A,black,line width=1pt] coordinates {%s};
                \\addplot+[name path=B,black,line width=.1pt] coordinates {%s};
                \\addplot+[name path=C,black,line width=.1pt] coordinates {%s};
                \\addplot+[name path=D,black,line width=.1pt] coordinates {%s};
                \\addplot+[name path=E,black,line width=.1pt] coordinates {%s};

                \\addplot[blue!50,fill opacity=0.2] fill between[of=A and B];
                \\addplot[blue!50,fill opacity=0.2] fill between[of=A and C];
                \\addplot[blue!50,fill opacity=0.2] fill between[of=A and D];
                \\addplot[blue!50,fill opacity=0.2] fill between[of=A and E];
                \\end{axis}
            \\end{tikzpicture}
            """ % (
                key, q50_coords, q10_coords, q25_coords, q75_coords, q90_coords
            )
    gen += "\\end{document}"
    return gen


def run():
    if len(sys.argv) < 2:
        for filename in os.listdir('runs/'):
            path = os.path.join('runs/', filename)
            if not os.path.isdir(path):
                continue
            print('Example path: ', path)
            return

    path = sys.argv[1]
    print('Loading', path)
    agg_runs, agg_keys = aggregate(path)
    gen = gen_tex(agg_runs, agg_keys)
    with open('testingtex/a.tex', 'w+') as f:
        f.write(gen)


if __name__ == '__main__':
    run()

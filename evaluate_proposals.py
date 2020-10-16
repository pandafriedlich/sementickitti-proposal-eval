#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 10:36:29 2020

@author: ydzhao
"""

import argparse
import os
import numpy as np
import yaml
import json
from matplotlib import pyplot as plt
import base64

import semkittieval

if __name__ == "__main__":
    # possible splits
    splits = ["train", "valid", "test"]

    parser = argparse.ArgumentParser("./evaluate_proposals.py")
    parser.add_argument(
        '--dataset',
        '-d',
        type=str,
        required=True,
        help='Dataset dir. No Default',
    )
    parser.add_argument(
        '--predictions',
        '-p',
        type=str,
        required=None,
        help='Prediction dir. Same organization as dataset, but predictions in'
        'each sequences "prediction" directory. No Default. If no option is set'
        ' we look for the labels in the same directory as dataset')
    parser.add_argument(
        '--split',
        '-s',
        type=str,
        required=False,
        choices=["train", "valid", "test"],
        default="valid",
        help='Split to evaluate on. One of ' + str(splits) +
        '. Defaults to %(default)s',
    )
    parser.add_argument(
        '--data_cfg',
        '-dc',
        type=str,
        required=False,
        default="config/semantic-kitti.yaml",
        help='Dataset config file. Defaults to %(default)s',
    )
    parser.add_argument(
        '--output',
        type=str,
        required=False,
        default=None,
        help='Output directory for scores.txt and detailed_results.html.',
    )

    FLAGS, unparsed = parser.parse_known_args()

    # assert split
    assert (FLAGS.split in splits)

    # print summary of what we will do
    print("*" * 80)
    print("INTERFACE:")
    print("Data: ", FLAGS.dataset)
    print("Predictions: ", FLAGS.predictions)
    print("Split: ", FLAGS.split)
    print("Config: ", FLAGS.data_cfg)
    print("Output directory: ", FLAGS.output)
    print("*" * 80)

    # set path to proposals and to dataset
    proposals_path = os.path.join(FLAGS.predictions,
                                  "proposals")  # 'data/proposals'
    semantic_kitti_base = FLAGS.dataset  # '/home/snej/data/kitti-odometry/dataset'

    # evaluation
    '''
    The returned eval_results object is a dict-object, whose keys are the object classes like car, person, etc.,
    as well as an additional "mean", which means the mean average recall of all objects. The values of this dict
    are numpy arrays, which record how the recall change when number of proposal grows. For example,
    eval_results['car'][99] means the average recall of cars when 100 objects are accepted.

    '''
    eval_results = semkittieval.evaluate_proposals(
        proposals_path,
        semantic_kitti_base,
        nthreads=4,
        nproposals=400,
        split='test')

    if FLAGS.output is not None:
        print("Generating output files.")

        # print a table of average recalls
        nproposals = np.array([
            20, 30, 50, 100, 200
        ]) - 1  # consider AR_20, AR_30, AR_50, AR_100, AR_200
        table_headers = '\t'.join(["AR_%d" % (np + 1) for np in nproposals])
        print("\t", table_headers)

        for objtype, evalres in eval_results.items():
            # for evaluation result of each class
            res_str_list = ['%.4f' % x
                            for x in evalres[nproposals]]  # AR values
            res_str = '\t'.join(res_str_list)
            if objtype != "mean":
                # make sure mean average recall occupies the last row
                print(objtype, "\t", res_str)
            else:
                mean_res_str = res_str
                print("mean", "\t", mean_res_str)  # print mean average recall

        # plot recall vs. number of proposals
        plt.figure()
        legends = []
        for objtype, evalres in eval_results.items():
            plt.plot(np.arange(400) + 1,
                     evalres)  # plot average recall from 1 to 400
            legends.append(objtype)
        plt.legend(legends)
        plt.xlabel("#proposals")
        plt.ylabel("AR")

        plot_path = os.path.join(FLAGS.output, "ar_curve.png")
        plt.savefig(plot_path, bbox_inches=0, dpi=300)

        data_uri = base64.b64encode(open(plot_path,
                                         'rb').read()).decode().replace(
                                             '\n', '')

        img_tag1 = 'src="data:image/png;base64,{0}"'.format(data_uri)

        nproposals = np.array([20, 30, 50, 100, 200]) - 1
        table_headers = ["AR_%d" % (np + 1) for np in nproposals]

        codalab_output = {}
        for key, value in zip(table_headers, nproposals):
            codalab_output[key] = float(eval_results["mean"][value])

        # generate a table and a plot => detailed_results.html
        output_filename = os.path.join(FLAGS.output, 'scores.txt')
        with open(output_filename, 'w') as outfile:
            yaml.dump(codalab_output, outfile, default_flow_style=False)

        table = []
        mean_entries = {}
        for objtype, evalres in eval_results.items():
            entries = {"class": objtype}
            for key, value in zip(table_headers, nproposals):
                entries[key] = "{:.4f}".format(evalres[value])
            if objtype != "mean":
                table.append(entries)
            else:
                mean_entries = entries

        table.append(mean_entries)

        column_data = """{title: "Class", field:"class", width:200}"""

        for header in table_headers:
            column_data = column_data + ",{title: \"" + header + "\", field:\"" + header + "\", width:150, align: \"center\"}"

        ## producing a detailed result page.
        output_filename = os.path.join(FLAGS.output, "detailed_results.html")
        with open(output_filename, "w") as html_file:
            html_file.write("""
<!doctype html>
<html lang="en" style="scroll-behavior: smooth;">
<head>
  <script src='https://cdnjs.cloudflare.com/ajax/libs/tabulator/4.4.3/js/tabulator.min.js'></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/tabulator/4.4.3/css/bulma/tabulator_bulma.min.css">
</head>
<body>
  <div id="classwise_results"></div>

<script>
  let table_data = """ + json.dumps(table) + """


  table = new Tabulator("#classwise_results", {
    layout: "fitData",
    data: table_data,
    columns: [""" + column_data + """]
  });
</script>
<img border="0" """ + img_tag1 + """ alt="FROC" width="576pt" height="432pt">
</body>
</html>""")

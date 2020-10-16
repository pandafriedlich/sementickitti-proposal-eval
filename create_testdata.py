from genericpath import exists
from os.path import isdir
import sys
import os
from shutil import copy

if __name__ == "__main__":
    print(sys.argv)
    if len(sys.argv) < 3:
        print(
            "Usage: create_testdata.py <datafolder> <proposalfolder> <outputfolder>"
        )
        sys.exit(0)
    data_folder = sys.argv[1]
    proposal_folder = sys.argv[2]
    output_folder = sys.argv[3]

    if not os.path.exists(output_folder): os.makedirs(output_folder)

    proposal_folder = os.path.abspath(
        os.path.join(proposal_folder, "proposals"))

    assert os.path.exists(proposal_folder)
    sequence_dirs = [
        d for d in os.listdir(proposal_folder)
        if os.path.isdir(os.path.join(proposal_folder, d))
    ]

    os.makedirs(os.path.join(output_folder, "sequences"), exist_ok=True)

    for seq in sequence_dirs:
        os.makedirs(
            os.path.join(output_folder, "sequences", seq, "labels"),
            exist_ok=True)
        os.makedirs(
            os.path.join(output_folder, "sequences", seq, "velodyne"),
            exist_ok=True)

        files = os.listdir(os.path.join(proposal_folder, seq))
        indexes = [f.split(".")[0] for f in files]
        for index in indexes:

            def copy_file(file_location):
                copy(
                    os.path.join(data_folder, file_location),
                    os.path.join(output_folder, file_location))

            copy_file(
                os.path.join("sequences", seq, "labels", index + ".label"))
            copy_file(
                os.path.join("sequences", seq, "velodyne", index + ".bin"))
            copy_file(os.path.join("sequences", seq, "calib.txt"))
            copy_file(os.path.join("sequences", seq, "poses.txt"))

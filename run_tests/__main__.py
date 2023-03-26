import argparse
import gdown
import numpy as np

# Student submission file
import submission

def _load_data(data_id):
    local_file = "%s.npz" % data_id
    gdown.download(id=data_id, output=local_file, quiet=True)
    return np.load(local_file)['arr_0']

def run_1a(data):
    for i in range(len(data)):
        assert submission.acceptance_ratio(data[i, 0], data[i, 1]) == data[i, 2]


tests = {
    "1a": run_1a
}

parser = argparse.ArgumentParser()
parser.add_argument("test_id")
parser.add_argument("data_id")
args = parser.parse_args()
tests[args['test_id']](_load_data(args['data_id']))


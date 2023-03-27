import argparse
import importlib
import gdown
import numpy as np


def _load_data(data_id):
    local_file = "%s.npz" % data_id
    gdown.download(id=data_id, output=local_file, quiet=True)
    return np.load(local_file)['arr_0']


parser = argparse.ArgumentParser()
parser.add_argument("submission_id")
parser.add_argument("test_id")
parser.add_argument("data_id")
parser.add_argument("--precision", default=4)
args = parser.parse_args()

module = importlib.import_module(args.submission_id)
func = getattr(module, args.test_id)
data = _load_data(args.data_id)

for i in range(len(data)):
    np.testing.assert_almost_equal(func(*data[i, :-1]), data[i, -1],
                                   decimal=args.precision)

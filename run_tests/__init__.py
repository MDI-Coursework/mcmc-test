import importlib
import gdown
import numpy as np

def _load_data(data_id):
    local_file = "%s.npz" % data_id
    gdown.download(id=data_id, output=local_file, quiet=True)
    return np.load(local_file)['arr_0']


def run(submission_id, test_id, data_id, precision):
    module = importlib.import_module(submission_id)
    func = getattr(module, test_id)
    data = _load_data(data_id)

    for i in range(len(data)):
        np.testing.assert_almost_equal(func(*data[i, :-1]), data[i, -1],
                                       decimal=precision)
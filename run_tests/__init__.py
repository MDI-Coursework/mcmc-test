import importlib
import gdown
import numpy as np


def _load_data(data_id):
    local_file = "%s.npz" % data_id
    gdown.download(id=data_id, output=local_file, quiet=True)
    return np.load(local_file)['arr_0']


def run(submission_id, test_id, data_id, precision, output_dim=1):
    module = importlib.import_module(submission_id)
    func = getattr(module, test_id)
    data = _load_data(data_id)

    for i in range(len(data)):
        np.testing.assert_almost_equal(func(*data[i, :-output_dim]),
                                       data[i, -output_dim:], decimal=precision)


def gen(submission_id, test_id, prefix,
        output_dim=1, hook_f=None,
        input_data=None,
        seed=None, num_data=None, low=None, high=None):
    module = importlib.import_module(submission_id)
    func = getattr(module, test_id)

    # Generate data if not provided
    if input_data is None:
        # uniformly sample input data
        input_dim = len(low)
        rng = np.random.default_rng(seed=seed)
        input_data = rng.uniform(low=low, high=high,
                                 size=(num_data, input_dim))
        prefix = "%s_%016d_%05d" % (prefix, seed, num_data)

    # apply hook / input data transformation if needed
    if hook_f is not None:
        input_data = hook_f(input_data)

    # apply the function to inputs
    data = np.hstack((input_data, np.zeros((len(input_data), output_dim))))
    for i in range(len(input_data)):
        data[i, -output_dim:] = func(*data[i, :-output_dim])

    np.savez('%s.npz' % prefix, data)
    return data

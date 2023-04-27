import importlib
from numbers import Number
from typing import Callable, Optional
import gdown
import numpy as np
from numpy.typing import ArrayLike, NDArray


def _load_data(data_id):
    local_file = "%s.npz" % data_id
    gdown.download(id=data_id, output=local_file, quiet=True)
    return np.load(local_file)['arr_0']


def run(test_id, data_id, precision = 4, 
        submission_id = "submission", output_dim=1, pass_percent=0.9):
    module = importlib.import_module(submission_id)
    func = getattr(module, test_id)
    data = _load_data(data_id)

    num_error = 0
    for i in range(len(data)):
        try:
            np.testing.assert_almost_equal(func(*data[i, :-output_dim]),
                                           data[i, -output_dim:], 
                                           decimal=precision)
        except AssertionError:
            num_error +=1

    assert (num_error / len(data)) < (1.0 - pass_percent)
        


def gen(test_id:str, prefix:str, submission_id:Optional[str]= "submission", 
        precision:Optional[int]=4, input_data:Optional[NDArray]=None, 
        seed:Optional[int]=None, num_data:Optional[int]=None, 
        low:Optional[int|ArrayLike]=None, high:Optional[float|ArrayLike]=None, 
        gen_f:Optional[Callable] = None, hook_f:Optional[Callable]=None):

    """ Generates test data, including optionally generating test inputs

    Args:
        test_id (str): The function name in submission file, i.e. what is 
            imported from the submission module to run the test
        prefix (str): The output file prefix (could have other metadata 
            appended to it when saving the actual test data).
        submission_id (Optional[str]): The submission module (i.e. the name of 
            the submission python file without the '.py' extension.  
            Default = 'submission'.
        precision (Optional[int]):The number of decimal places to round the
            output values to. Default = 4.
        input_data (Optional[NDArray]): An explicit 2D NDArray of input data.
            If specified, then this data will be used, rather than generating
            random input data. Default = None.
        seed (Optional[int]): A seed to the Numpy default RNG to 
            instantiate a RNG in case one is needed to sample input data. 
            Default = None.
        num_data (Optional[int]): The number of input/output data to generate 
            in case we need to sample input data. Default = None.
        low (Optional[int | ArrayLike]): The lower bound(s) of the input data
            to sample uniformly if not using a generator function. 
            Default = None.
        high (Optional[float | ArrayLike]): The upper bound(s) of the input data
            to sample uniformly if not using a generator function.
            Default = None
        gen_f (Optional[Callable]): A generator function that takes in a 
            Numpy RNG and the number of input data rows to sample, and returns
            that many datapoints. Either `one of a) input_data`, b) `num_data, 
            seed, gen_f` or c) `num_data, seed, low, high` must be specified to 
            proerly define input data. Defaults to None.
        hook_f (Optional[Callable], optional): Post-processing hook that 
            transforms input data after it's been specified / sampled. Must 
            accept a Numpy array of input data and return another Numpy array of 
            input data.

    Returns:
        A 2D numpy array of generated input/output data pairs.
    """

    module = importlib.import_module(submission_id)
    func = getattr(module, test_id)

    # Generate data if not provided
    if input_data is None:
        rng = np.random.default_rng(seed=seed)
        prefix = "%s_%016d_%05d" % (prefix, seed, num_data)

        # If generator function is not provided, then just uniformly sample
        # inputs
        if gen_f is None:
            if isinstance(low, Number): # Standardize form of low and high
                low, high = [low], [high]
            input_data = rng.uniform(low=low, high=high, 
                                     size=(num_data, len(low)))

        else: # If generator is provided, just call it
            input_data = gen_f(rng, num_data)

    # apply hook / input data transformation if needed
    if hook_f is not None:
        input_data = hook_f(input_data)

    # Calculate output dimensionality
    output_dim = len(func(*input_data[0]))

    # apply the function to inputs
    data = np.hstack((input_data, np.zeros((len(input_data), output_dim))))

    for i in range(len(input_data)):
        data[i, -output_dim:] = np.around(func(*data[i, :-output_dim]),
                                          decimals=precision)
    np.savez('%s.npz' % prefix, data)
    return data

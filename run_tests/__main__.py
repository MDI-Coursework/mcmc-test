import argparse
from . import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("submission_id")
    parser.add_argument("test_id")
    parser.add_argument("data_id")
    parser.add_argument("--precision", default=4)
    args = parser.parse_args()
    run(args.submission_id, args.test_id, args.data_id, args.precision)

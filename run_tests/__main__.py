import argparse
from . import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("test_id")
    parser.add_argument("data_id")
    parser.add_argument("--submission_id", default="submission")
    parser.add_argument("--precision", default=4, type=int)
    parser.add_argument("--output_dim", default=1, type=int)
    args = parser.parse_args()
    run(args.test_id, args.data_id, 
        precision=args.precision,
        submission_id="submission",
        output_dim=args.output_dim)

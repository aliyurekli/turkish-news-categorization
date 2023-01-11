import argparse

from ttc3600 import stratified_k_fold

CLI = argparse.ArgumentParser()
CLI.add_argument("csv_path", help="Absolute path of the dataset csv file")
CLI.add_argument("target_dir", help="Absolute path of the target directory")

if __name__ == '__main__':
    args = CLI.parse_args()
    stratified_k_fold(csv_path=args.csv_path, target_dir=args.target_dir)
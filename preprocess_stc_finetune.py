import json
import os
import argparse
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_dir", default="/home/gzy/workspace/CPM/nlpdata/STC", type=str, help="The input dir of original ChID data.")
    parser.add_argument("--output_dir", default="/home/gzy/workspace/CPM/CPM-1-Finetune/data/STC", type=str, help="The processed data output dir.")

    args = parser.parse_args()

    # train
    with open(os.path.join(args.data_dir, 'STC.json'), 'r') as f:
        raw_data = json.loads(f.read())
        raw_data = raw_data["train"]

    with open(os.path.join(args.output_dir, 'train_all.txt'), 'w') as f_out:
        for pr_pair in tqdm(raw_data, desc="Building Train All"):
            f_out.write("对话上文:" + "".join(pr_pair[0].strip().split()) + " 回复:" + "".join(pr_pair[1].strip().split()) + "\n")

    with open(os.path.join(args.output_dir, 'train.txt'), 'w') as f_out:
        for pr_pair in tqdm(raw_data[:int(0.1 * len(raw_data))], desc="Building Train"):
            f_out.write("对话上文:" + "".join(pr_pair[0].strip().split()) + " 回复:" + "".join(pr_pair[1].strip().split()) + "\n")

    # valid
    with open(os.path.join(args.data_dir, 'STC.json'), 'r') as f:
        raw_data = json.loads(f.read())
        raw_data = raw_data["valid"]

    with open(os.path.join(args.output_dir, 'valid.txt'), 'w') as f_out:
        for pr_pair in tqdm(raw_data, desc="Building Valid"):
            f_out.write("对话上文:" + "".join(pr_pair[0].strip().split()) + " 回复:" + "".join(pr_pair[1].strip().split()) + "\n")
    
    # test
    with open(os.path.join(args.data_dir, 'STC_test.json'), 'r') as f:
        raw_data = json.loads(f.read())
        raw_data = raw_data["test"]

    with open(os.path.join(args.output_dir, 'test.txt'), 'w') as f_out:
        for pr_pair in tqdm(raw_data, desc="Building Test"):
            f_out.write("对话上文:" + "".join(pr_pair[0].strip().split()) + " 回复:" + "".join(pr_pair[1].strip().split()) + "\n")

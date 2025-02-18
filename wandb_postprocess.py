import argparse
import json
import os
import tiktoken

import pandas as pd


def main(input_path, output_path):

    if input_path.endswith(".csv"):
        df = pd.read_csv(input_path)
    elif input_path.endswith(".json"):
        with open(input_path, "r") as f:
            dataset = json.load(f)
        df = pd.DataFrame(dataset["data"], columns=dataset["columns"])
    # print(df)

    data_list = []
    count = 0
    for i, data in df.iterrows():
        # print(data)
        data_dict = {}
        model_name = data["estimator"].split("/")[-1]

        data_dict['question_id'] = data['question_id']
        data_dict['instruction'] = data['prompt']
        data_dict['model'] = model_name
        data_dict['language'] = "English"
        try:
            data_dict['output'] = eval(data["generations"])[0]
        except:
            data_dict['output'] = "Error"
            count += 1
            continue
        
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        data_dict['token_len'] = len(encoding.encode(data_dict['output'], disallowed_special=()))

        data_list.append(data_dict)

    print(count)

    with open(os.path.join(output_path, f'{model_name}.jsonl'), "w") as f:
        for data in data_list:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run generation on a dataset")
    parser.add_argument(
        "--input_path",
        type=str,
        default="/home/olivernan_cohere_com/local-disk/multilingual_LMM_vision_winrate_results/wildvision/dx31ultp_alpha5.csv",
        help="Path to the dataset",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="/home/olivernan_cohere_com/WildVision-Bench/data/vision_bench_0617/model_answers/",
        help="Path to the dataset",
    )

    args = parser.parse_args()
    print(args)
    main(args.input_path, args.output_path)


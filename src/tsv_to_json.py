import argparse
import json

def convert_tsv_to_json(tsv_filename: str, json_filename: str):
    with open(tsv_filename, 'r') as file:
        header = file.readline().strip().split('\t')

        data = [
            dict(zip(header, line.strip().split('\t'))) 
            for line in file.readlines()
        ]

    with open(json_filename, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    parser.add_argument('out_filename')
    args = parser.parse_args()

    convert_tsv_to_json(args.filename, args.out_filename)
    '''
    filename = '/Users/spygaurad/Xai_llm/Interpret_Instruction_Tuning_LLMs/results/linear_interpret/linear_concept_seed0.tsv'
    out_filename = 'linear_concept_seed0.json'
    convert_tsv_to_json(filename, out_filename)
    # '''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json

# Input and output file paths
input_file = "/gpfs/radev/pi/q_chen/yf329/PMC_instruction_data/pilot-test/final_instruction_subimage_option_6_qa.json"
output_file = "/gpfs/radev/pi/q_chen/yf329/PMC_instruction_data/benchmark/benchmark_data/final_instruction_option_original.json"

# Indices you want to extract
indices = [0, 1, 2, 3, 5, 8, 9, 11, 13, 14,15,17,20,22,23,25,26,28,29,30,32,33,34,37,38,39,52,53,54,55,56,58,61,63,65,66,67,69,70,72,76,77,78,85,86,87,89,91,92,93,94,95,96,97,98,101]
#indices = [0,4,5,6,8,10,12,14,15,18,19,20,21,25,27,29,31,34,35,36,37,41,44,45,49,50,53,55,58,60,61,62,63,64,65,68,70,71,72,73,79,82,87,88,90,91,92,93,94,96]
#indices = [1,2,3,4,5,6,7,8,9,10,12,13,14,15,16,17,19,20,21,22,23,24,25,26,27,28,29,30,32,33,34,35,36,38,39,40,41,43,44,45,47,49,50,52,53,54,55,56,57,58]
#indices = [1,2,4,8,10,12,13,18,19,20,21,22,23,24,25,26,27,28,30,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,50,51,52,53,54,56,57,59,61,62,64,65,68,70,71,72,73,74,75]
#indices=[1,2,8,9,10,11,12,13,14,15,16,17,18,19,23,24,25,26,27,28,29,30,31,32,33,35,36,37,38,39,40,42,43,44,45,47,48,49,50,51,52,53,54,55,56,57,58,60,61,63]
#indices=[1,4,8,10,12,21,33,34,35,36,37,38,43,44,45,48,50,51,52,53,54,57,61,64,65,70,71,72,73,75,76,79,84,87,88,90,91,94,96,97,98,101,102,103,106,107,109,111,114,117,118,119,120,121,122,123,124,125]
#indices= [1,2,5,7,11,12,13,14,15,16,19,18,20,21,22,24,25,26,27,29,30,31,32,33,34,35,36,37,38,41,42,44,45,46,50,51,53,55,57,58,59,60,62,64,65,66,67,68,69,71]

def main():
    # Load JSON list
    with open(input_file, "r") as f:
        data = json.load(f)

    # Extract selected samples
    extracted = [data[i] for i in indices if i < len(data)]

    # Save extracted list as JSON
    with open(output_file, "w") as f:
        json.dump(extracted, f, indent=2, ensure_ascii=False)

    print(f"Extracted {len(extracted)} samples and saved to {output_file}")

if __name__ == "__main__":
    main()

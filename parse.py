import os
import pandas as pd

def parse_file(file_path):
    op_sequence = []
    with open(file_path, 'r', encoding='iso8859-15') as asm:
        for line in asm.readlines():
            line_split = line.strip().split()
            for split in line_split:
                # ingore that line when a comment starts
                if split.startswith(";"):
                    break
                if split.lower() in opcodes:
                    op_sequence.append(split)
                    continue

    return op_sequence

if __name__ == "__main__":
    global opcodes
    with open("opcodes.txt") as opcode_file:
        opcodes = opcode_file.readlines()
    for i, opcode in enumerate(opcodes):
        opcodes[i] = opcode.strip()

    TRAIN_SET = 'train'
    train_df = pd.read_csv('trainLabels.csv')
    
    classes = [i for i in range(1, 10)]
    samples = [0]

    for _class in classes:
        _class_items = train_df[train_df['Class'] == _class]
        _class_samples = _class_items.sample(n=10, random_state=42)
        samples.append(_class_samples)

    
    for _class in classes:
        os.makedirs(os.path.join("opcodes", str(_class)), exist_ok=True)
        with open(os.path.join("opcodes", str(_class),"parsed_opcodes.txt"), "w") as out_file:
            for i, sample in enumerate(samples[_class]['Id']):
                parsed_codes = parse_file(os.path.join(TRAIN_SET, f"{sample}.asm"))
#         parsed_codes = parse_file('sample.asm')
                out_file.write(f"{sample}" + ": ")
                for code in parsed_codes[:-1]:
                    out_file.write(code + ",")
                print(i, sample, "done", len(parsed_codes))
                out_file.write(parsed_codes[-1] + "\n")
        
    opcode_samples = {}
    for _class in classes:
        with open(os.path.join("opcodes", str(_class), "parsed_opcodes.txt"), "r") as in_file:
            for line in in_file.readlines():
                split = line.split(":")
                filename = split[0]
                code_split = split[1]
                opcode_samples[filename] = code_split.strip().split(",")
                
    for sample_file in opcode_samples.keys():
        print(sample_file + ".asm: ", len(opcode_samples[sample_file]), "opcodes")
                
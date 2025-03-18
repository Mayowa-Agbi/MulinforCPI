import torch
import esm
from tqdm import tqdm
import pandas as pd
import os
import sys

# Detect CUDA device automatically
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load ESMFold model
model = esm.pretrained.esmfold_v1().eval().to(device)

def main(data_path):
    files = os.listdir(data_path)
    
    for file in files:
        if file.endswith('.csv'):
            re_fold = os.path.join(data_path, 'esm1_' + file[:-4])  # Fixed variable name
            os.makedirs(re_fold, exist_ok=True)
            
            data = pd.read_csv(os.path.join(data_path, file))
            uni_sequences = sorted(list(set(data['sequence'])), key=len)  # Unique sequences sorted by length

            # Create sequence-to-index mapping
            seq_to_index = {seq: i for i, seq in enumerate(uni_sequences)}
            data['sequence_index'] = data['sequence'].map(seq_to_index)

            # Save sequence mapping dictionary
            list_match_df = pd.DataFrame(list(seq_to_index.items()), columns=['sequence', 'index'])
            list_match_df.to_csv(os.path.join(re_fold, file + '_dic.csv'), index=False)

            # Save updated data
            data.to_csv(os.path.join(re_fold, 'esm_' + file), index=False)

            # Predict protein structures
            for index, sequence in enumerate(tqdm(uni_sequences, desc="Processing Sequences")):
                with torch.no_grad():
                    output, _, _, _ = model.infer_pdb(sequence[:500])  # Truncate sequences to 500 AA
                with open(os.path.join(re_fold, f"{index}_result.pdb"), "w") as f:
                    f.write(output[0])

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python generate_protein_fold.py <data_folder>")
        sys.exit(1)
    
    data_path = sys.argv[1]
    main(data_path)

#!/usr/bin/env python3
"""列出 EMDB 数据集中指定 split 的所有序列"""

import argparse
import pickle as pkl
from glob import glob
import os

def main():
    parser = argparse.ArgumentParser(description='List EMDB sequences for a specific split')
    parser.add_argument('--split', type=int, default=2, help='EMDB split number (1, 2, or 3)')
    parser.add_argument('--dataset_path', type=str, 
                       default='/home/gejunchen/Work/2026-1/Datasets/EMDB',
                       help='Path to EMDB dataset')
    parser.add_argument('--person', type=str, default=None,
                       help='Filter by person (e.g., P0)')
    args = parser.parse_args()
    
    dataset_path = args.dataset_path
    split = args.split
    
    # Determine which persons to check
    if args.person:
        person_ids = [int(args.person[1:])] if args.person.startswith('P') else [int(args.person)]
    else:
        person_ids = range(10)
    
    print(f"{'='*80}")
    print(f"EMDB Split {split} Sequences")
    print(f"Dataset: {dataset_path}")
    print(f"{'='*80}\n")
    
    total_count = 0
    person_sequences = {}
    
    for p in person_ids:
        folder = f'{dataset_path}/P{p}'
        if not os.path.exists(folder):
            continue
        
        roots = sorted(glob(f'{folder}/*'))
        sequences = []
        
        for root in roots:
            seq_name = root.split('/')[-1]
            annfile = f'{root}/P{p}_{seq_name}_data.pkl'
            
            if not os.path.exists(annfile):
                continue
            
            try:
                ann = pkl.load(open(annfile, 'rb'))
                if ann.get(f'emdb{split}', False):
                    sequences.append(seq_name)
                    total_count += 1
            except Exception as e:
                print(f"Warning: Failed to load {annfile}: {e}")
        
        if sequences:
            person_sequences[f'P{p}'] = sequences
    
    # Print results
    for person, seqs in sorted(person_sequences.items()):
        print(f"{person}: ({len(seqs)} sequences)")
        for seq in seqs:
            print(f"  - {seq}")
        print()
    
    print(f"{'='*80}")
    print(f"Total: {total_count} sequences in split {split}")
    print(f"{'='*80}")
    
    # Example usage
    if total_count > 0:
        first_person = list(person_sequences.keys())[0]
        first_seq = person_sequences[first_person][0]
        print(f"\n示例用法:")
        print(f"  python scripts/emdb_eval/eval_emdb_unified.py --seq {first_seq} --person {first_person} --split {split}")

if __name__ == '__main__':
    main()



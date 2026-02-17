import argparse
import sys
from src.train.train import main as train_main
from src.inference import main as inference_main

def main():
    parser = argparse.ArgumentParser(description='Brain Metastasis Detection')
    parser.add_argument('mode', choices=['train', 'inference'], help='Mode: train or inference')
    
    args = parser.parse_args()
    if args.mode == 'train':
        train_main()
    elif args.mode == 'inference':  
        inference_main()

if __name__ == '__main__':
    main()

import os
import torch
import numpy as np
import argparse
    
def calc_acc(pred_path, label_path):
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    labels = {}
    for line in lines[1:]:
        id, label = line.split(',')
        labels[id] = label
    
    with open(pred_path, 'r') as f:
        lines = f.readlines()

    correct = 0
    for line in lines[1:]:
        id, label = line.split(',')
        if id not in labels:
            print("Invalid format: different length.")
            return 0
        
        correct += int((label == labels[id]))
    
    acc = correct / (len(lines)-1)
    return acc

        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-label_path', help='path to ground truth', type=str)
    parser.add_argument('-pred_path', help='path to prediction', type=str)
    args = parser.parse_args()

    acc = calc_acc(args.pred_path, args.label_path)
    print("Accuracy: {}%".format(acc*100))
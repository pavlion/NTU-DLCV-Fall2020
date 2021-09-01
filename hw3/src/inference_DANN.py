
import os
import torch
import argparse
from torch.utils.data import DataLoader

from models import DANN
from models.SWD.SWD import SWD_test
from dataset import DigitDataset


def main(args):
    #print("Args:", args)
    torch.manual_seed(422)

    source_domain = {
        'svhn': 'mnistm',
        'usps': 'svhn',
        'mnistm': 'usps'
    }[args.target_domain]
    print("Task:", f"{source_domain} >>>>> {args.target_domain}")

    #print("Setting up model.")
    if args.model_type == 'DANN':
        model = DANN()
        model_path = os.path.join(
            "ckpt", 
            args.model_type, 
            f"{source_domain}_{args.target_domain}.pth"
        )
        model.load_state_dict(
            torch.load(model_path)
        )
        
    elif args.model_type == 'SWD':
        if source_domain == 'usps':
            model = SWD_test(model_type='usps')
        else:
            model = SWD_test()

        model_dir = os.path.join(
            "ckpt", 
            args.model_type, 
            f"{source_domain}_{args.target_domain}"
        )

        model.G.load_state_dict(
            torch.load(os.path.join(model_dir, 'G.pth'))
        )
        model.C1.load_state_dict(
            torch.load(os.path.join(model_dir, 'C1.pth'))
        )
        model.C2.load_state_dict(
            torch.load(os.path.join(model_dir, 'C2.pth'))
        )
    
    
    

    val_dset=DigitDataset(
        data_path = args.img_dir,
        label_path = None,
        mode = 'test'
    )
    test_loader=DataLoader(
        dataset = val_dset,
        batch_size = args.batch_size,
        shuffle = False,
        collate_fn = val_dset.collate_fn
    )

    ###################
    ##  Inferencing. ##
    ###################
    #print("Inferencing.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    predictions=[]
    file_names = []
    with torch.no_grad():
        for ii, batch in enumerate(test_loader):
            images=batch['image'].to(device)
            file_id=batch['file_id']
            
            if args.model_type == 'DANN':
                preds, _ = model(images, reverse=False)
            else: # model=SWD
                preds = model(images)

            predictions += preds.argmax(1).cpu().tolist()  # [batch_size]
            file_names += file_id
            #print(preds.argmax(1).cpu())
            # print("Evaluating {:3d}/{:3d}" \
            #     .format(ii, len(test_loader))+" "*10, end = '\r')
    
    #print("Dumping."+" "*10)
    dest_path = os.path.join(args.dest_path, "test_pred.csv")
    dest_path = args.dest_path
    f = open(dest_path, "w")
    f.write("image_name,label\n")
    for i in range(len(predictions)):
        f.write(f"{file_names[i]},{predictions[i]}\n")
    f.close()

    print("Finish inferencing."+" "*10+"\n")

def _parse_args():
    parser=argparse.ArgumentParser(description = 'Script to train.')
    parser.add_argument('-img_dir', '-i', default="ckpt", type=str)
    parser.add_argument('-dest_path', '-d', default="ckpt", type=str)

    parser.add_argument('-target_domain', '-t', default="mnistm",
                        type=str, choices=('mnistm', 'svhn', 'usps'))

    parser.add_argument('-batch_size', '-b', default=64, type=int)
    parser.add_argument('-model_type', default='DANN', type=str)
    parser.add_argument('-model_path', '-m', type=str)


    args=parser.parse_args()
    return args


if __name__ == '__main__':
    args=_parse_args()
    main(args)

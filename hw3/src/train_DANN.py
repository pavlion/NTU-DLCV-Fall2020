import os
import argparse
import torch
torch.manual_seed(422)
from torch.utils.data import DataLoader

from dataset import DigitDataset
from models import DANN_MODEL, DANN_Config

from models.logger import wandb

wandb.init()

def main(args):
    print("Args:", args)
    
    assert args.source_domain in DigitDataset.VALID_DOMAIN, f"source domain:{args.source_domain} is invalid."
    assert args.target_domain in DigitDataset.VALID_DOMAIN, f"target domain:{args.target_domain} is invalid."
    print(f"***Domain: {args.source_domain}>>>>>>>>>>>{args.target_domain}***\n")

    print("Setting up model" + " "*10)
    cfg = DANN_Config(
        batch_size=args.batch_size,
        epochs=args.epoch,
        learning_rate=args.lr
    )
    model = DANN_MODEL(cfg)
    print("Training config:", cfg.get_cfg_dict())

    if not args.test:
        print("Getting Datasets")
        src_train_loader, src_val_loader = get_DataLoader(args.source_domain, args.batch_size)
        trg_train_loader, trg_val_loader = get_DataLoader(args.target_domain, args.batch_size)
        train_loaders = {
            'source': src_train_loader,
            'target': trg_train_loader if not args.no_da else src_train_loader
        }
        val_loaders = {
            'source': src_val_loader,
            'target': trg_val_loader
        }

        print("Start Training!")
        working_dir = os.path.join(args.ckpt_dir, f"DANN_lr{cfg.learning_rate}_{args.source_domain}-{args.target_domain}")
        _check_dir(working_dir)
        _check_dir(os.path.join(working_dir, "gen_img"))
        model.train(train_loaders, val_loaders, dest_dir=working_dir, domain_adaption=(not args.no_da))

    print(f"***Domain: {args.source_domain} >>>>>>>>>>> {args.target_domain}***\n")
    print("Evaluate on test data:")
    #model.load_ckpt(os.path.join(args.ckpt_dir, f"DANN_lr{cfg.learning_rate}_{args.source_domain}-{args.target_domain}", f"best_loss.pth"))
    #model.load_ckpt("ckpt/DANN_lr0.034503324915917004_usps-mnistm/best_loss.pth")
    model.load_ckpt(args.ckpt_path)
    val_dset = DigitDataset(
        data_path=os.path.join('hw3_data', 'digits', args.target_domain, 'test'),
        label_path=os.path.join('hw3_data', 'digits', args.target_domain, 'test.csv'),
        mode='val'
    )
    val_loader = DataLoader(
        dataset=val_dset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=val_dset.collate_fn
    )
    acc, loss = model.evaluate(val_loader, print_std=False)
    print("Accuracy: {:.4f}% | Loss: {:.6f}".format(acc*100.0, loss))
    print("\n\n\n")
    wandb.log({"acc": acc, "loss": loss})

def get_DataLoader(domain, batch_size):

    #gray2rgb = True if domain == 'usps' else False
    gray2rgb = False
    train_dset = DigitDataset(
        data_path=os.path.join('hw3_data', 'digits', domain, 'train'),
        label_path=os.path.join('hw3_data', 'digits', domain, 'train.csv'),
        mode='train',
        gray2rgb=gray2rgb
    )
    val_dset = DigitDataset(
        data_path=os.path.join('hw3_data', 'digits', domain, 'train'),
        label_path=os.path.join('hw3_data', 'digits', domain, 'train.csv'),
        mode='val',
        gray2rgb=gray2rgb
    )

    train_loader = DataLoader(
        dataset=train_dset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_dset.collate_fn
    )
    val_loader = DataLoader(
        dataset=val_dset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=val_dset.collate_fn
    )

    return train_loader, val_loader


def _check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def _parse_args():
    parser = argparse.ArgumentParser(description='Script to train.')
    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--no_da', default=False, action='store_true', help='no domain adaption')
    parser.add_argument('-ckpt_dir', '-p', default="ckpt", type=str)
    parser.add_argument('-ckpt_path', type=str, help="path to ckpt for testing")

    parser.add_argument('-source_domain', '-s', default="usps",
                        type=str, choices=('mnistm', 'svhn', 'usps'))
    parser.add_argument('-target_domain', '-t', default="mnistm",
                        type=str, choices=('mnistm', 'svhn', 'usps'))

    parser.add_argument('-batch_size', '-b', default=64, type=int)
    parser.add_argument('-epoch', '-e', default=30, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = _parse_args()
    main(args)


import os
import argparse
import torch
torch.manual_seed(422)
from torch.utils.data import DataLoader

from dataset import DigitDataset
from models.SWD import SWD, SWD_Trainer, SWD_Config

from models.logger import wandb
#import wandb

wandb.init("sweep_dlcv3-4_u2m")

def main(args):
    print("Args:", args)
    
    assert args.source_domain in DigitDataset.VALID_DOMAIN, f"source domain:{args.source_domain} is invalid."
    assert args.target_domain in DigitDataset.VALID_DOMAIN, f"target domain:{args.target_domain} is invalid."
    print(f"***Domain: {args.source_domain}>>>>>>>>>>>{args.target_domain}***\n")
    
    working_dir = os.path.join(args.ckpt_dir, f"SWD_lr{args.lr}_{args.source_domain}-{args.target_domain}")
    _check_dir(working_dir)

    #print("Setting up model" + " "*10)
    cfg = SWD_Config(
        batch_size=args.batch_size,
        epochs=args.epoch,
        learning_rate=args.lr,
        num_filters=64,
        adapt_loss_opt='SWD',
        M=args.M,
        src_domain=args.source_domain,
        trg_domain = args.target_domain
    )
    trainer = SWD_Trainer(cfg)
    print("Training config:", cfg.get_cfg_dict())

    if not args.test:
        #print("Getting Datasets")
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

        #print("Start Training!")        
        trainer.train(train_loaders, val_loaders, dest_dir=working_dir, domain_adaption=(not args.no_da))

    #working_dir = f"ckpt/SWD_lr0.001_{args.source_domain}-{args.source_domain}"
    print(f"***Domain: {args.source_domain} >>>>>>>>>>> {args.target_domain}***")
    print("Evaluate on test data:")
    best_value = ''
    metirc_type = 'loss'
    trainer.load_ckpt(
        os.path.join(working_dir, f"best_{metirc_type}_G{best_value}.pth"),
        os.path.join(working_dir, f"best_{metirc_type}_C1{best_value}.pth"),
        os.path.join(working_dir, f"best_{metirc_type}_C2{best_value}.pth")
    )
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
    acc, loss = trainer.evaluate(val_loader, print_std=False)
    print("(Best Loss)Accuracy: {:.4f}% | Loss: {:.6f}".format(acc*100., loss) + " "*20)
    
    #dump_logs(args, cfg, acc, loss, metirc_type, trainer, working_dir)
    
    best_value = ''
    metirc_type = 'acc'
    trainer.load_ckpt(
        os.path.join(working_dir, f"best_{metirc_type}_G{best_value}.pth"),
        os.path.join(working_dir, f"best_{metirc_type}_C1{best_value}.pth"),
        os.path.join(working_dir, f"best_{metirc_type}_C2{best_value}.pth")
    )
    acc2, loss2 = trainer.evaluate(val_loader, print_std=False)
    print("(Best Acc)Accuracy: {:.4f}% | Loss: {:.6f}".format(acc2*100., loss2) + " "*20)
    print("\n\n\n")
    #dump_logs(args, cfg, acc2, loss2, metirc_type, trainer, working_dir)

    wandb.log({
        'acc': max(acc, acc2)*100.0
    })

def dump_logs(args, cfg, acc, loss, metirc_type, trainer, working_dir):
    with open(f"ckpt/best_swd_{args.source_domain}-{args.target_domain}.txt", "r") as f: 
        best_acc = float(f.read().strip())

    if best_acc < acc*100.0:
        best_acc = acc*100.0
        trainer.save_ckpt(
            os.path.join("ckpt", "SWD", f"best_{metirc_type}_G_{acc*100}.pth"),
            os.path.join("ckpt", "SWD", f"best_{metirc_type}_C1_{acc*100}.pth"),
            os.path.join("ckpt", "SWD", f"best_{metirc_type}_C2_{acc*100}.pth")
        )
        print("ckpt saved. acc =", acc*100.0)
        with open(f"ckpt/best_swd_{args.source_domain}-{args.target_domain}.txt", "w") as f: 
            f.write(str(best_acc))  
    else:
        os.remove(os.path.join(working_dir, f"best_{metirc_type}_G.pth"))
        os.remove(os.path.join(working_dir, f"best_{metirc_type}_C1.pth"))
        os.remove(os.path.join(working_dir, f"best_{metirc_type}_C2.pth"))
        
    
    with open(f"ckpt/sweep_history_swd_{args.source_domain}-{args.target_domain}.txt", "a") as fp: 
        print("\n"*2 + "="*30, file=fp)
        #print(args, file=fp)
        #print("args:", args, file=fp)
        #print("cfg:", cfg.get_cfg_dict(), file=fp)
        print(f"lr={args.lr}, M={cfg.M}, type={metirc_type}", file=fp)
        print(f"Validating: acc={trainer.max_acc*100:.4f}%, loss={trainer.min_loss:.6f}", file=fp)
        print(f"Evaluating: acc={acc*100:.4f}%, loss={loss:.6f}", file=fp)
        print("="*30 + "\n", file=fp)



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
    parser.add_argument('-batch_size', '-b', default=128, type=int)

    parser.add_argument('-source_domain', '-s', default="usps",
                        type=str, choices=('mnistm', 'svhn', 'usps'))
    parser.add_argument('-target_domain', '-t', default="mnistm",
                        type=str, choices=('mnistm', 'svhn', 'usps'))

    
    parser.add_argument('-epoch', '-e', default=20, type=int)
    
    # params to be tuned
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--M', default=256, type=float)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = _parse_args()
    main(args)


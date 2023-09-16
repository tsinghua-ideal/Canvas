import torch
import os
import json

from base import dataset, device, models, parser, trainer, log, darts_trainer, entrans_trainer


if __name__ == '__main__':
    # Get arguments.
    logger = log.get_logger()
    args = parser.arg_parse()

    # Check available devices and set distributed.
    device.initialize(args)
    for i in range(3):
        # match i:
        #     case 0:
        #         args.lr = 0.0375
            # case 1:
            #     args.lr = 0.0025
            # case 2:
            #     args.lr = 0.001
            
            
       
        # Training utils.
        model = models.get_model(args, search_mode=False)
        train_loader, eval_loader = dataset.get_loaders(args)

        # Load checkpoint.
        if args.canvas_load_checkpoint:
            logger.info(f'Loading checkpoint from {args.canvas_load_checkpoint}')
            checkpoint = torch.load(args.canvas_load_checkpoint)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        
        # Start to train!
        all_train_eval_metrics = trainer.train(args, model=model, train_loader=train_loader, eval_loader=eval_loader)
        
        # Save into logging directory
        path = args.canvas_log_dir
        
        with open(os.path.join(path, f'van_lr_{args.lr}_metrics.json'), 'w') as file:
                        json.dump({'args': vars(args), 
                                'train_metrics': all_train_eval_metrics["all_train_metrics"], 'eval_metrics': all_train_eval_metrics["all_eval_metrics"]
                                },
                                fp=file, sort_keys=True, indent=4, separators=(',', ':'), ensure_ascii=False)

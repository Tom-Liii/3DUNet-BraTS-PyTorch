# 3D U-NET BraTS PyTorch

This repo is a PyTorch implementation of 3D U-Net and Multi-encoder 3D U-Net for Multimodal MRI Brain Tumor Segmentation (BraTS 2021).

## Arguments
- Important arguments
    - `--expdir`: the directory of output
    - `--data_root`: the root of data directory
- Example command: 
```bash
python train_brats2021.py --dataset brats2023 --data_root ../brats2023/ --cases_split ./data/split/brats2023_split_fold0.csv --epochs 3 --eval_freq 1
```
```bash
usage: train_brats2021.py [-h] [--comment COMMENT] [--gpus GPUS [GPUS ...]] [--seed SEED] [--num_workers NUM_WORKERS] [--amp]
                          [--data_parallel] [--exp_dir EXP_DIR] [--save_freq SAVE_FREQ] [--print_freq PRINT_FREQ]
                          [--dataset {brats2021,brats2018}] [--data_root DATA_ROOT] [--cases_split CASES_SPLIT]
                          [--input_channels INPUT_CHANNELS] [--patch_size PATCH_SIZE] [--pos_ratio POS_RATIO]
                          [--neg_ratio NEG_RATIO] [--epochs EPOCHS] [--batch_size BATCH_SIZE] [--lr LR] [--optim {adam,adamw,sgd}]
                          [--beta1 M] [--beta2 M] [--weight-decay WEIGHT_DECAY] [--scheduler {warmup_cosine,cosine,step,poly,none}]
                          [--warmup_epochs WARMUP_EPOCHS] [--milestones MILESTONES [MILESTONES ...]] [--lr_gamma LR_GAMMA]
                          [--clip_grad] [--unet_arch {unet,multiencoder_unet,unetr}] [--block {plain,res}]
                          [--channels_list CHANNELS_LIST [CHANNELS_LIST ...]] [--kernel_size KERNEL_SIZE]
                          [--dropout_prob DROPOUT_PROB] [--norm {instance,batch,group}] [--num_classes NUM_CLASSES]
                          [--weight_path WEIGHT_PATH] [--deep_supervision] [--ds_layer DS_LAYER] [--save_model] [--save_pred]
                          [--eval_freq EVAL_FREQ] [--infer_batch_size INFER_BATCH_SIZE] [--patch_overlap PATCH_OVERLAP]
                          [--sw_batch_size SW_BATCH_SIZE] [--sliding_window_mode {constant,gaussian}]

optional arguments:
  -h, --help            show this help message and exit
  --comment COMMENT     save comment
  --gpus GPUS [GPUS ...]
  --seed SEED
  --num_workers NUM_WORKERS
                        number of workers to load data
  --amp                 using mixed precision
  --data_parallel       using data parallel
  --exp_dir EXP_DIR     experiment dir
  --save_freq SAVE_FREQ
                        model save frequency (epoch)
  --print_freq PRINT_FREQ
                        print frequency (iteration)
  --dataset {brats2021,brats2018}
                        dataset hint
  --data_root DATA_ROOT
                        root dir of dataset
  --cases_split CASES_SPLIT
                        name & split
  --input_channels INPUT_CHANNELS, --n_views INPUT_CHANNELS
                        #channels of input data, equal to #encoders in multiencoder unet and#view in multiview contrastive learning
  --patch_size PATCH_SIZE
                        patch size
  --pos_ratio POS_RATIO
                        prob of picking positive patch (center in foreground)
  --neg_ratio NEG_RATIO
                        prob of picking negative patch (center in background)
  --epochs EPOCHS
  --batch_size BATCH_SIZE
  --lr LR               learning rate
  --optim {adam,adamw,sgd}
                        optimizer
  --beta1 M             momentum for sgd, beta1 for adam
  --beta2 M             beta2 for adam
  --weight-decay WEIGHT_DECAY, --wd WEIGHT_DECAY
                        weight decay
  --scheduler {warmup_cosine,cosine,step,poly,none}
                        scheduler
  --warmup_epochs WARMUP_EPOCHS
                        warm up epochs
  --milestones MILESTONES [MILESTONES ...]
                        milestones for multistep decay
  --lr_gamma LR_GAMMA   decay factor for multistep decay
  --clip_grad           whether to clip gradient
  --unet_arch {unet,multiencoder_unet,unetr}
                        Architecuture of the U-Net
  --block {plain,res}   Type of convolution block
  --channels_list CHANNELS_LIST [CHANNELS_LIST ...]
                        #channels of every levels of decoder in a top-down order
  --kernel_size KERNEL_SIZE
                        size of conv kernels
  --dropout_prob DROPOUT_PROB
                        prob of dropout
  --norm {instance,batch,group}
                        type of norm
  --num_classes NUM_CLASSES
                        number of predicted classs
  --weight_path WEIGHT_PATH
                        path to pretrained encoder or decoder weight, None for train-from-scratch
  --deep_supervision    whether use deep supervision
  --ds_layer DS_LAYER   last n layer to use deep supervision
  --save_model          whether save model state
  --save_pred           whether save individual prediction
  --eval_freq EVAL_FREQ
                        eval frequency
  --infer_batch_size INFER_BATCH_SIZE
                        batchsize for inference
  --patch_overlap PATCH_OVERLAP
                        overlap ratio between patches
  --sw_batch_size SW_BATCH_SIZE
                        sliding window batch size
  --sliding_window_mode {constant,gaussian}
                        sliding window importance map mode
```
import pandas as pd

ann_files = pd.read_csv('data.csv')

class Config():
    weight_decay = 0
    lr = 1e-4
    opt = 'adam' #'lookahead_adam' to use `lookahead`
    momentum = 0.9
    model = 'resnext50_32x4d'
    eval_metric = 'loss'
    clip_grad = 10.0
    clip_mode = "norm"
    log_interval = 50
    save_images = True
    device = 'cuda'
    epochs = 20
    sched = "cosine"
    min_lr = 1e-5
    warmup_lr = 0.0001
    warmup_epochs = 3
    cooldown_epochs = 10
    batch_size = 16
    world_size = 1
    local_rank = 0
    num_classes = max(ann_files['label'].tolist()) + 1
    gamma = 1.5
    weight = None
    backbone_pretrained = True
    fold = 0
    metric_score = ['f1-score', 'acc-score']
    drop_rate = 0.3
    feature_dim = 768

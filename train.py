import os
from args import parse_train_opt
from EDGE import EDGE
from accelerate.utils import set_seed

def train(opt):
    set_seed(42)
    model = EDGE(
        opt.feature_type,
        learning_rate=opt.learning_rate,
        checkpoint_path=opt.checkpoint,
        load_optim_state=opt.load_optim_state,
        use_beats_anno=opt.use_beats_anno,
        use_music_beat_feat=opt.use_music_beat_feat,
        freeze_layers=opt.freeze_layers,
        EMA=not opt.ema_off,
    )
    model.train_loop(opt)


if __name__ == "__main__":
    # os.environ["HTTPS_PROXY"] = "http://localhost:7890/"
    # os.environ["HTTP_PROXY"] = "http://localhost:7890/"

    opt = parse_train_opt()
    # opt.batch_size = 128
    # opt.checkpoint = "/Projects/Github/paper_project/EDGE/checkpoint.pt"
    # opt.load_optim_state = False
    # opt.save_interval = 1
    # opt.start_epoch = 0
    # opt.epochs = 1
    # opt.feature_type = "jukebox"
    # opt.learning_rate = 1e-5
    # opt.no_cache = True
    # opt.force_reload = True
    # opt.use_beats_anno = True
    # opt.ema_off = True

    # # #freeze layers
    # opt.freeze_layers = True

    # # # add beat onehot feature
    # opt.use_music_beat_feat = True
    train(opt)

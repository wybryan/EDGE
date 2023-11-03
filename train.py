import os
from args import parse_train_opt
from EDGE import EDGE


def train(opt):
    model = EDGE(
        opt.feature_type,
        learning_rate=opt.learning_rate,
        checkpoint_path=opt.checkpoint,
        load_optim_state=opt.load_optim_state,
        use_beats_anno=opt.use_beats_anno,
    )
    model.train_loop(opt)


if __name__ == "__main__":
    # os.environ["HTTPS_PROXY"] = "http://localhost:7890/"
    # os.environ["HTTP_PROXY"] = "http://localhost:7890/"

    opt = parse_train_opt()
    # opt.batch_size = 128
    # opt.checkpoint = "/Projects/Github/paper_project/EDGE/checkpoint.pt"
    # opt.load_optim_state = False
    # opt.save_interval = 50
    # opt.start_epoch = 0
    # opt.epochs = 100
    # opt.feature_type = "jukebox"
    # opt.learning_rate = 0.0002
    opt.no_cache = True
    opt.force_reload = True
    train(opt)

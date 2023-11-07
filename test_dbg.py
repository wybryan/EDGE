
from args import parse_test_opt
from EDGE import EDGE

data_path = "data/"
exp_name = "exp17"

opt = parse_test_opt()
opt.data_path = data_path
opt.no_cache = True
opt.force_reload = True
opt.use_beats_anno = False




for i in range(10):
    epoch_no = i + 1
    ckpt = f"/Projects/Github/paper_project/EDGE/runs/train/{exp_name}/weights/train-{epoch_no}.pt"
    out_dir = f"eval/{exp_name}/beats_on_motion_{epoch_no}e"

    model = EDGE(
        feature_type=opt.feature_type,
        checkpoint_path=ckpt,
        EMA=False,
    )
    model.eval()
    model.validate_loop(opt, out_dir)
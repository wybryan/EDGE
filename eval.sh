#!/usr/bin/bash
cd eval
# FT only on beat loss + beat feature loss for 1 epoch of 20 samples
# infer without beat feature input, PFC=0.417
python eval_pfc.py --motion_path exp70/beats_on_motion_1e/
# FT only on beat loss + beat feature loss for 1 epoch of 20 samples
# infer with beat feature input, PFC=0.580
python eval_pfc.py --motion_path exp71/beats_on_motion_1e/
# FT only on beat loss for 1 epoch of 20 samples, PFC=0.598
python eval_pfc.py --motion_path exp47/beats_on_motion_1e/
# FT only on beat loss + beat feature loss for 1 epoch of 100 samples, PFC=0.628
python eval_pfc.py --motion_path exp72/beats_on_motion_1e/
# FT only on beat loss for 1 epoch of 100 samples, PFC=0.717
python eval_pfc.py --motion_path exp33/beats_on_motion_1e/
# baseline: PFC=1.048
python eval_pfc.py --motion_path beats_off_motion/
cd ..
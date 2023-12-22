#!/usr/bin/bash
cd eval

# FT only on beat loss + beat feature loss for 1 epoch of ~700 samples
# python eval_pfc.py --motion_path full_700/beats_on_motion_1e/

# 课程学习 -50-sample (3) -> 20-sample (2)
python eval_pfc.py --motion_path exp78/beats_on_motion_1e/
python eval_pfc.py --motion_path exp79/beats_on_motion_1e/
python eval_pfc.py --motion_path exp80/beats_on_motion_1e/
python eval_pfc.py --motion_path exp81/beats_on_motion_1e/
python eval_pfc.py --motion_path exp82/beats_on_motion_1e/


# # FT only on beat loss + beat feature loss for 1 epoch of 20 samples
# 课程学习 - 20-sample
python eval_pfc.py --motion_path easy/beats_on_motion_1e/
python eval_pfc.py --motion_path normal/beats_on_motion_1e/
python eval_pfc.py --motion_path hard/beats_on_motion_1e/

# python eval_pfc.py --motion_path exp74/beats_on_motion_1e/

# infer without beat feature input, PFC=0.417
# python eval_pfc.py --motion_path exp73/beats_on_motion_1e/
# python eval_pfc.py --motion_path exp70/beats_on_motion_1e/
# # FT only on beat loss + beat feature loss for 1 epoch of 20 samples
# # infer with beat feature input, PFC=0.580
# python eval_pfc.py --motion_path exp71/beats_on_motion_1e/
# # FT only on beat loss for 1 epoch of 20 samples, PFC=0.598
# python eval_pfc.py --motion_path exp47/beats_on_motion_1e/
# # FT only on beat loss + beat feature loss for 1 epoch of 100 samples, PFC=0.628
# python eval_pfc.py --motion_path exp72/beats_on_motion_1e/
# # FT only on beat loss for 1 epoch of 100 samples, PFC=0.717
# python eval_pfc.py --motion_path exp33/beats_on_motion_1e/
# # baseline: PFC=1.048
# python eval_pfc.py --motion_path beats_off_motion/
cd ..
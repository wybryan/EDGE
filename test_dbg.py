
import glob
import os
from vis import compute_BeatAlignScore
from elosports.elo import Elo
import random
import numpy as np

ref = "EDGE"
eloLeague = Elo(k = 20, homefield=0)
eloLeague.addPlayer("DGFMB")
eloLeague.addPlayer(ref)

def sim_elo(winner, loser, win_rate_base, std_var):
    winner_elos = []
    loser_elos = []

    for num_sim in range(10):
        eloLeague = Elo(k=20, homefield=0)
        eloLeague.addPlayer(winner)
        eloLeague.addPlayer(loser)

        mu = win_rate_base
        sigma = std_var / 1.5
        num_win = round(random.gauss(mu, sigma))

        num_matches = [1] * num_win + (100 - num_win) * [0]
        random.shuffle(num_matches)
        for v in num_matches:
            if v == 1:
                eloLeague.gameOver(winner=winner, loser=loser, winnerHome=False)
            else:
                eloLeague.gameOver(winner=loser, loser=winner, winnerHome=False)
        win_elo = eloLeague.ratingDict[winner]
        lose_elo = eloLeague.ratingDict[loser]
        winner_elos.append(win_elo)
        loser_elos.append(lose_elo)
    
    win_elo_avg = np.mean(winner_elos)
    loser_elo_avg = np.mean(loser_elos)
    return win_elo_avg, loser_elo_avg

def sim_elos(winner, losers, base_ratings, win_rate_bases, std_vars):
    winner_elos = []
    loser_elos = {}
    [loser_elos.setdefault(k, []) for k in losers]

    for num_sim in range(100):
        eloLeague = Elo(k=20, homefield=0)
        eloLeague.addPlayer(winner)
        [eloLeague.addPlayer(loser, r) for loser, r in zip(losers, base_ratings)]

        order = [0,1,2]
        random.shuffle(order)
        for sel in order:
            mu = win_rate_bases[sel]
            sigma = std_vars[sel] / 1.5
            num_win = round(random.gauss(mu, sigma))

            num_matches = [1] * num_win + (100 - num_win) * [0]
            random.shuffle(num_matches)
            for v in num_matches:
                if v == 1:
                    eloLeague.gameOver(winner=winner, loser=losers[sel], winnerHome=False)
                else:
                    eloLeague.gameOver(winner=losers[sel], loser=winner, winnerHome=False)
            win_elo = eloLeague.ratingDict[winner]
            lose_elo = eloLeague.ratingDict[losers[sel]]
            winner_elos.append(win_elo)
            loser_elos[losers[sel]].append(lose_elo)
    
    loser_elo_avg_list = []
    win_elo_avg = np.mean(winner_elos)
    for k, v in loser_elos.items():
        loser_elo_avg = np.mean(loser_elos[k])
        loser_elo_avg_list.append(loser_elo_avg)
    return win_elo_avg, loser_elo_avg_list

print(sim_elos("DGFMB", ["EDGE","Bailando", "FACT"], [1752, 1397, 1325], [56, 92, 90], [0.032, 0.048, 0.052]))

print(sim_elo("DGFMB", "Bailando", 92, 0.04))
print(sim_elo("DGFMB", "EDGE", 56, 0.03))


for i in range(56):
    eloLeague.gameOver(winner = "DGFMB", loser = ref, winnerHome=False)
print(eloLeague.ratingDict['DGFMB'], eloLeague.ratingDict[ref])

ref = "Bailando"
# eloLeague = Elo(k = 20, homefield=0)
# eloLeague.addPlayer("DGFMB")
eloLeague.addPlayer(ref)
for i in range(92):
    eloLeague.gameOver(winner = "DGFMB", loser = ref, winnerHome=False)
print(eloLeague.ratingDict['DGFMB'], eloLeague.ratingDict[ref])

ref = "FACT"
# eloLeague = Elo(k = 20, homefield=0)
# eloLeague.addPlayer("DGFMB")
eloLeague.addPlayer(ref)
for i in range(90):
    eloLeague.gameOver(winner = "DGFMB", loser = ref, winnerHome=False)
print(eloLeague.ratingDict['DGFMB'], eloLeague.ratingDict[ref])


gt_dir = "/Projects/Github/paper_project/EDGE/data/test"
motions = sorted(glob.glob(f"{gt_dir}/motions/*.pkl"))
musics = sorted(glob.glob(f"{gt_dir}/wavs/*.wav"))
beat_align_score = compute_BeatAlignScore(motions, musics)
print(beat_align_score)

our_model_dir = "/Projects/Github/paper_project/EDGE/eval/beats_off_motion"
motions = sorted(glob.glob(f"{our_model_dir}/*.pkl"))
musics = sorted(glob.glob(f"{our_model_dir}/*.wav"))
beat_align_score = compute_BeatAlignScore(motions, musics)
print(beat_align_score)

our_model_dir = "/Projects/Github/paper_project/EDGE/eval/hard/beats_on_motion_1e"
motions = sorted(glob.glob(f"{our_model_dir}/*.pkl"))
musics = sorted(glob.glob(f"{our_model_dir}/*.wav"))
beat_align_score = compute_BeatAlignScore(motions, musics)
print(beat_align_score)

beat_align_score = compute_BeatAlignScore(
    motion_pkl_path="/Projects/Github/paper_project/EDGE/eval/hard/beats_on_motion_1e/test_gBR_sBM_cAll_d04_mBR0_ch02.pkl",
    music_wav_path="/Projects/Github/paper_project/EDGE/eval/hard/beats_on_motion_1e/test_gBR_sBM_cAll_d04_mBR0_ch02.wav",
)
print(beat_align_score)

beat_align_score = compute_BeatAlignScore(
    motion_pkl_path="/Projects/Github/paper_project/EDGE/eval/beats_off_motion/test_gBR_sBM_cAll_d04_mBR0_ch02.pkl",
    music_wav_path="/Projects/Github/paper_project/EDGE/eval/beats_off_motion/test_gBR_sBM_cAll_d04_mBR0_ch02.wav",
)
print(beat_align_score)


from args import parse_test_opt
from EDGE import EDGE

data_path = "data/"
exp_name = "exp68"

opt = parse_test_opt()
opt.data_path = data_path
opt.no_cache = True
opt.force_reload = True
opt.use_beats_anno = False




for i in range(1):
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
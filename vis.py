import os
from pathlib import Path
from tempfile import TemporaryDirectory
import pickle

import librosa as lr
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
from matplotlib import cm
from matplotlib.colors import ListedColormap
from pytorch3d.transforms import (axis_angle_to_quaternion, quaternion_apply,
                                  quaternion_multiply, quaternion_to_axis_angle,
                                  RotateAxisAngle)
from tqdm import tqdm

from dataset.quaternion import ax_from_6v

smpl_joints = [
    "root",  # 0
    "lhip",  # 1
    "rhip",  # 2
    "belly", # 3
    "lknee", # 4
    "rknee", # 5
    "spine", # 6
    "lankle",# 7
    "rankle",# 8
    "chest", # 9
    "ltoes", # 10
    "rtoes", # 11
    "neck",  # 12
    "linshoulder", # 13
    "rinshoulder", # 14
    "head", # 15
    "lshoulder", # 16
    "rshoulder",  # 17
    "lelbow", # 18
    "relbow",  # 19
    "lwrist", # 20
    "rwrist", # 21
    "lhand", # 22
    "rhand", # 23
]

smpl_parents = [
    -1,
    0,
    0,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    9,
    9,
    12,
    13,
    14,
    16,
    17,
    18,
    19,
    20,
    21,
]

smpl_offsets = [
    [0.0, 0.0, 0.0],
    [0.05858135, -0.08228004, -0.01766408],
    [-0.06030973, -0.09051332, -0.01354254],
    [0.00443945, 0.12440352, -0.03838522],
    [0.04345142, -0.38646945, 0.008037],
    [-0.04325663, -0.38368791, -0.00484304],
    [0.00448844, 0.1379564, 0.02682033],
    [-0.01479032, -0.42687458, -0.037428],
    [0.01905555, -0.4200455, -0.03456167],
    [-0.00226458, 0.05603239, 0.00285505],
    [0.04105436, -0.06028581, 0.12204243],
    [-0.03483987, -0.06210566, 0.13032329],
    [-0.0133902, 0.21163553, -0.03346758],
    [0.07170245, 0.11399969, -0.01889817],
    [-0.08295366, 0.11247234, -0.02370739],
    [0.01011321, 0.08893734, 0.05040987],
    [0.12292141, 0.04520509, -0.019046],
    [-0.11322832, 0.04685326, -0.00847207],
    [0.2553319, -0.01564902, -0.02294649],
    [-0.26012748, -0.01436928, -0.03126873],
    [0.26570925, 0.01269811, -0.00737473],
    [-0.26910836, 0.00679372, -0.00602676],
    [0.08669055, -0.01063603, -0.01559429],
    [-0.0887537, -0.00865157, -0.01010708],
]


def set_line_data_3d(line, x):
    line.set_data(x[:, :2].T)
    line.set_3d_properties(x[:, 2])


def set_scatter_data_3d(scat, x, c):
    scat.set_offsets(x[:, :2])
    scat.set_3d_properties(x[:, 2], "z")
    scat.set_facecolors([c])


def get_axrange(poses):
    pose = poses[0]
    x_min = pose[:, 0].min()
    x_max = pose[:, 0].max()

    y_min = pose[:, 1].min()
    y_max = pose[:, 1].max()

    z_min = pose[:, 2].min()
    z_max = pose[:, 2].max()

    xdiff = x_max - x_min
    ydiff = y_max - y_min
    zdiff = z_max - z_min

    biggestdiff = max([xdiff, ydiff, zdiff])
    return biggestdiff


def plot_single_pose(num, poses, lines, ax, axrange, scat, contact):
    pose = poses[num]
    static = contact[num]
    indices = [7, 8, 10, 11]

    for i, (point, idx) in enumerate(zip(scat, indices)):
        position = pose[idx : idx + 1]
        color = "r" if static[i] else "g"
        set_scatter_data_3d(point, position, color)

    for i, (p, line) in enumerate(zip(smpl_parents, lines)):
        # don't plot root
        if i == 0:
            continue
        # stack to create a line
        data = np.stack((pose[i], pose[p]), axis=0)
        set_line_data_3d(line, data)

    if num == 0:
        if isinstance(axrange, int):
            axrange = (axrange, axrange, axrange)
        xcenter, ycenter, zcenter = 0, 0, 2.5
        stepx, stepy, stepz = axrange[0] / 2, axrange[1] / 2, axrange[2] / 2

        x_min, x_max = xcenter - stepx, xcenter + stepx
        y_min, y_max = ycenter - stepy, ycenter + stepy
        z_min, z_max = zcenter - stepz, zcenter + stepz

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)


def skeleton_render(
    poses,
    epoch=0,
    out="renders",
    name="",
    sound=True,
    stitch=False,
    sound_folder="ood_sliced",
    contact=None,
    render=True,
    fps=30,
):
    if stitch:
        # otherwise it crushes
        render = False

    if render:
        # generate the pose with FK
        Path(out).mkdir(parents=True, exist_ok=True)
        num_steps = poses.shape[0]
        
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        
        point = np.array([0, 0, 1])
        normal = np.array([0, 0, 1])
        d = -point.dot(normal)
        xx, yy = np.meshgrid(np.linspace(-1.5, 1.5, 2), np.linspace(-1.5, 1.5, 2))
        z = (-normal[0] * xx - normal[1] * yy - d) * 1.0 / normal[2]
        # plot the plane
        ax.plot_surface(xx, yy, z, zorder=-11, cmap=cm.twilight)
        # Create lines initially without data
        lines = [
            ax.plot([], [], [], zorder=10, linewidth=1.5)[0]
            for _ in smpl_parents
        ]
        scat = [
            ax.scatter([], [], [], zorder=10, s=0, cmap=ListedColormap(["r", "g", "b"]))
            for _ in range(4)
        ]
        axrange = 3

        # create contact labels
        feet = poses[:, (7, 8, 10, 11)]
        feetv = np.zeros(feet.shape[:2])
        feetv[:-1] = np.linalg.norm(feet[1:] - feet[:-1], axis=-1)
        if contact is None:
            contact = feetv < 0.01
        else:
            contact = contact > 0.95

        # Creating the Animation object
        anim = animation.FuncAnimation(
            fig,
            plot_single_pose,
            num_steps,
            fargs=(poses, lines, ax, axrange, scat, contact),
            interval=1000 // fps,
        )
    if sound:
        # make a temporary directory to save the intermediate gif in
        if render:
            # temp_dir = TemporaryDirectory()
            # gifname = os.path.join(temp_dir.name, f"{epoch}.gif")
            # anim.save(gifname)
            
            # actually save the gif
            path = os.path.normpath(name)
            pathparts = path.split(os.sep)
            gifname = os.path.join(out, f"{pathparts[-1][:-4]}.gif")
            anim.save(gifname, savefig_kwargs={"transparent": True, "facecolor": "none"}, fps=fps)

        # stitch wavs
        if stitch:
            assert type(name) == list  # must be a list of names to do stitching
            temp_dir = TemporaryDirectory()
            name_ = [os.path.splitext(x)[0] + ".wav" for x in name]
            audio, sr = lr.load(name_[0], sr=None)
            ll, half = len(audio), len(audio) // 2
            total_wav = np.zeros(ll + half * (len(name_) - 1))
            total_wav[:ll] = audio
            idx = ll
            for n_ in name_[1:]:
                audio, sr = lr.load(n_, sr=None)
                total_wav[idx : idx + half] = audio[half:]
                idx += half
            # save a dummy spliced audio
            audioname = f"{temp_dir.name}/tempsound.wav" if render else os.path.join(out, f'{epoch}_{"_".join(os.path.splitext(os.path.basename(name[0]))[0].split("_")[:-1])}.wav')
            sf.write(audioname, total_wav, sr)
            outname = os.path.join(
                out,
                f'{epoch}_{"_".join(os.path.splitext(os.path.basename(name[0]))[0].split("_")[:-1])}.mp4',
            )
        else:
            assert type(name) == str
            assert name != "", "Must provide an audio filename"
            audioname = name
            outname = os.path.join(
                out, f"{epoch}_{os.path.splitext(os.path.basename(name))[0]}.mp4"
            )
        if render:
            out = os.system(
                f"ffmpeg -loglevel error -stream_loop 0 -y -r {fps} -i {gifname} -i {audioname} -shortest -c:v libx264 -crf 26 -c:a aac -q:a 4 {outname}"
            )
    else:
        if render:
            # actually save the gif
            path = os.path.normpath(name)
            pathparts = path.split(os.sep)
            gifname = os.path.join(out, f"{pathparts[-1][:-4]}.gif")
            anim.save(gifname, savefig_kwargs={"transparent": True, "facecolor": "none"}, fps=fps)
    plt.close()


class SMPLSkeleton:
    def __init__(
        self, device=None,
    ):
        offsets = smpl_offsets
        parents = smpl_parents
        assert len(offsets) == len(parents)

        self._offsets = torch.Tensor(offsets).to(device)
        self._parents = np.array(parents)
        self._compute_metadata()

    def _compute_metadata(self):
        self._has_children = np.zeros(len(self._parents)).astype(bool)
        for i, parent in enumerate(self._parents):
            if parent != -1:
                self._has_children[parent] = True

        self._children = []
        for i, parent in enumerate(self._parents):
            self._children.append([])
        for i, parent in enumerate(self._parents):
            if parent != -1:
                self._children[parent].append(i)

    def forward(self, rotations, root_positions):
        """
        Perform forward kinematics using the given trajectory and local rotations.
        Arguments (where N = batch size, L = sequence length, J = number of joints):
         -- rotations: (N, L, J, 3) tensor of axis-angle rotations describing the local rotations of each joint.
         -- root_positions: (N, L, 3) tensor describing the root joint positions.
        """
        assert len(rotations.shape) == 4
        assert len(root_positions.shape) == 3
        # transform from axis angle to quaternion
        rotations = axis_angle_to_quaternion(rotations)

        positions_world = []
        rotations_world = []

        expanded_offsets = self._offsets.expand(
            rotations.shape[0],
            rotations.shape[1],
            self._offsets.shape[0],
            self._offsets.shape[1],
        )

        # Parallelize along the batch and time dimensions
        for i in range(self._offsets.shape[0]):
            if self._parents[i] == -1:
                positions_world.append(root_positions)
                rotations_world.append(rotations[:, :, 0])
            else:
                positions_world.append(
                    quaternion_apply(
                        rotations_world[self._parents[i]], expanded_offsets[:, :, i]
                    )
                    + positions_world[self._parents[i]]
                )
                if self._has_children[i]:
                    rotations_world.append(
                        quaternion_multiply(
                            rotations_world[self._parents[i]], rotations[:, :, i]
                        )
                    )
                else:
                    # This joint is a terminal node -> it would be useless to compute the transformation
                    rotations_world.append(None)

        return torch.stack(positions_world, dim=3).permute(0, 1, 3, 2)


def visualize_data(
    motion_file_or_motion_data,
    wav_file=None,
    fps_override=None,
    render_out_dir="debug/",
    render_gif_fname="vis.gif",
):
    def process_dataset(root_pos, local_q):
        # FK skeleton
        smpl = SMPLSkeleton()
        # to Tensor
        root_pos = torch.Tensor(root_pos)
        local_q = torch.Tensor(local_q)
        # to ax
        bs, sq, c = local_q.shape
        local_q = local_q.reshape((bs, sq, -1, 3))

        # AISTPP dataset comes y-up - rotate to z-up to standardize against the pretrain dataset
        root_q = local_q[:, :, :1, :]  # sequence x 1 x 3
        root_q_quat = axis_angle_to_quaternion(root_q)
        rotation = torch.Tensor(
            [0.7071068, 0.7071068, 0, 0]
        )  # 90 degrees about the x axis
        root_q_quat = quaternion_multiply(rotation, root_q_quat)
        root_q = quaternion_to_axis_angle(root_q_quat)
        local_q[:, :, :1, :] = root_q

        # don't forget to rotate the root position too 😩
        pos_rotation = RotateAxisAngle(90, axis="X", degrees=True)
        root_pos = pos_rotation.transform_points(
            root_pos
        )  # basically (y, z) -> (-z, y), expressed as a rotation for readability

        # do FK
        positions = smpl.forward(local_q, root_pos)  # batch x sequence x 24 x 3
        feet = positions[:, :, (7, 8, 10, 11)]
        feetv = torch.zeros(feet.shape[:3])
        feetv[:, :-1] = (feet[:, 1:] - feet[:, :-1]).norm(dim=-1)
        contacts = (feetv < 0.01).to(local_q)  # cast to right dtype

        return positions, contacts

    if isinstance(motion_file_or_motion_data, str):
        motion_file = motion_file_or_motion_data
        with open(motion_file, "rb") as f:
            motion = pickle.load(f)

        if "scale" in motion.keys():
            # this is AST++ dataset
            # hardcoded in skeleton_render
            raw_fps = 60
            if fps_override is not None:
                raw_fps = fps_override

            pos, q = motion["pos"], motion["q"]
            scale = motion["scale"][0]
            
            # normalize root position
            pos /= scale
            
            pos = np.expand_dims(pos, 0)
            q = np.expand_dims(q, 0)
            poses, contacts = process_dataset(pos, q)

            if wav_file is not None:
                render_gif_fname = wav_file

            skeleton_render(
                poses[0],
                contact=contacts[0],
                epoch="",
                out=render_out_dir,
                name=render_gif_fname,
                sound=True if wav_file is not None else False,
                stitch=False,
                render=True,
                fps=raw_fps,
            )
        elif "full_pose" in motion.keys():
            # this is model generated motion pickle
            raw_fps = 30
            if fps_override is not None:
                raw_fps = fps_override
            
            if wav_file is not None:
                render_gif_fname = wav_file

            poses = motion["full_pose"]
            if len(poses.shape) < 4:
                poses = np.expand_dims(poses, 0)

            skeleton_render(
                poses[0],
                contact=None,
                epoch="",
                out=render_out_dir,
                name=render_gif_fname,
                sound=True if wav_file is not None else False,
                stitch=False,
                render=True,
                fps=raw_fps,
            )
        else:
            raise NotImplementedError
    elif isinstance(motion_file_or_motion_data, torch.Tensor):
        x = motion_file_or_motion_data
        if len(x.shape) < 3:
            x = x.unsqueeze(0)

        if x.shape[2] == 151:
            contacts, x = torch.split(
                x, (4, x.shape[2] - 4), dim=2
            )
        else:
            contacts = None

        # do the FK all at once
        b, s, c = x.shape
        pos = x[:, :, :3]  # np.zeros((sample.shape[0], 3))
        q = x[:, :, 3:].reshape(b, s, 24, 6)
        # go 6d to ax
        q = ax_from_6v(q)

        poses = SMPLSkeleton().forward(q, pos).detach().cpu().numpy()
        contacts = (
            contacts.detach().cpu().numpy()
            if contacts is not None
            else None
        )

        # hardcoded in skeleton_render
        raw_fps = 30
        if fps_override is not None:
            raw_fps = fps_override

        if wav_file is not None:
            render_gif_fname = wav_file

        skeleton_render(
            poses[0],
            contact=contacts[0],
            epoch="",
            out=render_out_dir,
            name=render_gif_fname,
            sound=True if wav_file is not None else False,
            stitch=False,
            render=True,
            fps=raw_fps,
        )
    else:
        raise NotImplementedError

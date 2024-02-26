from argparse import ArgumentParser
import os
from os import makedirs
import numpy as np
import torch
import torchvision

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from scene import Scene2 as Scene

class DummyCamera:
    def __init__(self,
                 R: np.ndarray, T: np.ndarray, FoVx: float, FoVy: float, W: int, H: int,
                 trans: np.ndarray = np.array([0, 0, 0]), scale: float = 1.0):
        # camera real world coords and rotation
        self.R = R
        self.T = T

        # camera parameters
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.zfar = 100.0
        self.znear = 0.01
        self.image_width = W
        self.image_height = H
        self.trans = trans
        self.scale = scale

        # do the projection
        self.world_view_transform = torch.tensor(self.get_world_2_view()).transpose(0, 1).cuda()
        self.projection_matrix = self.get_projection_matrix().transpose(0, 1).cuda()
        self.full_proj_transform = self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))
        self.full_proj_transform = self.full_proj_transform.squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def get_world_2_view(self):
        Rt = np.zeros((4, 4))
        Rt[:3, :3] = self.R.transpose()
        Rt[:3, 3] = self.T
        Rt[3, 3] = 1.0
        return np.float32(Rt)

    def get_projection_matrix(self):
        tan_half_fov_x = np.tan((self.FoVx / 2))
        tan_half_fov_y = np.tan((self.FoVy / 2))

        top = tan_half_fov_y * self.znear
        bottom = -top
        right = tan_half_fov_x * self.znear
        left = -right

        z_sign = 1.0
        P = torch.zeros(4, 4)
        P[0, 0] = 2.0 * self.znear / (right - left)
        P[1, 1] = 2.0 * self.znear / (top - bottom)
        P[0, 2] = (right + left) / (right - left)
        P[1, 2] = (top + bottom) / (top - bottom)
        P[3, 2] = z_sign
        P[2, 2] = z_sign * self.zfar / (self.zfar - self.znear)
        P[2, 3] = -(self.zfar * self.znear) / (self.zfar - self.znear)
        return P


def make_rotation(rotx: float, roty: float, rotz: float) -> np.ndarray:
    Rx = np.array(object=[[1, 0, 0],
                          [0, np.cos(rotx), -np.sin(rotx)],
                          [0, np.sin(rotx), np.cos(rotx)]], dtype=np.float32)
    Ry = np.array(object=[[np.cos(roty), 0, np.sin(roty)],
                          [0, 1, 0],
                          [-np.sin(roty), 0, np.cos(roty)]], dtype=np.float32)
    Rz = np.array(object=[[np.cos(rotz), -np.sin(rotz), 0],
                          [np.sin(rotz), np.cos(rotz), 0],
                          [0, 0, 1]], dtype=np.float32)
    R = (Rz @ Ry @ Rx).astype(dtype=np.float32)
    return R


class DummyPipeline:
    convert_SHs_python = False
    compute_cov3D_python = False
    debug = False

parser = ArgumentParser(description="Testing script parameters")
model = ModelParams(parser, sentinel=True)
args = get_combined_args(parser)
print("Rendering " + args.model_path)

iteration = -1
model = model.extract(args)
gaussians = GaussianModel(model.sh_degree)
scene = Scene(model, gaussians, load_iteration=iteration, shuffle=False)

bg_color = [1, 1, 1] if model.white_background else [0, 0, 0]
background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
render_path = os.path.join(model.model_path, "custom", "ours_{}".format(iteration), "renders")
makedirs(render_path, exist_ok=True)


rotx = np.array([-20]) * np.pi / 180
roty = np.array([165]) * np.pi / 180  # need to negate to match o3d
rotz = np.array([0]) * np.pi / 180
R = make_rotation(rotx, roty, rotz)

tx = np.array([.5])
ty = np.array([-1])
tz = np.array([5])

T = np.array([tx, ty, tz])

FoVx, FoVy = 1.0, 1.0
W, H = 1500, 1500
mycam = DummyCamera(R, T, FoVx, FoVy, W=W, H=W)
rendering = render(mycam, gaussians, DummyPipeline(), background)["render"]
torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(0) + ".png"))

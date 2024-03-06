
from flatland.utils.rendertools import RenderTool

import PIL
import imageio
from IPython.display import Image, clear_output

import numpy as np

# Adapted for the labs code and code from flatland documentation
class RenderWrapper:
    '''Wrapper for the flatland environment to render it in a Jupyter Notebook
    Args:
        env (flatland.envs.rail_env.RailEnv): Environment to render
        real_time_render (bool): If True, the environment will be rendered in real time
        force_gif (bool): If True, calling make_gif will generate a gif from the rendered frames'''
    def __init__(self, env, real_time_render=False, force_gif=False):
        self.env_renderer = RenderTool(env, gl="PILSVG")
        self.real_time_render = real_time_render
        self.force_gif = force_gif
        self.reset()

    def reset(self):
        self.images = []

    def render(self):

        self.env_renderer.render_env()

        image = self.env_renderer.get_image()
        pil_image = PIL.Image.fromarray(image)

        if self.real_time_render :
            clear_output(wait=True)
            display(pil_image)

        if self.force_gif:
            self.images.append(pil_image)

    def make_gif(self, filename="render"):
        if self.force_gif:
            imageio.mimsave(filename + '.gif', [np.array(img) for i, img in enumerate(self.images) if i%2 == 0], duration=100, loop=0)
            return Image(open(filename + '.gif','rb').read())
import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
import collision


########################################################################################################################
########################################################################################################################


class CarrinhoEnv(gym.Env):

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }
    
########################################################################################################################

    def __init__(self,
                 comprimento=0.2, largura=0.1,
                 v_max=0.1, vp_max=0.1, phi_max=0.5, phip_max=0.5, xp_lim=0.1, yp_lim=0.1,
                 estac_paralelo=True, random_init=False, init_var=(0.0, +0.05, 0.0), target=None,
                 limite=2, maxtsteps=2000, tstep=1/60, dist_tol=0.02, screen_size=800):

        super(CarrinhoEnv, self).__init__()

        self.action_space = spaces.Box(low=np.array([-vp_max, -phip_max]),
                                       high=np.array([vp_max, phip_max]),
                                       dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([0.0, 0.0, -np.inf,
                                                          -xp_lim, -yp_lim, -np.inf]),
                                            high=np.array([limite, limite, np.inf,
                                                           xp_lim, yp_lim, np.inf]),
                                            dtype=np.float32)

        self.comprimento = comprimento
        self.largura = largura

        self.estac_paralelo = True if estac_paralelo else False
        self.random_init = random_init
        self.random_var_posx = init_var[0]
        self.random_var_posy = init_var[1]
        self.random_var_theta = init_var[2]

        self.limite = limite
        self.phi_max = phi_max
        self.phi_min = -phi_max
        self.phip_max = phip_max
        self.phip_min = -phip_max
        self.v_max = v_max
        self.v_min = -v_max
        self.vp_max = vp_max
        self.vp_min = -vp_max

        if estac_paralelo:
            self.target_x = limite / 2 if target is None else 1.0
            self.target_y = self.largura / 2 if target is None else 1.0
            self.init_notrandom_hor = np.array([(limite / 2) + (5 * comprimento / 4) + init_var[0],
                                                (3 * largura / 2) + init_var[1],
                                                0 + init_var[2],
                                                0, 0, 0])
        else:
            self.target_x = limite / 2
            self.target_y = limite - (self.comprimento / 2)
            self.init_notrandom_ver = np.array([(limite / 2) + (5 * largura / 4) + init_var[0],
                                                limite - (3 * largura / 2) + init_var[1],
                                                0 + init_var[2],
                                                0, 0, 0])

        self.tstep = tstep
        self.tsteps = 0
        self.dist_tol = dist_tol

        self.state = self.reset()
        self.x, self.y, self.theta, self.xp, self.yp, self.thetap = self.state
        self.v = 0.0
        self.vp = 0.0
        self.phi = 0.0
        self.phip = 0.0
        self.dist = np.sqrt((self.x - self.target_x) ** 2 + (self.y - self.target_y) ** 2)

        self.maxtsteps = maxtsteps
        self.viewer = None
        self.carrinho_transform = None
        self.carrinho_comp_transform = None
        self.screen_size = screen_size
        self.ppm = self.screen_size / limite

        self.carrinho_collision_shape = self.create_rect_collision_shape(self.x,
                                                                         self.y,
                                                                         self.comprimento,
                                                                         self.largura,
                                                                         self.theta)
        rect_collision_shape_bl = self.create_rect_collision_shape((limite/2) - (3 * self.comprimento / 4),
                                                                        self.largura / 2,
                                                                        self.comprimento,
                                                                        self.largura,
                                                                        0)
        rect_collision_shape_br = self.create_rect_collision_shape((limite / 2) + (3 * self.comprimento / 4),
                                                                        self.largura / 2,
                                                                        self.comprimento,
                                                                        self.largura,
                                                                        0)
        rect_collision_shape_ul = self.create_rect_collision_shape((limite / 2) + (3 * self.largura / 4),
                                                                        self.limite - self.comprimento / 2,
                                                                        self.largura,
                                                                        self.comprimento,
                                                                        0)
        rect_collision_shape_ur = self.create_rect_collision_shape((limite / 2) - (3 * self.largura / 4),
                                                                        self.limite - self.comprimento / 2,
                                                                        self.largura,
                                                                        self.comprimento,
                                                                        0)

        self.rect_collision_shape_list = [rect_collision_shape_bl, rect_collision_shape_br,
                                          rect_collision_shape_ul, rect_collision_shape_ur]

        self.collided = None

        self.info = {
            'states': [],
            'actions': [],
            'rewards': [],
            'tsteps': []
        }

        self.reward_type = 'sparse'

########################################################################################################################

    def step(self, acao):

        self.vp, self.phip = acao
        self.vp, self.phip = np.clip(acao,
                                     [self.vp_min, self.phip_min],
                                     [self.vp_max, self.phip_max])

        self.v, self.phi = np.clip([self.v + self.tstep * self.vp, self.phi + self.tstep * self.phip],
                                   [self.v_min, self.v_max],
                                   [self.v_max, self.phi_max])

        self.xp = self.v * np.cos(self.theta)
        self.yp = self.v * np.sin(self.theta)
        self.thetap = self.v * np.tan(self.phi) / self.comprimento

        self.x += self.tstep * self.xp
        self.y += self.tstep * self.yp
        self.theta += np.mod(self.tstep * self.thetap, 2 * np.pi)

        self.state = np.array([self.x, self.y, self.theta,
                               self.xp, self.yp, self.thetap])

        self.carrinho_collision_shape.pos = collision.Vector(self.x, self.y)
        self.carrinho_collision_shape.angle = self.theta

        resultados_colisoes = []
        for rect in self.rect_collision_shape_list:
            resultado_collisao = collision.collide(self.carrinho_collision_shape, rect)
            resultados_colisoes.append(resultado_collisao)
        if True in resultados_colisoes:
            self.collided = True
        else:
            self.collided = False

        self.dist = np.sqrt((self.x - self.target_x) ** 2 + (self.y - self.target_y) ** 2)

        self.tsteps += 1

        if not self.observation_space.contains(self.state) or self.collided:
            reward = -1000
            done = True

        elif self.tsteps > self.maxtsteps:
            reward = 0
            done = True

        elif self.dist <= self.dist_tol:
            reward = 1000
            done = True

        else:
            reward = -self.dist
            done = False

        self.info['states'].append(self.state)
        self.info['actions'].append(acao)
        self.info['rewards'].append(reward)
        self.info['tsteps'].append(self.tsteps)


        return self.state, reward, done, self.info

########################################################################################################################

    def reset(self):

        posxvar = self.random_var_posx
        posyvar = self.random_var_posy
        angvar = self.random_var_theta

        if self.estac_paralelo:
            if self.random_init:
                self.state = np.random.uniform(
                    low=self.init_notrandom_hor - np.array([posxvar / 2, posyvar / 2, angvar / 2, 0, 0, 0]),
                    high=self.init_notrandom_hor + np.array([posxvar / 2, posyvar / 2, angvar / 2, 0, 0, 0]))
            else:
                self.state = self.init_notrandom_hor

        else:
            if self.random_init:
                self.state = np.random.uniform(
                    low=self.init_notrandom_ver - np.array([posxvar / 2, posyvar / 2, angvar / 2, 0, 0, 0]),
                    high=self.init_notrandom_ver + np.array([posxvar / 2, posyvar / 2, angvar / 2, 0, 0, 0]))
            else:
                self.state = self.init_notrandom_ver

        self.tsteps = 0

        return self.state

########################################################################################################################

    def render(self, mode='human', close=False):

        c = self.comprimento
        l = self.largura
        lim = self.limite

        if self.viewer is None:

            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(self.screen_size, self.screen_size)
            self.viewer.set_bounds(0, self.limite, 0, self.limite)

            carrinho = rendering.make_polygon([(- c / 2, + l / 2),
                                               (- c / 2, - l / 2),
                                               (+ c / 2, - l / 2),
                                               (+ c / 2, + l / 2)],
                                              True)

            carrinho.set_color(0, 0.6, 0)

            self.carrinho_transform = rendering.Transform()
            carrinho.add_attr(self.carrinho_transform)

            self.viewer.add_geom(carrinho)

        self.carrinho_transform.set_rotation(self.theta)
        self.carrinho_transform.set_translation(self.x, self.y)

        xesq = (lim / 2) - (3 * c / 4)
        xdir = (lim / 2) + (3 * c / 4)
        self.viewer.draw_polygon([(xesq, 0), (xesq, l), (xesq - c, l), (xesq - c, 0)])
        self.viewer.draw_polygon([(xdir, 0), (xdir, l), (xdir + c, l), (xdir + c, 0)])

        xesq = (lim / 2) - (3 * l / 4)
        ydir = (lim / 2) + (3 * l / 4)
        self.viewer.draw_polygon([(xesq - l, lim - c), (xesq, lim - c), (xesq, lim), (xesq - l, lim)])
        self.viewer.draw_polygon([(xdir, lim - c), (xdir + l, lim - c), (xdir + l, lim), (xdir, lim)])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

########################################################################################################################

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

########################################################################################################################

    def create_rect_collision_shape(self, center_x, center_y, width, height, angle):

        rect_vec_center = collision.Vector(center_x, center_y)

        vertice_vec_bl = collision.Vector(- width / 2, - height / 2)
        vertice_vec_br = collision.Vector(+ width / 2, - height / 2)
        vertice_vec_ur = collision.Vector(+ width / 2, + height / 2)
        vertice_vec_ul = collision.Vector(- width / 2, + height / 2)

        vertices_list_vec = [vertice_vec_bl, vertice_vec_br, vertice_vec_ur, vertice_vec_ul]

        rect_collision_shape = collision.Poly(rect_vec_center, vertices_list_vec, angle)

        return rect_collision_shape

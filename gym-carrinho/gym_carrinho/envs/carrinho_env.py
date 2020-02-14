import gym

from gym import error, spaces, utils
from gym.utils import seeding
from gym_carrinho.envs.utils import *


class CarrinhoEnv(gym.Env):

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self,
                 modo='acel', tipo_pose='coord', obstaculo=True,
                 pesos=(0.5, 0.01, 0.01),
                 dt=1/60, tempo_max=1200,
                 pose_inic=(1, 1, 0), desvio_pose_inic=(0.5, 0.5, np.pi),
                 target_inic=(1.5, 1.5, 0), desvio_target=(2, 2, -np.pi/3), dist_target_tol=0.05, angulo=False,
                 tam_janela=800, lim_em_metros=4,
                 compr=0.2, larg=0.1,
                 max_v=0.1, max_phi=0.5, max_v_p=0.1, max_phi_p=0.5,
                 fator_larg=1, fator_compr=0.1):

        super(CarrinhoEnv, self).__init__()

        self.modo = modo
        self.tipo_pose = tipo_pose
        self.obstaculo = obstaculo
        self.pesos = pesos
        self.dt = dt
        self.tempo_max = tempo_max
        self.pose_inic = pose_inic
        self.desvio_pose_inic = desvio_pose_inic
        self.target_inic = target_inic
        self.desvio_target = desvio_target
        self.dist_target_tol = dist_target_tol
        self.angulo = angulo
        self.tam_janela = tam_janela
        self.lim_em_metros = lim_em_metros
        self.compr = compr
        self.larg = larg
        self.max_v = max_v
        self.max_phi = max_phi
        self.max_v_p = max_v_p
        self.max_phi_p = max_phi_p
        self.fator_larg = fator_larg
        self.fator_compr = fator_compr

        self.fator_larg = 100 if self.obstaculo is False else self.fator_larg
        self.fator_compr = 100 if self.obstaculo is False else self.fator_compr
        self.tempo = 0

        self.renderizador = None  #### tests
        # self.renderizador = Renderizador(tam_janela=tam_janela, lim_em_metros=lim_em_metros)
        self.juizcolisao = JuizColisao()

        self.vaga = Vaga(target_inic=target_inic, desvio_target=desvio_target, dist_target_tol=dist_target_tol,
                         compr=compr, larg=larg, fator_larg=self.fator_larg, fator_compr=self.fator_compr)

        self.carro = Carro(max_v=max_v, max_phi=max_phi, max_v_p=max_v_p, max_phi_p=max_phi_p,
                           compr=compr, larg=larg,
                           pose_inic=pose_inic, desvio_pose_inic=desvio_pose_inic, tipo_pose=tipo_pose,
                           vaga=self.vaga)

        # self.renderizador.adc_carro(self.carro)
        # self.renderizador.adc_vaga(self.vaga)

        self.juizcolisao.adic_obj(self.vaga.rect_esq)
        self.juizcolisao.adic_obj(self.vaga.rect_dir)

        self.action_space = spaces.Box(low=np.array([-max_v, -max_phi]),
                                       high=np.array([+max_v, +max_phi]),
                                       dtype=np.float32)

        if tipo_pose == 'coord':
            self.observation_space = spaces.Box(low=np.array([0, 0, -np.inf,
                                                              -max_v_p, -max_v_p, -np.inf]),
                                                high=np.array([lim_em_metros, lim_em_metros, -np.inf,
                                                               max_v_p, max_v_p, np.inf]),
                                                dtype=np.float32)

        elif tipo_pose == 'dist':
            self.observation_space = spaces.Box(low=np.array([- lim_em_metros, - lim_em_metros, -np.inf,
                                                              -max_v_p, -max_v_p, -np.inf]),
                                                high=np.array([+ lim_em_metros, + lim_em_metros, +np.inf,
                                                               max_v_p, max_v_p, np.inf]),
                                                dtype=np.float32)

        self.info = {
            'estados': [],
            'acoes': [],
            'recompensas': [],
            'dts': [],
            'v': [],
            'phi': [],
            'x': [],
            'y': [],
            'theta': [],
            'xp': [],
            'yp': [],
            'thetap': [],
            'dif_ang': None
        }

    def step(self, acao):

        v, phi = acao

        obs = self.carro.update(v=v, phi=phi, dt=self.dt, modo=self.modo)

        colidiu = self.juizcolisao.checar_colisao(self.carro.shape) if self.obstaculo else False

        self.tempo += 1

        dif_ang = None

        if colidiu:
            recomp, terminou = -2000, True
            print('colisão')

        elif self.tempo >= self.tempo_max:
            terminou = True
            recomp, lixo1, lixo2 = self.carro.pegar_recompensa(vaga=self.vaga, pesos=self.pesos, angulo=self.angulo)

        elif not self.observation_space.contains(obs):
            terminou = True
            recomp = -2000
        else:
            recomp, terminou, dif_ang = self.carro.pegar_recompensa(vaga=self.vaga, pesos=self.pesos,
                                                                    angulo=self.angulo)
            self.info['dif_ang'] = dif_ang

        self.info['estados'].append(obs)
        self.info['acoes'].append(acao)
        self.info['recompensas'].append(recomp)
        self.info['dts'].append(self.tempo)
        self.info['v'].append(v)
        self.info['phi'].append(phi)
        self.info['x'].append(self.carro.pose[0])
        self.info['y'].append(self.carro.pose[1])
        self.info['theta'].append(self.carro.pose[2])
        self.info['xp'].append(self.carro.pose[3])
        self.info['yp'].append(self.carro.pose[4])
        self.info['thetap'].append(self.carro.pose[5])

        return obs, recomp, terminou, self.info

    def reset(self, pose_especifica=None):

        self.info = {
            'estados': [],
            'acoes': [],
            'recompensas': [],
            'dts': [],
            'v': [],
            'phi': [],
            'x': [],
            'y': [],
            'theta': [],
            'xp': [],
            'yp': [],
            'thetap': [],
            'dif_ang': 1000
        }
        self.tempo = 0

        if pose_especifica is not None:
            return self.carro.set_pose_especifica(pose_especifica)

        return self.carro.set_pose_aleatoria(pose_inic=self.pose_inic, desvio_pose_inic=self.desvio_pose_inic)


    def render(self, mode='human', close=False):

        if self.renderizador is None:
            self.renderizador = Renderizador(tam_janela=self.tam_janela, lim_em_metros=self.lim_em_metros)
            self.renderizador.adc_carro(self.carro)
            self.renderizador.adc_vaga(self.vaga)

        return self.renderizador.desenha()

    def close(self):
        if self.renderizador.janela:
            self.renderizador.janela.close()
            self.renderizador.janela = None


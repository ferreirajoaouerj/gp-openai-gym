import random
import numpy as np
from scipy.stats import multivariate_normal
from numpy.linalg import norm
from collision import *
from gym.envs.classic_control import rendering


class Vaga:
    """
    Representa uma vaga entre dois carros de largura e comprimento definidos. Os carros apresentam uma forma de colisão.
    """

    def __init__(self, target_inic, desvio_target, dist_target_tol, compr, larg, fator_larg, fator_compr):
        """
        Inicialização da vaga. Vértices são nomeados a partir da parte mais baixa e à esquerda, seguindo o sentido AH.
        Args:
            target_inic (tuple): coordenadas e orientação do target inicial (vaga).
            desvio_target (tuple): desvio de cada coordenada do target inicial.
            compr (float): comprimento do carro.
            larg (float): largura do carro.
            fator_larg (float): a largura da vaga é dada por fator*larg.
            fator_compr (float):  distância do carro ao final é dado por
        Returns:
            Vaga: objeto composto de dois retângulos e um espaço entre eles.
        """

        self.target_inic = target_inic
        self.desvio_target = desvio_target
        self.compr = compr
        self.larg = larg
        self.fator_larg = fator_larg
        self.fato_compr = fator_compr
        self.dist_target_tol = dist_target_tol

        centro_x = target_inic[0] + random.uniform(- desvio_target[0], + desvio_target[0])
        centro_y = target_inic[1] + random.uniform(- desvio_target[1], + desvio_target[1])
        angle = target_inic[2] + random.uniform(- desvio_target[2], + desvio_target[2])

        self.rect_esq = Poly(pos=Vector(centro_x - (larg / 2) * (1 + fator_larg), centro_y),
                             points=[
                                 Vector(- larg/2,
                                        - compr/2),
                                 Vector(+ larg/2,
                                        - compr/2),
                                 Vector(+ larg/2,
                                        + compr/2),
                                 Vector(- larg/2,
                                        + compr/2)])

        self.rect_dir = Poly(pos=Vector(centro_x + (larg / 2) * (1 + fator_larg), centro_y),
                             points=[
                                 Vector(- larg / 2,
                                        - compr / 2),
                                 Vector(+ larg / 2,
                                        - compr / 2),
                                 Vector(+ larg / 2,
                                        + compr / 2),
                                 Vector(- larg / 2,
                                        + compr / 2)])

        c1 = list(self.rect_esq.points[2])
        a2 = list(self.rect_dir.points[0])
        d2 = list(self.rect_dir.points[3])

        x = [a2[0], d2[0]]
        y = [a2[1], d2[1]]

        coef = np.polyfit(x, y, 1)
        m = coef[0]
        k = coef[1]

        dist_desej = np.abs(c1[0] - d2[0])

        self.rect_esq.angle = angle
        self.rect_dir.angle = angle

        c1 = np.asarray(list(self.rect_esq.points[2]))
        a2 = np.asarray(list(self.rect_dir.points[0]))
        d2 = np.asarray(list(self.rect_dir.points[3]))

        dist_apos_rot = np.abs(np.cross(d2 - a2, c1 - a2)) / norm(d2 - a2)

        delta_dist_desej = np.abs(dist_desej - dist_apos_rot)

        delta_x_necess = (delta_dist_desej
                          * np.sqrt(1 + m)
                          * np.abs(k + m * c1[0] - c1[1])) / np.abs(k + m * c1[0] - c1[1])

        self.rect_esq.pos = Vector(self.rect_esq.pos[0] - delta_x_necess/2, self.rect_esq.pos[1])
        self.rect_dir.pos = Vector(self.rect_dir.pos[0] + delta_x_necess/2, self.rect_esq.pos[1])

        max_y = 0
        for idx, pt in enumerate(self.rect_esq.points):
            if pt[1] > max_y:
                max_y = pt[1]
                idx_max_pt = idx

        pt_a = self.rect_esq.points[idx_max_pt]
        pt_b = self.rect_dir.points[idx_max_pt]

        dist_pt_a_b = np.abs(pt_a[0] - pt_b[0])

        centro_calcada = Vector(pt_a[0] + dist_pt_a_b / 2,
                                pt_a[1] + 1.5*compr*fator_compr)

        self.calcada = Poly.from_box(center=centro_calcada, width=dist_pt_a_b, height=compr*fator_compr)

        self.centro_x = centro_x
        self.centro_y = centro_y
        self.angl = angle


    def pegar_lista_tuplas_pontos(self, rect):
        """
        Pega a lista de todos os pontos que compõe o polígono. Inclui o primeiro ponto duas vezes, pois parece ser
        necessário para o 'draw_polyline'.
        Returns:
            list of tuple: retorna lista de tuplas contendo as coordenadas dos pontos.
        """

        lst = [tuple(coord) for coord in rect.points]

        lst.append(lst[0])

        return lst


class JuizColisao:
    """
    O juíz irá verificar a colisão de um objeto móvel com uma lista de objetos estáticos.
    """

    def __init__(self):
        """
        O único atributo do juíz é uma lista que contém todos os objetos estáticos do ambiente.
        """

        self.list_objs = []

    def adic_obj(self, obj):
        """
        Adiciona um objeto estático que deve-se ser inspecionado para verificação de colisões.
        Args:
            obj (Shape): objeto com formato definido a partir da biblioteca collisions.
        """

        self.list_objs.append(obj)

    def checar_colisao(self, obj_movel):
        """
        Checa se houve colisão entre os objetos da lista e um objeto móvel.
        Args:
            obj_movel (Shape): forma do objeto móvel.
        Returns:
            bool: True se houve colisão e False caso contrário.
        """

        res_colisoes = []
        for shape in self.list_objs:
            res_colisao_atual = collide(obj_movel, shape)
            res_colisoes.append(res_colisao_atual)
        if True in res_colisoes:
            return True
        else:
            return False


class Carro:
    """
    Classe que implementa o carrinho, sua cinemática e forma de colisão.
    """

    def __init__(self,
                 max_v, max_phi, max_v_p, max_phi_p,
                 compr, larg,
                 pose_inic, tipo_pose, desvio_pose_inic,
                 vaga=None):
        """
        Inicializa um carro.
        Args:
            max_v (float): velocidade máxima.
            max_phi (float): ângulo de 'phiing' máximo
            compr (float): comprimento do carrinho.
            larg (float): largura do carrinho
            pose_inic (tuple): pose inicial
        Returns:
            Carro: instância do carro.
        """

        self.dif_angs = []

        self.max_v = max_v
        self.max_phi = max_phi
        self.max_v_p = max_v_p
        self.max_phi_p = max_phi_p
        self.compr = compr
        self.larg = larg

        self.tipo_pose = tipo_pose
        self.vaga = vaga

        if tipo_pose == 'dist':
            assert self.vaga is not None

        centro = Vector(pose_inic[0], pose_inic[1])

        vert_a = Vector(- compr / 2, - larg / 2)
        vert_b = Vector(+ compr / 2, - larg / 2)
        vert_c = Vector(+ compr / 2, + larg / 2)
        vert_d = Vector(- compr / 2, + larg / 2)

        vertices = [vert_a, vert_b, vert_c, vert_d]

        self.shape = Poly(centro, vertices, pose_inic[2])

        self.v = 0
        self.phi = 0
        self.theta = pose_inic[2]

        self.pose = np.array([
            pose_inic[0] + random.uniform(- desvio_pose_inic[0], + desvio_pose_inic[0]),
            pose_inic[1] + random.uniform(- desvio_pose_inic[1], + desvio_pose_inic[1]),
            pose_inic[2] + random.uniform(- desvio_pose_inic[2], + desvio_pose_inic[2]),
            0,
            0,
            0
        ])

        if tipo_pose == 'dist':
            self.pose_dist = np.array([
                self.pose[0] - self.vaga.centro_x,
                self.pose[1] - self.vaga.centro_y,
                self.pose[2],
                0,
                0,
                0
            ])

    def update(self, v, phi, dt, modo='acel'):
        """
        Avança um passo de tempo 'dt' com nova velocidade 'v' e ângulo de 'steering' 'phi', calculando a nova pose.
        Args:
            v (float): nova velocidade do carrinho, usada para calcular a pose.
            phi (float): novo ângulo de 'steering', usado para calcular a pose.
            dt (float): intervalo de tempo de integração.
            modo (str): se o controle é feito através das derivadas ou não.
        Returns:
            pose (list): retorna a lista contendo [x, y, theta]
        """

        if modo == 'acel':

            v_p, phi_p = v, phi

            v_p_clip, phi_p_clip = np.clip([v_p, phi_p],
                                           [- self.max_v_p, - self.max_phi_p],
                                           [+ self.max_v_p, + self.max_phi_p])

            v_clip, phi_clip = np.clip([self.v + dt * v_p_clip, self.phi + dt * phi_p_clip],
                                       [- self.max_v, - self.max_phi],
                                       [+ self.max_v, + self.max_phi])

        else:

            v_clip = np.clip(v, -self.max_v, +self.max_v)
            phi_clip = np.clip(phi, -self.max_phi, +self.max_phi)

        x_p = v_clip * np.cos(self.pose[2])
        y_p = v_clip * np.sin(self.pose[2])
        theta_p = v_clip * np.tan(phi_clip) / self.compr

        angl = np.mod(self.pose[2] + theta_p * dt, 2 * np.pi)

        self.pose = np.array([
            self.pose[0] + x_p * dt,
            self.pose[1] + y_p * dt,
            np.mod(self.pose[2] + theta_p * dt, 2 * np.pi),
            x_p,
            y_p,
            theta_p
        ])

        self.v = v_clip
        self.phi = phi_clip

        self.shape.pos = Vector(self.pose[0], self.pose[1])
        self.shape.angle = self.pose[2]

        if self.tipo_pose == 'dist':
            self.pose_dist = np.array([
                self.pose[0] - self.vaga.centro_x,
                self.pose[1] - self.vaga.centro_y,
                self.pose[2],
                self.pose[3],
                self.pose[4],
                self.pose[5]
            ])

            return self.pose_dist

        self.pose = np.array(self.pose)

        return self.pose

    def set_pose_aleatoria(self, pose_inic, desvio_pose_inic):
        """
        Coloca o carro em uma pose pré-definida ou aleatória.
        Args:
            pose_inic (tuple): pose inicial de referência.
            desvio_pose_inic (tuple): tupla com os desvios de cada elemento da pose atual.
        Returns:
            pose (list): nova pose do carro.
        """

        angl = pose_inic[2] + random.uniform(- desvio_pose_inic[2], + desvio_pose_inic[2])

        self.pose = np.array([
            pose_inic[0] + random.uniform(- desvio_pose_inic[0], + desvio_pose_inic[0]),
            pose_inic[1] + random.uniform(- desvio_pose_inic[1], + desvio_pose_inic[1]),
            angl,
            0,
            0,
            0
        ])

        self.shape.pos = Vector(self.pose[0], self.pose[1])
        self.shape.angle = self.pose[2]

        if self.tipo_pose == 'dist':
            self.pose_dist = np.array([
                self.pose[0] - self.vaga.centro_x,
                self.pose[1] - self.vaga.centro_y,
                angl,
                0,
                0,
                0
            ])

            return self.pose_dist

        self.pose = np.array(self.pose)

        return self.pose

    def set_pose_especifica(self, pose):
        """
        Coloca o carro em uma pose específica.
        Args:
            pose (tuple): pose desejada.
        Returns:
            pose (tuple): nova pose.
        """

        self.pose = np.array([pose[0],
                              pose[1],
                              pose[2],
                             0,
                             0,
                             0])

        self.shape.pos = Vector(self.pose[0], self.pose[1])
        self.shape.angle = self.pose[2]

        if self.tipo_pose == 'dist':
            self.pose_dist = np.array([
                self.pose[0] - self.vaga.centro_x,
                self.pose[1] - self.vaga.centro_y,
                self.pose[2],
                0,
                0,
                0])
            return self.pose_dist

        self.pose = np.array(self.pose)

        return self.pose


    def pegar_recompensa(self, vaga, pesos=(1, 1, 1), angulo=False):
        """
        Obtém a recompensa a partir de uma determinada vaga.
        Args:
            vaga (Vaga): objeto que representa uma vaga.
            pesos (tuple): pesos para a função custo.
            angulo (bool): utilizar ou não o ângulo.
        Returns:
            recomp (float), done (bool): recompensa obtida na situação atual e se estado é terminal.
        """

        dif_pos = np.sqrt(self.pose_dist[0]**2 + self.pose_dist[1]**2)
        # dif_ang = np.abs(np.arctan2(np.sin(vaga.angl-self.pose[2]), np.cos(vaga.angl-self.pose[2])))
        dif_v = self.v

        ## Reward shaping
        # x = np.dot(pesos, [dif_pos, dif_ang, dif_v])
        # recomp = multivariate_normal.pdf(x, mean=0, cov=0.15)

        recomp = -dif_pos

        if dif_pos <= vaga.dist_target_tol:
            if angulo:
                dif_ang = np.abs(np.arctan2(np.sin(vaga.angl - self.pose[2]), np.cos(vaga.angl - self.pose[2])))
                return 2000 * np.abs(1 - dif_ang/(2*np.pi)), True, dif_ang
            else:
                return 2000, True, None
        else:
            return recomp, False, None


class Renderizador:
    """
    Responsável pela renderização de todos os objetos estáticos e dinâmicos.
    """

    def __init__(self, tam_janela=800, lim_em_metros=2):
        """
        Define resolução quadrada e quantos metros isso corresponderá.
        Args:
            tam_janela (int): quantidade de pixels na janela (altura e largura).
            lim_em_metros (int): qual será a altura (e largura), em metros, correspondente.
        Returns:
            Renderizador: instância de um objeto para renderizar os gráficos.
        """

        self.janela = rendering.Viewer(tam_janela, tam_janela)
        self.janela.set_bounds(0, lim_em_metros, 0, lim_em_metros)

        self.carro = None
        self.des_carro = None
        self.des_carro_transf = None
        self.des_capo_carro = None
        self.des_capo_carro_transf = None

        self.vaga = None

    def adc_carro(self, carro, cor=(0, 0.6, 0)):

        self.carro = carro
        self.des_carro = rendering.make_polygon([(- self.carro.compr / 2, + self.carro.larg / 2),
                                                 (- self.carro.compr / 2, - self.carro.larg / 2),
                                                 (+ self.carro.compr / 2, - self.carro.larg / 2),
                                                 (+ self.carro.compr / 2, + self.carro.larg / 2)],
                                                True)

        self.des_capo_carro = rendering.make_polygon([(- self.carro.compr / 16, + self.carro.compr / 16),
                                                      (- self.carro.compr / 16, - self.carro.compr / 16),
                                                      (+ self.carro.compr / 16, - self.carro.compr / 16),
                                                      (+ self.carro.compr / 16, + self.carro.compr / 16)],
                                                     True)

        self.des_carro_transf = rendering.Transform()
        self.des_capo_carro_transf = rendering.Transform()

        self.des_carro.set_color(*cor)
        self.des_capo_carro.set_color(0, 0, 255)

        self.des_carro.add_attr(self.des_carro_transf)
        self.des_capo_carro.add_attr(self.des_capo_carro_transf)

        self.janela.add_geom(self.des_carro)
        self.janela.add_geom(self.des_capo_carro)

    def adc_vaga(self, vaga):

        if self.vaga is None:
            self.vaga = vaga

    def desenha(self, mode='human'):

        if self.carro is not None:
            self.des_carro_transf.set_rotation(self.carro.pose[2])
            self.des_carro_transf.set_translation(self.carro.pose[0],
                                                  self.carro.pose[1])
            self.des_capo_carro_transf.set_rotation(self.carro.pose[2])
            self.des_capo_carro_transf.set_translation(self.carro.pose[0] +
                                                       0.88*(self.carro.compr/2)*np.cos(self.carro.pose[2]),
                                                       self.carro.pose[1] +
                                                       0.88*(self.carro.compr/2)*np.sin(self.carro.pose[2]))

        if self.vaga is not None:
            target_transform = rendering.Transform(translation=(self.vaga.centro_x, self.vaga.centro_y))
            target = self.janela.draw_circle(radius=self.vaga.dist_target_tol)
            target.set_color(1, 0, 0)
            target.add_attr(target_transform)

            self.janela.draw_polyline(self.vaga.pegar_lista_tuplas_pontos(self.vaga.rect_esq),
                                      linewidth=1)
            self.janela.draw_polyline(self.vaga.pegar_lista_tuplas_pontos(self.vaga.rect_dir),
                                      linewidth=1)

            self.janela.draw_polyline(self.vaga.pegar_lista_tuplas_pontos(self.vaga.calcada),
                                      linewidth=2)

        return self.janela.render(return_rgb_array=mode == 'rgb_array')

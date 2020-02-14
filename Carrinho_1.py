#-*- coding:utf-8 -*-
"""
Documentation
"""

######################################################################################################

import os
import pickle
import time
import copy
from collections import defaultdict

import random
import operator
import numpy as np

from deap import base
from deap import creator
from deap import gp
from deap import tools
from deap import algorithms

import matplotlib.pyplot as plt
from matplotlib import colors
import networkx as nx

import gym
import gym_carrinho

import warnings
warnings.simplefilter('ignore', np.RankWarning)


######################################################################################################
#################################### PARÂMETROS ######################################################
######################################################################################################


# random.seed(1)

param_pg = {
    'tam_pop': 500,
    'pb_cx': 0.75,
    'pb_mut': 0.05,
    'n_geracoes': 15,
    'tipo_apt': 1.0,
    'n_entradas': 6,
    'faixa_cst': [-1, 1],
    'n_episodios': 1,
    'camp_apt': 6,
    'camp_d': 1.2,
    'd_min': 2,
    'd_max': 5,
    'max_d_mut': 7,
    'limite_d': 17,
}

param_aux = {
    'n_exec': 1,
    'amb': 'Carrinho-v0',
    'mujoco': False
}

env_param = {
    'modo': 'const',
    'tipo_pose': 'dist',
    'obstaculo': False,
    'tempo_max': 1500,
    'lim_em_metros': 1.2,
    'pose_inic': (0.2, 0.2, 0),
    'desvio_pose_inic': (0, 0, 0),
    'target_inic': (1.0, 1.0, 0),
    'desvio_target': (0, 0, 0),
    'pesos': (1, 0, 0),
    'dist_target_tol': 0.03,
    'max_v': 0.1,
    'max_phi': 0.5,
    'dt': 1/30
}


param_graf = {
    'nomes_var_tex': (r'$x$', r'$y$', r'$\theta$', r'$\dot{x}$', r'$\dot{y}$', r'$\dot{\theta}$',),
    'plot_var': ('Ação', 'Resultado', r"$x$", r"$y$", r'$\theta$'),
    'var_adic': ('v', 'phi'),
    'nomes_var_adic_tex': (r'$v$', r'$\phi$'),
    'operadores': {'add': r'$+$', 'sub': r'$-$', 'mul': r'$\times$', 'div': r'$\div$',
                   'gt': r'$>$', 'sr': r'$\sqrt{\,}$', 'sen': r'$\sin$', 'sgn': 'sgn',
                   'constante': r'$R$'},
    'hist_n_linhas': 3,
    'hist_n_cols': 2,
    'bins_min': -2000,
    'bins_max': 2250,
    'bins_passo': 250
}

var_aux = {
    'n_dt': 0,
    'n_simul': 0,
    't_exec': 0
}

param_store = {
    'pg': param_pg,
    'aux': param_aux,
    'graf': param_graf,
    'var_aux': var_aux
}

script_filename = os.path.basename(__file__)
store_filename = script_filename.replace(".py", "_Stats.pkl")


######################################################################################################
################################ FUNÇÕES AUXLIARES - AVALIAÇÃO DE INDIVÍDUO ##########################
######################################################################################################


def wrap(velphi_action):
    return np.array(np.clip(velphi_action, [-0.1, -0.5], [0.1, 0.5]), dtype=np.float32)


def calc_custo_inst(obs=None, recomp=None):
    return 0


def calc_custo_acum(obs=None, recomp=None):
    return recomp


def calc_custo(custo_inst=None, custo_acum=None, obs=None, recomp=None):
    return custo_acum


def avaliar_individuo_multi(ind, num_episodios=param_pg['n_episodios'], num_entradas=param_pg['n_entradas'],
                            ambiente=param_aux['amb'], nomes_var=param_graf['nomes_var_tex'],
                            plotar_var=param_graf['plot_var'], graficos=False, video=False, mujoco=param_aux['mujoco'],
                            var_adic=param_graf['var_adic'], nomes_var_adic_tex=param_graf['nomes_var_adic_tex']):
    aptidoes = []
    tempo = 0
    dic_stats = {'Tempo': [], 'Resultado': [], 'Acao': [], 'Custo Acumulado': []}
    funcao_de_controle_vel = toolbox.compilar_individuo(ind[0])
    funcao_de_controle_phi = toolbox.compilar_individuo(ind[1])
    ambiente = gym.make(ambiente, **env_param)
    for obs in range(num_entradas):
        dic_stats['ARG' + str(obs)] = []
    if graficos:
        num_episodios = 1
    if mujoco and video:
        ambiente.render()
    for episodio in range(num_episodios):
        var_aux['n_simul'] += 1
        tempo_ep = 0
        custo_acumulado = custo_instantaneo = 0
        termino = False
        observacao = ambiente.reset()
        while not termino:
            var_aux['n_dt'] += 1
            tempo_ep += 1
            resultado_vel = funcao_de_controle_vel(*tuple(observacao))
            resultado_phi = funcao_de_controle_phi(*tuple(observacao))
            resultado = [resultado_vel, resultado_phi]
            acao = wrap(resultado)
            observacao, recompensa, termino, info = ambiente.step(acao)
            custo_instantaneo = calc_custo_inst()
            custo_acumulado += calc_custo_acum(recomp=recompensa)
            custo = calc_custo(custo_acum=custo_acumulado)
            if video and not mujoco:
                ambiente.render()
            if graficos:
                tempo += 1
                dic_stats['Tempo'].append(tempo), dic_stats['Resultado'].append(resultado),
                dic_stats['Acao'].append(acao), dic_stats['Custo Acumulado'].append(custo_acumulado)
                for obs in range(num_entradas):
                    dic_stats['ARG' + str(obs)].append(observacao[obs])
        aptidoes.append(custo)
    aptidao_media = np.average(aptidoes)
    if graficos:
        plotar_vars_avaliacao(data=dic_stats, nomes_var=nomes_var,
                              plotar_var=plotar_var, tempo_final=tempo_ep/5)
        plotar_info_adicional(info, variaveis=var_adic, nomes_var=nomes_var_adic_tex)
        plotar_trajetoria(info, 20)
    if not mujoco and video:
        ambiente.close()
    return aptidao_media,


######################################################################################################
############################# FUNÇÕES AUXLIARES - COMPILAÇÃO DE ESTATÍSTICAS #########################
######################################################################################################


def get_data(l):
    new = []
    for i in l:
        if not np.isnan(i):
            new.append(i)
    return new


def minimo(lista):
    array = np.array(lista)
    array = array[~np.isnan(array)]
    return np.min(array)


def maximo(lista):
    array = np.array(lista)
    array = array[~np.isnan(array)]
    return np.max(array)


def media(lista):
    array = np.array(lista)
    array = array[~np.isnan(array)]
    return np.average(array)


def desvio(lista):
    array = np.array(lista)
    array = array[~np.isnan(array)]
    return np.std(array)


def copiar_estatisticas(logbk=None):
    newdic = {}
    for key in logbk.chapters.keys():
        newdic[key] = {}
    for key in logbk.chapters.keys():
        newdic[key].update(max=logbk.chapters[key].select('max'), min=logbk.chapters[key].select('min'),
                           media=logbk.chapters[key].select('media'), desvio=logbk.chapters[key].select('desvio'),
                           ocorrencias=logbk.chapters[key].select('ocorrencias'), gen=logbk.select('gen'))
    return newdic


def contar_operadores(conjprim, pop):
    op_count_dict = {'constante': 0}
    valores_constantes = []
    for op in conjprim.primitives[object]:
        if not op_count_dict.keys().__contains__(op.name):
            op_count_dict[op.name] = 0
    for arg in conjprim.arguments:
        if not op_count_dict.keys().__contains__(arg):
            op_count_dict[arg] = 0
    for individuo in pop:
        for no in individuo:
            if 'ARG' not in no.name:
                if not op_count_dict.keys().__contains__(no.name):
                    op_count_dict['constante'] += 1
                    valores_constantes.append(no.value)
                else:
                    op_count_dict[no.name] += 1
            else:
                op_count_dict[no.value] += 1
    return op_count_dict, valores_constantes


def media_estatisticas(stat_dic_list: list, occ_dic_list: list, const_list: list):
    for stat_dict in stat_dic_list:
        for key in stat_dict.keys():
            for subkey in estatisticas[key]:
                stat_dict[key][subkey] = np.array(stat_dict[key][subkey])
    for occ_dic in occ_dic_list:
        for key in occ_dic.keys():
            occ_dic[key] = np.array(occ_dic[key])
    sum_dict = copy.deepcopy(stat_dic_list[0])
    occ_sum = copy.deepcopy(occ_dic_list[0])
    for stat_dict in stat_dic_list[1:]:
        for key in stat_dict.keys():
            for subkey in estatisticas[key]:
                sum_dict[key][subkey] += stat_dict[key][subkey]
    for key in sum_dict.keys():
        for subkey in sum_dict[key].keys():
            sum_dict[key][subkey] = sum_dict[key][subkey] / len(stat_dic_list)
    for occ_dic in occ_dic_list[1:]:
        for key in occ_dic.keys():
            occ_sum[key] += occ_dic[key]
    for key in occ_sum.keys():
        occ_sum[key] = occ_sum[key] / len(occ_dic_list)
    const_avg_list = []
    for cst_lst in const_list:
        if len(cst_lst) < 1:
            const_avg_list.append(0)
        else:
            const_avg_list.append(np.average(cst_lst))
    const_avg = np.average(const_avg_list)
    return sum_dict, occ_sum, const_avg


def calcular_media_exec(lista_dic_stats, lista_dic_occ, lista_lista_const):
    bins = np.arange(param_graf['bins_min'], param_graf['bins_max'], param_graf['bins_passo'])
    n_mat_list = []
    # occ_mat = np.ndarray(shape=(param_pg['n_geracoes'] + 1, param_pg['tam_pop']))
    occ_mat = []
    for dic_stats in lista_dic_stats:
        for key in dic_stats.keys():
            for subkey in dic_stats[key].keys():
                if subkey != 'ocorrencias':
                    dic_stats[key][subkey] = np.array(dic_stats[key][subkey], dtype='float64')
                elif key == 'aptidao' and subkey == 'ocorrencias':
                    occ_mat = []
                    for ger, lista_occ in enumerate(dic_stats[key][subkey]):
                        n, bins = np.histogram(lista_occ, bins=bins)
                        occ_mat.append(n)
                    n_mat_list.append(occ_mat)
    mat_acum = np.asarray(n_mat_list[0])
    for mat in n_mat_list[1:]:
        mat_acum += np.asarray(mat)
    n_avg = []
    for i in range(len(mat_acum[0])):
        n_avg.append(mat_acum[i] / len(n_mat_list))
    new_apt_ocorrencias = []
    for i in range(len(n_avg)):
        new_apt_ocorrencias.append([])
        for j in range(len(n_avg[i])):
            new_apt_ocorrencias[i].append(random.randint(bins[i], bins[i + 1]))
    lista_lista_const = np.array([np.array(x) for x in lista_lista_const])
    dic_stats_acum = copy.deepcopy(lista_dic_stats[0])
    dic_occ_acum = copy.deepcopy(lista_dic_occ[0])
    const_acum = 0
    n = len(lista_lista_const)
    for dic_stats in lista_dic_stats[1:]:
        for key in dic_stats.keys():
            for subkey in dic_stats[key]:
                if subkey != 'ocorrencias':
                    dic_stats_acum[key][subkey] = np.add(dic_stats_acum[key][subkey],
                                                         dic_stats[key][subkey])
    for key in dic_stats_acum.keys():
        for subkey in dic_stats_acum[key].keys():
            if subkey != 'ocorrencias':
                dic_stats_acum[key][subkey] = dic_stats_acum[key][subkey] / n
    dic_stats_media = dic_stats_acum
    for dic_occ in lista_dic_occ[1:]:
        for key in dic_occ.keys():
            dic_occ_acum[key] += dic_occ[key]
    for label in dic_occ_acum.keys():
        dic_occ_acum[label] = dic_occ_acum[label] / n
    dic_occ_media = dic_occ_acum
    for lista_const in lista_lista_const:
        if len(lista_const) > 0:
            const_acum += np.average(np.nan_to_num(lista_const))
    const_med = const_acum / n
    return dic_stats_media, dic_occ_media, const_med


######################################################################################################
############################# FUNÇÕES AUXLIARES - GERAÇÃO DE GRÁFICOS ################################
######################################################################################################


def plotar_arvore(ind):
    plt.figure()
    nodes, edges, labels = gp.graph(ind)
    for label_key, label_value in labels.items():
        for op_key, op_value in param_graf['operadores'].items():
            if str(label_value) == op_key:
                labels[label_key] = op_value
    for i in range(param_pg['n_entradas']):
        for key, value in labels.items():
            if value == ('ARG' + str(i)):
                labels[key] = param_graf['nomes_var_tex'][i]
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    pos = nx.drawing.nx_agraph.graphviz_layout(g, prog="dot")
    nx.draw_networkx_nodes(g, pos, node_size=1100, node_color='black')
    nx.draw_networkx_edges(g, pos)
    nx.draw_networkx_labels(g, pos, labels, font_color='white', font_size=12)
    plt.show()


def plotar_vars_avaliacao(data=None, nomes_var=None, plotar_var=None, tempo_final=None):
    figura, eixo = plt.subplots(nrows=len(plotar_var), ncols=1, sharex='all')
    data['Ação'] = data.pop('Acao')
    for n in range(len(nomes_var)):
        data[nomes_var[n]] = data.pop('ARG' + str(n))
    if len(plotar_var) < 2:
        eixo.plot('Tempo', plotar_var[0], data=data, ls='-.', marker='o', ms=4, color='blue', alpha=0.3)
        eixo.set_xlabel('Tempo')
        eixo.set_ylabel(plotar_var[0])
        eixo.grid()
    else:
        cores = ['b', 'g', 'r', 'm', 'c']
        for var in range(len(plotar_var)):
            eixo[var].plot('Tempo', plotar_var[var], data=data, ls='-.', marker='o', ms=4, color=cores[var], alpha=0.3)
            eixo[var].set_ylabel(plotar_var[var])
            eixo[var].grid(True)
    eixo[-1].set_xlabel('Tempo')  # nome apenas no último gráfico
    eixo[-1].set_xlim(0, tempo_final)


def plotar_info_adicional(data=None, variaveis=None, nomes_var=None):
    figura, eixo = plt.subplots(nrows=len(variaveis), ncols=1, sharex='all')
    cores = ['b', 'g', 'r', 'm', 'c']
    for i, var in enumerate(variaveis):
        eixo[i].plot('dts', var, data=data, ls='-.', marker='o', ms=4, color=cores[i], alpha=0.3)
        eixo[i].set_ylabel(nomes_var[i])
        eixo[i].grid(True)
    eixo[-1].set_xlabel('Tempo')


def plotar_trajetoria(dic=None, pontos=10):
    fig, ax = plt.subplots()
    ax.plot('x', 'y', data=dic, lw=2)
    for i in np.arange(dic['dts'][0], dic['dts'][-1], int(dic['dts'][-1]/pontos)):
        x, y, theta = dic['x'][i], dic['y'][i], dic['theta'][i]
        ax.annotate("",
                    xy=(x + 0.02*np.cos(theta), y + 0.02*np.sin(theta)), xycoords='data',
                    xytext=(x, y), textcoords='data',
                    arrowprops=dict(arrowstyle="-|>",
                                    connectionstyle="arc3",
                                    mutation_scale=20))
    plt.grid(True)
    plt.xlim([0, env_param['lim_em_metros']])
    plt.ylim([0, env_param['lim_em_metros']])
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')


def plotar_hits(data=None, nrows=param_graf['hist_n_linhas'], ncols=param_graf['hist_n_cols'],
                bins_min=param_graf['bins_min'], bins_max=param_graf['bins_max'], bins_step=param_graf['bins_passo']):
    fig, eixo = plt.subplots(nrows=nrows, ncols=ncols, sharex='all', sharey='all', constrained_layout=True)
    norm = colors.Normalize(bins_min, bins_max - bins_step)
    dados = data['aptidao']['ocorrencias']
    graf = (np.round(np.linspace(0, len(dados)-1, nrows*ncols))).astype('int')
    n = 0
    for j in range(ncols):
        for i in range(nrows):
            N, bins, patches = eixo[i][j].hist(dados[graf[n]], bins=np.arange(bins_min, bins_max, bins_step))
            for thisbin, thispatch in zip(bins, patches):
                color = plt.cm.viridis(norm(thisbin))
                thispatch.set_facecolor(color)
            eixo[i][j].set_title('Geração ' + str(graf[n]))
            eixo[i][j].set_ylabel('Ocorrencias') if j is 0 else None
            eixo[i][j].set_xlabel('Aptidão') if i is (nrows-1) else None
            n += 1


def plotar_estatisticas_evolucao(data=None):
    stats = ['aptidao']
    labels = ['Aptidão']
    ls = '-.'
    lw = 1
    ms = 7
    for stats, labels in zip(stats, labels):
        fig, eixo = plt.subplots()
        fig.suptitle(stats.capitalize() + ' por Geração', fontsize=14)
        eixo.plot('gen', 'min', 'bv:', data=data[stats], label=labels + ' Min', ls=ls, markersize=ms, lw=lw)
        eixo.plot('gen', 'max', 'r^:', data=data[stats], label=labels + ' Max', ls=ls, markersize=ms, lw=lw)
        eixo.plot('gen', 'media', 'g>:', data=data[stats], label=labels + ' Med', ls=ls, markersize=ms, lw=lw)
        eixo.set_xlabel('Geração')
        eixo.set_ylabel(stats.capitalize())
        if stats is 'aptidao':
            fig.suptitle('Aptidão por Geração', fontsize=14)
            apt = data[stats]
            eixo.fill_between('gen', [x + y for x, y in zip(apt['media'], apt['desvio'])],
                              [x - y for x, y in zip(apt['media'], apt['desvio'])], data=apt, label='Desvio Padrão',
                              alpha=0.2, edgecolor='#1B2ACC', facecolor='#089FFF', linewidth=0.1,
                              linestyle='dashed', antialiased=True)
            eixo.set_ylabel('Aptidão')
        eixo.legend(loc='best')
        plt.xlim(-0.1, len(data[stats]['gen']) - 0.9)
        eixo.grid()


def plotar_ocorrencias(data=None):
    newdic = copy.deepcopy(data)
    for i in range(param_pg['n_entradas']):
        newdic[param_graf['nomes_var_tex'][i]] = newdic.pop('ARG' + str(i))
    for key in data.keys():
        if param_graf['operadores'].__contains__(key):
            newdic[param_graf['operadores'][key]] = newdic.pop(key)
    fig, eixo = plt.subplots()
    bars = plt.bar(*zip(*newdic.items()))
    ax = bars[0].axes
    lim = ax.get_xlim() + ax.get_ylim()
    for bar in bars:
        bar.set_zorder(1)
        bar.set_facecolor("none")
        x, y = bar.get_xy()
        w, h = bar.get_width(), bar.get_height()
        grad = np.atleast_2d(np.linspace(0, h / max(data.values()), 256))
        eixo.imshow(grad.transpose(), extent=[x, x + w, y + h, y], aspect="auto", zorder=0,
                    norm=colors.NoNorm(vmin=0, vmax=1))
    ax.axis(lim)


def avaliar_hdf(hdf):
    aptidoes = []
    for ind in hdf[:10]:
        aptidao = avaliar_individuo_multi(ind, num_episodios=100)
        aptidoes.append(aptidao)
    aptidao_media_melhores = np.sum(aptidoes) / len(aptidoes)
    print("Aptidão média (10 primeiros do hall da fama): ", aptidao_media_melhores)
    return aptidao_media_melhores


def avaliar_melhor_hdf(hdf):
    best = -100000
    for ind in hdf:
        aptidao, = avaliar_individuo_multi(ind, num_episodios=100)
        best = aptidao if aptidao > best else best
    print("Melhor aptidão do hall da fama: ", best)
    return best

######################################################################################################
################################ FUNÇÕES AUXLIARES - MULTIGEN ########################################
######################################################################################################


def cxonepoint_multi(ind1, ind2):

    ind1[0], ind2[0] = gp.cxOnePoint(ind1[0], ind2[0])
    ind1[1], ind2[1] = gp.cxOnePoint(ind1[1], ind2[1])

    return ind1, ind2


def mutuniform_multi(individual, expr, pset):

    individual[0], = gp.mutUniform(individual[0], expr, pset)
    individual[1], = gp.mutUniform(individual[1], expr, pset)

    return individual,


def gerarexpr_multi(gen_phi, gen_vel):
    ind_phi = gen_phi()
    ind_vel = gen_vel()
    ind = [ind_phi, ind_vel]
    return ind


######################################################################################################
################################ FUNÇÕES AUXLIARES - FUNÇÕES PRIMITIVAS ##############################
######################################################################################################


def divisao_p(left, right):
    try:
        res = left / right
        if np.isnan(res) or np.isneginf(res) or np.isinf(res):
            return 1
        else:
            return res
    except ArithmeticError:
        return 1


def raizquadrada_p(num):
    if num > 0:
        return np.sqrt(num)
    else:
        return -np.sqrt(abs(num))


def maior(a, b):
    if a > b:
        return 1.0
    else:
        return -1.0


######################################################################################################
################################ DEFINIÇÃO DO CONJUNTO PRIMITIVO #####################################
######################################################################################################


ConjPrim = gp.PrimitiveSet("MAIN", arity=param_pg['n_entradas'])
ConjPrim.addPrimitive(np.add, 2, name="add")
ConjPrim.addPrimitive(np.subtract, 2, "sub")
ConjPrim.addPrimitive(np.multiply, 2, name="mul")
ConjPrim.addPrimitive(divisao_p, 2, name="div")
ConjPrim.addPrimitive(maior, 2, name="gt")
ConjPrim.addPrimitive(raizquadrada_p, 1, name="sr")
ConjPrim.addPrimitive(np.sign, 1, name="sgn")
ConjPrim.addPrimitive(np.sin, 1, name="sen")
ConjPrim.addEphemeralConstant("R", lambda: round(random.uniform(*param_pg['faixa_cst']), 2))


######################################################################################################
################################ FUNÇÕES INICIALIZAÇÃO, OPERAÇÃO GENÉTICA E AVALIAÇÃO ################
################n######################################################################################


creator.create("AptidaoMax", base.Fitness, weights=(param_pg['tipo_apt'],))
creator.create("IndivVel", gp.PrimitiveTree, pset=ConjPrim)
creator.create("IndivPhi", gp.PrimitiveTree, pset=ConjPrim)
creator.create("Individuo", list, fitness=creator.AptidaoMax, pset=ConjPrim)

toolbox = base.Toolbox()

toolbox.register("gerarExpr", gp.genHalfAndHalf, pset=ConjPrim, min_=1, max_=3)
toolbox.register("gerarExprPhi", tools.initIterate, creator.IndivPhi, toolbox.gerarExpr)
toolbox.register("gerarExprVel", tools.initIterate, creator.IndivVel, toolbox.gerarExpr)
toolbox.register("gerarExprMultiGen", gerarexpr_multi, gen_phi=toolbox.gerarExprPhi, gen_vel=toolbox.gerarExprVel)
toolbox.register("gerarExprMut", gp.genFull, pset=ConjPrim, min_=param_pg['d_min'], max_=param_pg['d_max'])

toolbox.register("gerarIndivMultiGen", tools.initIterate, creator.Individuo, toolbox.gerarExprMultiGen)
toolbox.register("gerar_populacao", tools.initRepeat, list, toolbox.gerarIndivMultiGen, param_pg['tam_pop'])

toolbox.register("compilar_individuo", gp.compile, pset=ConjPrim)

toolbox.register("evaluate", avaliar_individuo_multi)
toolbox.register("select", tools.selTournament, tournsize=param_pg['camp_apt'])
toolbox.register("mate", cxonepoint_multi)
toolbox.register("mutate", mutuniform_multi, expr=toolbox.gerarExpr, pset=ConjPrim)

toolbox.decorate("mate", gp.staticLimitMulti(key=operator.attrgetter("height"), max_value=param_pg['limite_d']))
toolbox.decorate("mutate", gp.staticLimitMulti(key=operator.attrgetter("height"), max_value=param_pg['limite_d']))


######################################################################################################
################################ LOG #################################################################
######################################################################################################


aptidao_lista = tools.Statistics(lambda ind: ind.fitness.values[0])

mstats = tools.MultiStatistics(aptidao=aptidao_lista)

mstats.register("min", minimo)
mstats.register("media", media)
mstats.register("max", maximo)
mstats.register("desvio", desvio)
mstats.register("ocorrencias", get_data)

logbook = tools.Logbook()

logbook.header = "min", "media", "max", "desvio"


######################################################################################################
################################ CICLO EVOLUCIONARIO #################################################
######################################################################################################

modo = input('New run? (y) or (n)\n')
if modo is 'y':

    estatisticas_lista = []
    occ_lista = []
    const_lista = []
    hdf_lista = []

    tinit = time.time()

    for run in range(param_aux['n_exec']):

        populacao = toolbox.gerar_populacao()
        halldafama = tools.HallOfFame(30)

        populacao, log = algorithms.eaMuCommaLambda(population=populacao, toolbox=toolbox,
                                                    mu=param_pg['tam_pop'], lambda_=param_pg['tam_pop'],
                                                    cxpb=param_pg['pb_cx'], mutpb=param_pg['pb_mut'],
                                                    ngen=param_pg['n_geracoes'],
                                                    stats=mstats, halloffame=halldafama, verbose=True)

        estatisticas = copiar_estatisticas(logbk=log)

    texec_media = (time.time() - tinit) / param_aux['n_exec']
    n_dt_media = var_aux['n_dt'] / param_aux['n_exec']
    n_simul_media = var_aux['n_simul'] / param_aux['n_exec']

    save = input('Save stats? (y) or (n)\n')
    if save is 'y':
        store_dict = {'parametros': param_store, 'hallsdafama': halldafama, 'estatisticas': estatisticas,
                      'texec': texec_media, 'n_dt': n_dt_media, 'n_simul': n_simul_media}
        store_file = open(store_filename, 'wb')
        pickle.dump(store_dict, store_file)
        store_file.close()
    else:
        pass

else:
    store_file = open(store_filename, 'rb')
    store_dict = pickle.load(store_file)
    store_file.close()

plotar_hits(store_dict['estatisticas'])
plotar_estatisticas_evolucao(store_dict['estatisticas'])

horas, resto = divmod(store_dict['texec'], 3600)
minutos, segundos = divmod(resto, 60)

print("Tempo de execução médio por rodada (h:m:s)", "{:0>2}:{:0>2}:{:05.2f}".format(int(horas),
                                                                                       int(minutos),
                                                                                       segundos))
print("Número de passos de tempo por rodada:", store_dict['n_dt'])
print("Número de simulações por rodada:", store_dict['n_simul'])
print("Processador: Ryzen 5 1600\n"
      "Placa-Mãe: ASRock A320M\n"
      "Memória RAM: 2x8GB Corsair Vengeance 3000MHz")

avaliar_individuo_multi(store_dict['hallsdafama'][0], graficos=True)
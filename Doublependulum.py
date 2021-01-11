#-*- coding:utf-8 -*-
"""
Documentation
"""

######################################################################################################

import os
import sys
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

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
import networkx as nx

import gym
import pybulletgym

sys.path.append('C:\\Softwares\\Miniconda3\\envs\\gp-openai-gym\\lib\\site-packages\\pybullet_envs\\bullet\\bullet_client')

######################################################################################################
#################################### PARÂMETROS ######################################################
######################################################################################################

matplotlib.rcParams.update({'font.size': 10})

random.seed(0)

param_pg = {
    'tam_pop': 500,
    'pb_cx': 0.75,
    'pb_mut': 0.05,
    'n_geracoes': 15,
    'tipo_apt': 1.0,
    'n_entradas': 11,
    'faixa_cst': [-1, 1],
    'n_episodios': 10,
    'camp_apt': 6,
    'camp_d': 1.2,
    'd_min': 2,
    'd_max': 5,
    'max_d_mut': 7,
    'limite_d': 17,
}

param_aux = {
    'n_exec': 10,
    'amb': gym.make('InvertedDoublePendulumMuJoCoEnv-v0'),
    'mujoco': True
}

param_graf = {
    'nomes_var_tex': (r"$s$", r'$\sin{(\theta)}$', r'$\sin{(\gamma)}$', r'$\cos{(\theta)}$', r'$\cos{(\gamma)}$',
                      r'$\dot{s}$', r'$\dot{\theta}$', r'$\dot{\gamma}$',
                      r'$f_r(s)$', r'$f_r(\theta)$', r'$f_r(\gamma)$'),
    'plot_var': ('Action', r"$s$", r'$\sin{(\theta)}$', r'$\sin{(\gamma)}$'),
    'operadores': {'add': r'$+$', 'sub': r'$-$', 'mul': r'$\times$', 'div': r'$\div$',
                   'gt': r'$>$', 'sr': r'$\sqrt{\,}$', 'sen': r'$\sin$', 'sgn': 'sgn',
                   'constante': r'$R$'},
    'hist_n_linhas': 3,
    'hist_n_cols': 2,
    'bins_min': 0,
    'bins_max': 11000,
    'bins_passo': 1000
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


def wrap(num):
    if num >= 1.0:
        return np.array([1.0], dtype='float32')
    elif num <= -1.0:
        return np.array([-1.0], dtype='float32')
    else:
        return np.array([num], dtype='float32')


def calc_custo_inst(obs=None, recomp=None, last=None):
    return 0


def calc_custo_acum(obs=None, recomp=None, ult_obs=None):
    return recomp


def calc_custo(custo_inst=None, custo_acum=None, obs=None, recomp=None):
    return custo_acum


def avaliar_individuo(ind, num_episodios=param_pg['n_episodios'], num_entradas=param_pg['n_entradas'],
                      ambiente=param_aux['amb'], nomes_var=param_graf['nomes_var_tex'],
                      plotar_var=param_graf['plot_var'], graficos=False, video=False, mujoco=param_aux['mujoco']):
    aptidoes = []
    tempo = 0
    dic_stats = {'Time': [], 'Output': [], 'Action': [], 'Cumulative Reward': []}
    funcao_de_controle = toolbox.compilar_individuo(ind)
    for obs in range(num_entradas):
        dic_stats['ARG' + str(obs)] = []
    if graficos or video:
        num_episodios = 1
    # if mujoco and video:
    #     ambiente.render()
    for episodio in range(num_episodios):
        var_aux['n_simul'] += 1
        tempo_ep = 0
        custo_acumulado = custo_instantaneo = 0
        termino = False
        observacao = ambiente.reset()
        while not termino:
            var_aux['n_dt'] += 1
            tempo_ep += 1
            resultado = funcao_de_controle(*tuple(observacao))
            acao = wrap(resultado)
            observacao, recompensa, termino, info = ambiente.step(acao)
            custo_instantaneo = calc_custo_inst()
            custo_acumulado += calc_custo_acum(recomp=recompensa)
            custo = calc_custo(custo_acum=custo_acumulado)
            if video and not mujoco:
                ambiente.render()
            if graficos:
                tempo += 1
                dic_stats['Time'].append(tempo), dic_stats['Output'].append(resultado),
                dic_stats['Action'].append(acao), dic_stats['Cumulative Reward'].append(custo_acumulado)
                for obs in range(num_entradas):
                    dic_stats['ARG' + str(obs)].append(observacao[obs])
        aptidoes.append(custo)
    aptidao_media = np.average(aptidoes)
    if graficos:
        plotar_vars_avaliacao_melhor(data=dic_stats, nomes_var=nomes_var,
                                     plotar_var=plotar_var, tempo_final=500)
    if not mujoco:
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
    plt.figure(figsize=(14,16))
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
    nx.draw_networkx_nodes(g, pos, node_size=1300, node_color='black')
    nx.draw_networkx_edges(g, pos)
    nx.draw_networkx_labels(g, pos, labels, font_color='white', font_size=15)
    plt.savefig('figs/fig_dp_tree.pdf')
    plt.show()


def plotar_vars_avaliacao(data=None, nomes_var=None, plotar_var=None, tempo_final=None):
    figura, eixo = plt.subplots(nrows=len(plotar_var), ncols=1, sharex='all')
    # data['Ação'] = data.pop('Acao')
    for n in range(len(nomes_var)):
        data[nomes_var[n]] = data.pop('ARG' + str(n))
    if len(plotar_var) < 2:
        eixo.plot('Time', plotar_var[0], data=data, ls='-.', marker='o', ms=4, color='blue', alpha=0.3)
        eixo.set_xlabel('Time')
        eixo.set_ylabel(plotar_var[0])
        eixo.grid()
    else:
        cores = ['b', 'g', 'r', 'm', 'c']
        for var in range(len(plotar_var)):
            eixo[var].plot('Time', plotar_var[var], data=data, ls='-.', marker='o', ms=4, color=cores[var], alpha=0.3)
            eixo[var].set_ylabel(plotar_var[var])
            eixo[var].grid(True)
    eixo[-1].set_xlabel('Time')  # nome apenas no último gráfico
    eixo[-1].set_xlim(0, tempo_final)


def plotar_vars_avaliacao_melhor(data=None, nomes_var=None, plotar_var=None, tempo_final=1000):
    figura, eixo = plt.subplots(nrows=len(plotar_var), ncols=1, sharex='all')
    data['Ação'] = data.pop('Acao')
    for n in range(len(nomes_var)):
        data[nomes_var[n]] = data.pop('ARG' + str(n))
    if len(plotar_var) < 2:
        eixo.plot('Tempo', plotar_var[0], data=data, ls='-', marker='.', ms=3, color='blue', alpha=0.5)
        eixo.set_xlabel('Tempo')
        eixo.set_ylabel(plotar_var[0])
        eixo.grid()
    else:
        cores = ['b', 'g', 'r', 'm', 'c']
        for var in range(len(plotar_var)):
            eixo[var].plot('Tempo', plotar_var[var], data=data, ls='-', marker='.', ms=3, color=cores[var], alpha=0.5)
            eixo[var].set_ylabel(plotar_var[var])
            eixo[var].grid(True)
    eixo[-1].set_xlabel('Tempo')  # nome apenas no último gráfico
    eixo[-1].set_xlim(0, tempo_final)


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
            eixo[i][j].set_title('Generation ' + str(graf[n]))
            eixo[i][j].set_ylabel('Count') if j is 0 else None
            eixo[i][j].set_xlabel('Fitness') if i is (nrows-1) else None
            n += 1


def plotar_estatisticas_evolucao(data=None):
    stats = ['aptidao', 'comprimento', 'complexidade']
    labels = ['Aptidão', 'Comprimento', 'Complexidade']
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


def avaliar_hdf(hdf, env):
    aptidoes = []
    for ind in hdf[:10]:
        aptidao, = avaliar_individuo(ind, num_episodios=100, ambiente=env)
        aptidoes.append(aptidao)
    aptidao_media_melhores = np.sum(aptidoes) / len(aptidoes)
    print("Aptidão média (10 primeiros do hall da fama): ", aptidao_media_melhores)
    return aptidao_media_melhores


def avaliar_melhor_hdf(hdf):
    best = -100000
    for idx, ind in enumerate(hdf):
        aptidao, = avaliar_individuo(ind, num_episodios=100)
        if aptidao > best:
            index = idx
            best = aptidao
    print("Melhor aptidão do hall da fama: ", best)
    print("Índice: ", index)
    return best


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
# ConjPrim.addPrimitive(np.sin, 1, name="sen")
ConjPrim.addEphemeralConstant("R", lambda: round(random.uniform(*param_pg['faixa_cst']), 2))


######################################################################################################
################################ FUNÇÕES INICIALIZAÇÃO, OPERAÇÃO GENÉTICA E AVALIAÇÃO ################
################n######################################################################################


toolbox = base.Toolbox()

creator.create("Aptidaomax", base.Fitness, weights=(param_pg['tipo_apt'],))
creator.create("Individuo", gp.PrimitiveTree, fitness=creator.Aptidaomax, pset=ConjPrim)

toolbox.register("gerar_expressao", gp.genHalfAndHalf, pset=ConjPrim, min_=param_pg['d_min'], max_=param_pg['d_max'])
toolbox.register("gerar_individuo", tools.initIterate, creator.Individuo, toolbox.gerar_expressao)
toolbox.register("gerar_populacao", tools.initRepeat, list, toolbox.gerar_individuo, param_pg['tam_pop'])

toolbox.register("compilar_individuo", gp.compile, pset=ConjPrim)

toolbox.register("evaluate", avaliar_individuo)
toolbox.register("select", tools.selDoubleTournament, fitness_first=False,
                 fitness_size=param_pg['camp_apt'], parsimony_size=param_pg['camp_d'])
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.gerar_expressao, pset=ConjPrim)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=param_pg['limite_d']))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=param_pg['limite_d']))


######################################################################################################
################################ LOG #################################################################
######################################################################################################


aptidao_lista = tools.Statistics(lambda ind: ind.fitness.values[0])
comprimento_lista = tools.Statistics(lambda ind: ind.height)
complexidade_lista = tools.Statistics(lambda ind: ind.__len__())

mstats = tools.MultiStatistics(aptidao=aptidao_lista, comprimento=comprimento_lista, complexidade=complexidade_lista)

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
        ocorrencias, const = contar_operadores(ConjPrim, populacao)

        estatisticas_lista.append(estatisticas)
        occ_lista.append(ocorrencias)
        const_lista.append(const)
        hdf_lista.append(halldafama)

    estatisticas_media, ocorrencias_media, const_media = calcular_media_exec(estatisticas_lista, occ_lista, const_lista)
    texec_media = (time.time() - tinit) / param_aux['n_exec']
    n_dt_media = var_aux['n_dt'] / param_aux['n_exec']
    n_simul_media = var_aux['n_simul'] / param_aux['n_exec']

    save = input('Save stats? (y) or (n)\n')
    if save is 'y':
        store_dict = {'parametros': param_store, 'hallsdafama': hdf_lista, 'estatisticas': estatisticas_media,
                      'ocorrencias': ocorrencias_media, 'constantes': const_media,
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
plt.savefig('figs/fig_dp_apt.pdf')
plotar_estatisticas_evolucao(store_dict['estatisticas'])
plotar_ocorrencias(store_dict['ocorrencias'])
avaliar_individuo(ind=store_dict['hallsdafama'][0][0], graficos=True, video=False)
plt.savefig('figs/fig_dp_control.pdf')
plotar_arvore(store_dict['hallsdafama'][0][0])

horas, resto = divmod(store_dict['texec'], 3600)
minutos, segundos = divmod(resto, 60)

print("Tempo de execução médio por rodada (h:m:s)", "{:0>2}:{:0>2}:{:05.2f}".format(int(horas),
                                                                                      int(minutos),
                                                                                      segundos))
print("Número médio de passos de tempo por rodada:", store_dict['n_dt'])
print("Número médio de simulações por rodada:", store_dict['n_simul'])
print("Valor médio das constantes por rodada:", np.average(store_dict['constantes']))
print("Processador: Ryzen 5 1600\n"
      "Placa-Mãe: ASRock A320M\n"
      "Memória RAM: 2x8GB Corsair Vengeance 3000MHz")

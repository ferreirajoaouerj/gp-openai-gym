from gym.envs.registration import register

register(id='Carrinho-v0',
         entry_point='gym_carrinho.envs:CarrinhoEnv')

# ################# REMOVER SE NECESSARIO
# register(id='CarrinhoGoal-v0',
#          entry_point='gym_carrinho.envs:CarrinhoGoalEnv',
#          max_episode_steps=1500)
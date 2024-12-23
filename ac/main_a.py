from ac_meta import *
from strategy_library import *
from search import *


def main(idx: indexinfo):
    groups_list = fetch_strategy_groups(idx.id)
    env = AnomalyMetricEnv(idx, groups_list, normal_value=(idx.max_normal_value - idx.min_normal_value) / 2,
                           max_deviation=50, stability=idx.max_normal_value - idx.min_normal_value)
    agent = MetaActorCriticAgent(env)

    value = [idx.outlier]
    agent.load_models('model/meta')
    agent.test(value)



from ac_no_meta import *
from ac_meta import *
from DQN import *
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm
import time

prop = fm.FontProperties(fname='C:\\Windows\\Fonts\\simsun.ttc', size=16)
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False


def meta():
    start_time1 = time.time()
    agent_meta = MetaActorCriticAgent(env)
    agent_meta.maml_train(meta_train_data)
    agent_meta.save_models('model/meta')
    agent_meta.load_models('model/meta')
    rewards_meta = agent_meta.test(test_data)
    end_time1 = time.time()

    start_time2 = time.time()
    agent_no_meta = ActorCriticAgent(env)
    agent_no_meta.train(meta_train_data)
    agent_no_meta.save_models('model/no_meta')
    agent_no_meta.load_models('model/no_meta')
    rewards_no_meta = agent_no_meta.test(test_data)
    end_time2 = time.time()

    start_time3 = time.time()
    agent_dqn = DQNAgent(env)
    agent_dqn.train(meta_train_data)
    agent_dqn.save_models('model/dqn')
    agent_dqn.load_models('model/dqn')
    rewards_dqn = agent_dqn.test(test_data)
    end_time3 = time.time()

    print("rewards_MAC.mean: ", np.mean(rewards_meta))
    print("time_MAC: ", end_time1 - start_time1)
    print("rewards_AC.mean: ", np.mean(rewards_no_meta))
    print("time_AC: ", end_time2 - start_time2)
    print("rewards_DQN.mean: ", np.mean(rewards_dqn))
    print("time_DQN: ", end_time3 - start_time3)

    dif1 = np.array(rewards_meta) - np.array(rewards_no_meta)
    dif2 = np.array(rewards_meta) - np.array(rewards_dqn)

    gs = gridspec.GridSpec(1, 1, height_ratios=[1], width_ratios=[1])
    fig = plt.figure(figsize=(26, 6))

    ax4 = fig.add_subplot(gs[0, 0])
    ax4.plot(dif1, marker='o', linestyle='-', color='b')
    title4 = ax4.set_title('Rewards MAC - Rewards AC')
    ax4.plot(dif2, marker='o', linestyle='-', color='r')
    plt.setp(title4, weight='bold', size=17)
    ax4.set_xlabel('测试数据', fontproperties=prop)
    ax4.set_ylabel('任务总奖励', fontproperties=prop)
    ax4.grid(True)

    # ax6 = fig.add_subplot(gs[1, 1])
    # ax6.axis('off')
    plt.subplots_adjust(wspace=0.45, hspace=0.4)
    plt.tight_layout()

    plt.show()


if __name__ == '__main__':
    low = -50.0
    high = 50.0

    np.random.seed(30)
    meta_train_data = np.random.uniform(low, high, 20)
    np.random.seed(70)
    test_data = np.random.uniform(low, high, 50)

    env = AnomalyMetricEnv(normal_value=0, max_deviation=50, stability=2)

    meta()

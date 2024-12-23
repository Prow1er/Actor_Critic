import json
import numpy as np
from ac_meta import *
from process import get_groups_list, fetch_strategy_groups
from strategy_library import *
from search import *
from threading import Timer


class TimeoutException(Exception):
    pass


class InputValidationException(Exception):
    pass


def validate_input(index_id, index_name, index_outlier, index_max, index_min):
    if not isinstance(index_id, int) or index_id <= 0:
        raise InputValidationException("指标id 必须是一个正整数.")
    if not isinstance(index_name, str) or len(index_name.strip()) == 0:
        raise InputValidationException("指标名称 不能为空.")
    if not isinstance(index_outlier, (int, float)):
        raise InputValidationException("指标异常值 必须是一个数字.")
    if not isinstance(index_max, (int, float)) or not isinstance(index_min, (int, float)):
        raise InputValidationException("指标最大正常值 和 指标最小正常值 必须是数字.")
    if index_max <= index_min:
        raise InputValidationException("指标最大正常值 必须大于 指标最小正常值.")


def timeout_handler():
    raise TimeoutException("Operation timed out.")


# 假设这是你的算法
def algorithm(index_id, index_name, index_outlier, index_max, index_min):
    try:
        # 验证输入
        validate_input(index_id, index_name, index_outlier, index_max, index_min)

        # 设置超时
        timer = Timer(20.0, timeout_handler)
        timer.start()

        idx = indexinfo(index_id, index_name, index_outlier, max_normal_value=index_max, min_normal_value=index_min)
        groups_list = fetch_strategy_groups(idx.id)
        groups_list = get_groups_list(groups_list, idx)

        env = AnomalyMetricEnv(idx, groups_list, normal_value=(idx.max_normal_value - idx.min_normal_value) / 2,
                               max_deviation=50, stability=idx.max_normal_value - idx.min_normal_value)
        agent = MetaActorCriticAgent(env)

        np.random.seed(20)
        meta_train_data = np.random.uniform(-10, 0, 50)

        agent.maml_train(meta_train_data)
        agent.save_models('model/meta')
        value = [index_outlier]
        agent.load_models('model/meta')
        rewards, out_dict = agent.test(value)

        out_dict = {
            k: (v.tolist() if isinstance(v, np.ndarray) else v.item() if isinstance(v, (np.number, np.bool_)) else v)
            for k, v in out_dict.items()}

        response = {
            "code": 200,
            "data": {"result": out_dict},
            "message": "正常"
        }
        return response

    except InputValidationException as ive:
        response = {
            "code": 401,
            "data": {"result": "输入验证失败"},
            "message": f"异常: {str(ive)}"
        }
        return response
    except TimeoutException:
        response = {
            "code": 400,
            "data": {"result": "没有针对此异常值的策略！"},
            "message": "异常"
        }
        return response
    except Exception as e:
        response = {
            "code": 400,
            "data": {"result": "没有针对此异常值的策略！"},
            "message": f"异常: {str(e)}"
        }
        return response
    finally:
        if 'timer' in locals():
            timer.cancel()  # 确保清除计时器


class Service:

    def __init__(self):
        pass

    def process(self, index_id, index_name, index_outlier, index_max, index_min):
        # 将值传入你的算法 并返回输出结果
        result = algorithm(index_id, index_name, index_outlier, index_max, index_min)
        return result


service = Service()
result = service.process(1, '利税比', -9, 2, -2)
print(result)

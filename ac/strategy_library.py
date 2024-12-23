from fetch import fetch_data
from indexinfo import *
from search import *


class StrategyLibrary:
    def __init__(self, index: indexinfo, groups_list_re: dict[int:list[Strategy_Group]]):

        self.strategies = {
            "[-50%, -20%)": groups_list_re[0],
            "[-20%, -10%)": groups_list_re[1],
            "[-10%, -5%)": groups_list_re[2],
            "[-5%, 0%)": groups_list_re[3],
            "[0%, 5%)": groups_list_re[4],
            "[5%, 10%)": groups_list_re[5],
            "[10%, 20%)": groups_list_re[6],
            "[20%, 50%)": groups_list_re[7],
        }

    def get_strategies(self, value):
        if -50 <= value < -20:
            return self.strategies["[-50%, -20%)"]
        if -20 <= value < -10:
            return self.strategies["[-20%, -10%)"]
        elif -10 <= value < -5:
            return self.strategies["[-10%, -5%)"]
        elif -5 <= value < 0:
            return self.strategies["[-5%, 0%)"]
        elif 0 <= value < 5:
            return self.strategies["[0%, 5%)"]
        elif 5 <= value < 10:
            return self.strategies["[5%, 10%)"]
        elif 10 <= value < 20:
            return self.strategies["[10%, 20%)"]
        elif 20 <= value < 50:
            return self.strategies["[20%, 50%)"]
        else:
            return []

    def get_all(self):
        # for key, value in self.strategies.items():
        #     print(f'key:{key},value:{value}')
        return self.strategies


if __name__ == '__main__':
    index = indexinfo(1, "利税比", -12, 2, -2)
    s_l = StrategyLibrary(index, fetch_strategy_groups(index.id))
    s_l.get_all()

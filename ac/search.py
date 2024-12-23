from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager

# 数据库连接配置
DB_URL = "mysql+pymysql://test:123456@129.211.189.196:3306/bi"
engine = create_engine(DB_URL)
Session = sessionmaker(bind=engine)

# 定义目标类
class Strategy_Group:
    def __init__(self, name, effect, strategies):
        self.name = name
        self.effect = effect
        self.strategies = strategies

    def __repr__(self):
        return f"StrategyGroup(name={self.name}, effect={self.effect}, strategies={self.strategies})"

    def apply(self, value):
        return value + self.effect

# 上下文管理器来管理会话
@contextmanager
def get_session():
    session = Session()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

# 查询和构建逻辑
def fetch_strategy_groups(index_id):
    result = {}

    with get_session() as session:
        try:
            # Step 1: 查询与 index_id 相关的 stra_index_value.id
            stra_index_values = session.execute(text(
                """
                SELECT id
                FROM stra_index_value
                WHERE index_id = :index_id
                """), {"index_id": index_id}
            ).fetchall()

            for index_value in stra_index_values:
                index_value_id = index_value[0]

                # 临时存储当前 index_value_id 对应的 StrategyGroup 列表
                strategy_groups = []

                # Step 2: 查询 cor_abnormal_group 中的 group_id
                abnormal_groups = session.execute(text(
                    """
                    SELECT group_id
                    FROM cor_abnormal_group
                    WHERE index_value_id = :index_value_id
                    """), {"index_value_id": index_value_id}
                ).fetchall()

                for abnormal_group in abnormal_groups:
                    group_id = abnormal_group[0]

                    # Step 3: 查询 cor_strategy_group 中的 strategy_id
                    strategies = session.execute(text(
                        """
                        SELECT strategy_id
                        FROM cor_strategy_group
                        WHERE group_id = :group_id
                        """), {"group_id": group_id}
                    ).fetchall()

                    strategies_dict = {}

                    # Step 4: 查询 stra_strategy 中的 name 和 scores
                    for strategy in strategies:
                        strategy_id = strategy[0]
                        strategy_data = session.execute(text(
                            """
                            SELECT dept_id, big_score, small_score, difficulty_small_score
                            FROM stra_strategy
                            WHERE id = :strategy_id
                            """), {"strategy_id": strategy_id}
                        ).fetchone()

                        if strategy_data is not None:
                            strategies_dict[strategy_data.dept_id] = [
                                strategy_data.big_score,
                                strategy_data.small_score,
                                strategy_data.difficulty_small_score
                            ]

                    # 构建 StrategyGroup
                    group_name = f"Group-{group_id}"
                    effect = None  # 这里应该根据你的业务逻辑设置 effect 的值
                    strategy_group = Strategy_Group(group_name, effect, strategies_dict)
                    strategy_groups.append(strategy_group)

                # 将当前 index_value_id 对应的 StrategyGroup 列表加入最终结果
                result[index_value_id] = strategy_groups

        except Exception as e:
            print(f"An error occurred while fetching strategy groups: {e}")
            raise

    return result

# 示例调用
if __name__ == "__main__":
    index_id = 1  # 替换为实际的 index_id
    try:
        result = fetch_strategy_groups(index_id)
        print(result)
        # 输出结果
        for idx, (index_value_id, strategy_groups) in enumerate(result.items(), start=1):
            print(f"Index Value {idx}:")
            for group in strategy_groups:
                print(f"  {group}")
    except Exception as e:
        print(f"Error during processing: {e}")
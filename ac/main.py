# ---encoding:utf-8---
# @Time    : 2024/11/21 09:57
# @Author  : AJAXXJ
# @Email   : 1751353695@qq.com
# @Project : actor_critic
# @Software: PyCharm
from fastapi import Body
from fastapi import FastAPI
from pydantic import BaseModel
from service import *

app = FastAPI()


# 指标信息
class INDEX(BaseModel):
    # 指标id
    index_id: int
    # 指标名称
    index_name: str
    # 指标异常值
    index_outlier: float
    # 指标最大正常值
    index_max: float
    # 指标最小正常值
    index_min: float


@app.post("/actor_critic")
async def tsc_predict(index: INDEX = Body(embed=True)):
    service = Service()
    result = service.process(index_id=index.index_id, index_name=index.index_name, index_outlier=index.index_outlier,
                             index_max=index.index_max, index_min=index.index_min)
    response = {'index_result': result}
    return response

import json

json_file = open("config.json","r",encoding="utf-8")    # 打开json文件
config = json.load(json_file)           # 解码json数据
assert isinstance(config, dict)     # ininstance()判断返回的是否为字典类型的数据，否则assert返回一个错误提示


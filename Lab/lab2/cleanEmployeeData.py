import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def cleanData(random_state = None):
    data = pd.read_csv("employees_dataset.csv")

    # 简单数据清理
    # m.eng = Master of Engineer,
    data.loc[data['degree']=="m.eng",'degree'] = 'master'

    # 将学位转换为数值
    degree_mapping = {"bachelor":0, "master":1, "phd":2}
    data['degreeID'] = data['degree'].map(degree_mapping)

    # 将毕业院校数字化

    # 985 和 英美等著名院校
    edu_top = {"shanghai jiao tong university",
    "nanjing university",
    "tongji university",
    "harbin institute of technology",
    "wuhan university",
    "beihang university",
    "fudan university",
    "huazhong university of science and technology",
    "university of essex",
    "santa clara university",
    "northwestern polytechnic university",
    "beijing institute of technology",
    "tsinghua university",
    "sun yat-sen university",
    "university of california  berkeley - walter a. haas school of business",
    "renmin university of china",
    "east china normal university",
    "xiamen university",
    "southeast university",
    "the university of manchester",
    "zhejiang university"}

    # 211
    edu_211 = {"beijing university of post and telecommunications",
    "shanghai university",
    "xidian university",
    "southwest jiaotong university",
    "nanchang university",
    "southwest china normal university",
    "nanjing university of science and technology",
    "inner mongolia university",
    "wuhan university of science and technology",
    "dalian university of technology",
    "the university of hong kong",
    "xi'dian university"}

    data['edu_top'] = data['education'].map(lambda x : 1 if x in edu_top else 0)
    data['edu_211'] = data['education'].map(lambda x : 1 if x in edu_211 else 0)

    # 技能取前3个并编码
    data['skills'].replace('c++/c', 'c/c++')

    skill_topN = 3
    df_skills = data['skills'].str.split(';', n=skill_topN,expand=True).iloc[:,0:skill_topN].fillna('N/A')

    labels = pd.factorize(df_skills.values.reshape(-1, ))
    skill_col_name = ['skills_' + str(i) for i in range(skill_topN)]
    df_skills_id = pd.DataFrame(labels[0].reshape(-1, 3), columns=skill_col_name)
    data = pd.concat([data, df_skills_id], axis=1)

    # 对position 进行factorize
    pos_factor = pd.factorize(data['position'])
    data['position_id'] = pos_factor[0]

    # 对position进行onehot
    # dumm = pd.get_dummies(data['position'],prefix="position")

    # 划分训练集和测试集
    sel_col = ['degreeID', 'edu_top', 'edu_211'] + skill_col_name
    X = data[sel_col]
    y = data['position_id']

    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=random_state,stratify=y)

    return train_x, test_x, train_y, test_y, pos_factor[1]

if __name__ == "__main__":
    train_x, test_x, train_y, test_y, pos_mapping = cleanData(random_state=3)
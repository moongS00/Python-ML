# %% [markdown]
# ```
# <컬럼의 의미>
# 
# pclass : 객실등급
# survived : 생존유무
# sex : 성별
# age : 나이
# sibsp : 형제 혹은 부부의 수
# parch : 부모 혹은 자녀의 수
# fare : 지불한 요금
# boat : 탈출했다면 탑승한 보트 번호
# ```

# %% [markdown]
# # 1. 데이터 탐색적 분석 - EDA

# %%
#!pip install plotly_express

# %%
# 데이터 읽기

import pandas as pd

titanic = pd.read_excel("../data/titanic.xls")
titanic.head()

# %% [markdown]
# ## 1) 생존 상황

# %%
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# %%
# value_counts() : 값의 개수를 세주는 모듈

titanic['survived'].value_counts()

# %%
# 파이차트로 그리기

titanic['survived'].value_counts().plot.pie();

# %%
titanic['survived'].value_counts().plot.pie(explode=[0, 0.1], autopct="%1.1f%%", shadow=True)

# %% [markdown]
# - explode : 조각 간 거리
# - autopct : 숫자를 입력하는 옵션, (1.1f : 소숫점 첫째자리까지)
# - ax
# - shdow : 파이차트를 3D 처럼 보이게 하는 그림자가 생김 (아주 작게 생김 ㅋㅋ)

# %%
f, ax = plt.subplots(1, 2, figsize=(16, 8))

titanic['survived'].value_counts().plot.pie(explode=[0, 0.1], autopct="%1.1f%%", ax=ax[0], shadow=True)

ax[0].set_title("Pie plt - Survived")
ax[0].set_ylabel("")
sns.countplot(x="survived", data=titanic, ax=ax[1])
ax[1].set_title("Count plot - Survived")

plt.show()

# %% [markdown]
# - f, ax = plt.subplots(1, 2, figsize=(18, 8))  : 그래프를 2개 그릴 예정. figsize=(18, 8) 를 1행 2열로 준비.  
# - f, ax는 subplots가 반환하는 값. ax는 각 그래프의 속성을 의미한다.

# %% [markdown]
# - 38.2%의 생존률, 약 500명의 사람이 살아남았다

# %% [markdown]
# ## 2) 성별에 따른 생존 상황

# %%
f, ax = plt.subplots(1, 2, figsize=(16, 8))

sns.countplot(x="sex", data=titanic, ax=ax[0])
ax[0].set_title("Count of Passengers of Sex")
ax[0].set_ylabel("")

sns.countplot(x="sex", hue='survived', data=titanic, ax=ax[1])
ax[1].set_title("Sex : Survived and Unsurvived")

plt.show()

# %% [markdown]
# - 여성 승객은 약 450명, 생존자는 약 350명인 반면, 남성 승객은 약 800명, 생존자는 약 200명이다.
# - 남성의 생존 가능성이 더 낮다고 볼수 있다.

# %% [markdown]
# ## 3) 경제력 대비 생존률

# %%
pd.crosstab(titanic['pclass'], titanic['survived'], margins=True)

# %% [markdown]
# - 1등실의 생존 가능성이 아주 높다.
# - 그런데 여성의 생존률도 높다.
# - 그럼 1등실에는 여성이 많이 타고 있었을까?

# %%
grid = sns.FacetGrid(titanic, row='pclass', col='sex', height=4, aspect=2)
grid.map(plt.hist, 'age', alpha=0.8, bins=20) # 히스토그램
grid.add_legend()

# %% [markdown]
# - 1등실, 2등실 모두 성별 별 나이 분포가 고르다. 
# - 3등실에는 20대 남성이 많았다.

# %%
# 승객의 나이분포를 히스토그램으로 알아보자.

import plotly.express as px

fig = px.histogram(titanic, x="age")
fig.show()

# %% [markdown]
# - 아이들 & 20,30대 청년층이 많았다

# %% [markdown]
# ## 4) 등실별 연령별 생존률

# %%
grid = sns.FacetGrid(titanic, col="survived", row="pclass", height=3, aspect=2)
grid.map(plt.hist, "age", alpha=0.5, bins=20)
grid.add_legend();

# %%
# 나이를 5단계로 정리하기 : pd의 cut 이용

titanic["age_cat"] = pd.cut(titanic['age'], bins=[0, 7, 15, 30, 60, 100], include_lowest=True,       # bins: 구간설정
                              labels=["baby", "teen", "young", "adult", "old"])

titanic.head()

# %% [markdown]
# ## 5) 나이, 성별, 등실별 생존율

# %%
plt.figure(figsize=(12, 6))

plt.subplot(131)  # 1행 3열 중 첫번째
sns.barplot(x='pclass', y='survived', data=titanic)

plt.subplot(132)  # 1행 3열 중 두번째
sns.barplot(x='age_cat', y='survived', data=titanic)

plt.subplot(133)  # 1행 3열 중 세번째
sns.barplot(x='sex', y='survived', data=titanic)

plt.subplots_adjust(top=1, bottom=0.1, left=0.1, right=1, hspace=0.5, wspace=0.5)

# %% [markdown]
# - 어리고, 여성이고, 1등실일수록 생존하기 유리했을까?

# %% [markdown]
# ## 6) 남/여 나이별 생존 상황 자세히 관찰

# %%
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

w = titanic[titanic['sex'] == 'female']
m = titanic[titanic['sex'] == 'male']

ax = sns.distplot(w[w['survived']==1]['age'], bins=20, label='survived', ax=axes[0], kde=False) # kde=False : 밀도함수 제거
ax = sns.distplot(w[w['survived']==0]['age'], bins=40, label='not_survived', ax=axes[0], kde=False)
ax.legend();
ax.set_title("Female")

ax = sns.distplot(m[m['survived']==1]['age'], bins=18, label='survived', ax=axes[1], kde=False)
ax = sns.distplot(m[m['survived']==0]['age'], bins=40, label='not_survived', ax=axes[1], kde=False) # bins 값이 클수록 잘게 나눈다
ax.legend();
ax.set_title("Male")

# %%
# 탑승객 이름으로 사회적 신분 유추 가능(Miss, Mr, Mrs, Master 등)

import re

title = []

for i, da in titanic.iterrows():
    tmp = da["name"]
    title.append(re.search('\,\s\w+(\s\w+)?\.', tmp).group()[2:-1]) # ,로 시작 + 단어 여러개 + 단어 여러개 (개수 미정) + . 으로 끝난다

title

# %%
# titanic 데이터에 넣기

titanic["title"] = title
titanic.head()

# %%
pd.crosstab(titanic['title'], titanic['sex'])

# %%
titanic['title'].unique()

# %%
titanic['title'] = titanic['title'].replace("Mlle", "Miss")  # 동의어 정리
titanic['title'] = titanic['title'].replace("Mme", "Miss")
titanic['title'] = titanic['title'].replace("Ms", "Miss")

rare_f = ['Dona', 'Lady', 'the Countess'] # 여성 귀족 이름만 따로 정리
rare_m = ['Capt', 'Col', 'Don', 'Major', 'Sir', 'Rev', 'Dr', 'Master', 'Jonkheer'] # 남성 귀족 이름만 따로 정리

# %%
for i in rare_f:
    titanic['title'] = titanic['title'].replace(i, 'Rare_f')

for i in rare_m:
    titanic['title'] = titanic['title'].replace(i, 'Rare_m')

# %%
titanic['title'].unique()

# %%
titanic[['title', 'survived']].groupby(['title'], as_index=False).mean()

# %% [markdown]
# - 귀족 남성은 평민 여성들보다도 생존율이 훨씬 낮다

# %% [markdown]
# # 2. 머신러닝을 이용한 생존자 예측

# %%
titanic.info()

# %% [markdown]
# ## 1) 문자열을 숫자형으로 변경
# - 머신러닝을 사용하려면 dtype이 숫자여야 한다.
# - sex가 문자열이므로, 숫자로 바꿔준다 : Label Encode 사용

# %%
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(titanic['sex'])  # 문자를 숫자로 변환해줌

titanic['gender'] = le.transform(titanic['sex'])
titanic.head()  # 여자=0, 남자=1

# %% [markdown]
# ## 2) 결측치 버리기
# - 사용할 컬럼에 결측치가 있다면, 쿨하게 보내주자..

# %%
titanic = titanic[titanic['age'].notnull()]
titanic = titanic[titanic['fare'].notnull()]

titanic.info()

# %% [markdown]
# ## 3) 상관관계 확인
# - survived 와 상관관계가 큰 것을 알아보자

# %%
cor = titanic.corr().round(1)
sns.heatmap(data=cor, annot=True, cmap='bwr')

# %% [markdown]
# ## 4) 데이터 분할

# %%
from sklearn.model_selection import train_test_split

X = titanic[['pclass', 'age', 'sibsp', 'parch', 'fare', 'gender']]  # 특성
y = titanic['survived'] # 분할할 데이터

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)

# %% [markdown]
# ## 5) Decision Tree

# %%
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

dt = DecisionTreeClassifier(max_depth=4, random_state=13)
dt.fit(X_train, y_train)

pred = dt.predict(X_test)
print(accuracy_score(y_test, pred))

# %% [markdown]
# ## 6) 디카프리오의 생존율을 구해보자!

# %%
import numpy as np

dica = np.array([[3, 18, 0, 0, 5, 1]])  # [['pclass', 'age', 'sibsp', 'parch', 'fare', 'gender']]
print("Decaprio : ", dt.predict_proba(dica)[0,1])

# %% [markdown]
# ## 7) 케이트윈슬렛의 생존율을 구해보자!
# 

# %%
win = np.array([[1, 16, 1, 1, 100, 0]])  # [['pclass', 'age', 'sibsp', 'parch', 'fare', 'gender']]
print("Winslet : ", dt.predict_proba(win)[0,1])

# %%




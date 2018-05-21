from __future__ import print_function
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
from statsmodels.tsa.arima_model import ARMA

##历史数据
# ##个人成交最低价
dta= [10000, 11600, 14000, 18700, 23900, 10000,10700, 11500, 11600, 12300
       , 13100, 14500, 16000, 14800,10300, 12000, 13200, 14500, 16800, 20200, 26000, 35000,
       10000, 13500, 16000, 19000, 24600, 21000, 19500, 21100, 23800, 25000, 23400, 10100, 15100, 18100, 19500, 18600
       , 15300, 15100, 15900, 17300, 18900, 21500, 23900, 25000, 26800, 28800, 30000, 30700, 18000
       , 21000, 22800, 25300, 32100]
##个人平均成交价
# dta = [22822, 14138, 11067, 15743, 20631, 26599, 30802, 16384, 12777
#        , 14074, 15391, 17024, 18310, 16654, 12382,13137, 14331, 15436, 17798, 21506, 27672
#        , 37805, 36231, 16886, 17487, 20609, 27077, 25727, 21884, 23315, 25701, 27127, 28541, 24324, 17330, 19614,
#        21551, 22300, 20591, 17508, 17419, 18358, 20127, 22996, 25498, 26668, 28561, 30535, 32449, 34046, 32312,
#        25213, 24560, 26939, 34455]

dta=np.array(dta,dtype=np.float)
dta=pd.Series(dta)
dta.index = pd.Index(sm.tsa.datetools.dates_from_range('2001','2055'))
dta.plot(figsize=(12,8))
fig = plt.figure(figsize=(12,8))
ax1= fig.add_subplot(111)
##差分运算
diff1 = dta.diff(1)
diff1.plot(ax=ax1)
fig = plt.figure(figsize=(12,8))
ax2= fig.add_subplot(111)
diff2 = dta.diff(2)
diff2.plot(ax=ax2)
##
diff1= dta.diff(1)
fig = plt.figure(figsize=(12,8))
ax1=fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(dta,lags=40,ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(dta,lags=40,ax=ax2)
###ARIMA模型的建立
arma_mod20 = sm.tsa.ARMA(dta,order=(7,0)).fit()
print('first',arma_mod20.aic,arma_mod20.bic,arma_mod20.hqic)
arma_mod30 = sm.tsa.ARMA(dta,(0,1)).fit()
print('second',arma_mod30.aic,arma_mod30.bic,arma_mod30.hqic)
arma_mod40 = sm.tsa.ARMA(dta,order=(7,1)).fit()
print('third',arma_mod40.aic,arma_mod40.bic,arma_mod40.hqic)
arma_mod50 = sm.tsa.ARMA(dta,(8,0)).fit()
print('fourth',arma_mod50.aic,arma_mod50.bic,arma_mod50.hqic)
##预测值
predict_sunspots = arma_mod20.predict('2056', '2057', dynamic=True)
print(predict_sunspots)
##画图
fig, ax = plt.subplots(figsize=(12, 8))
ax = dta.ix['2001':].plot(ax=ax)
predict_sunspots.plot(ax=ax)
plt.show()

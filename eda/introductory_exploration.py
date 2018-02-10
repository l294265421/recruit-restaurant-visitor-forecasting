from data.raw_data2 import *
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


plt.figure(figsize= (10,7))
goldenweek1 = pd.date_range('2016-04-29','2016-5-5')
goldenweek2 = pd.date_range('2017-04-29','2017-5-7')
plt.subplot(2,1,1)
plt.axvspan(goldenweek1[0],goldenweek1[-1], color = 'y',
            alpha=0.5, label = 'Golden week')
plt.axvspan(goldenweek2[0],goldenweek2[-1], color = 'y',
            alpha=0.5)
plt.scatter(list(date_info.calendar_datetime.loc[date_info.holiday_flg == 1]),
            date_info.holiday_flg.loc[date_info.holiday_flg == 1],
            color = 'r', label = 'Holidays', marker = '^')
air_daily_reservations = air_reserve[['visit_datetime','reserve_visitors']].set_index('visit_datetime').resample('d').sum()
plt.plot(air_daily_reservations.index, air_daily_reservations.reserve_visitors, alpha = 0.5, label = 'AIR')
plt.xlabel('Reservation Date')
plt.ylabel('Number of visitors')
plt.legend()
plt.subplot(2,1,2)
plt.axvspan(goldenweek1[0],goldenweek1[-1], color = 'y',
            alpha=0.5, label = 'Golden week')
plt.axvspan(goldenweek2[0],goldenweek2[-1], color = 'y',
            alpha=0.5)
plt.scatter(list(date_info.calendar_datetime.loc[date_info.holiday_flg == 1]),
            date_info.holiday_flg.loc[date_info.holiday_flg == 1],
            color = 'r', label = 'Holidays', marker = '^')
hpg_daily_reservations = hpg_reserve[['visit_datetime','reserve_visitors']].set_index('visit_datetime').resample('d').sum()
plt.plot(hpg_daily_reservations.index, hpg_daily_reservations.reserve_visitors, alpha = 0.5, label = 'HPG')
plt.xlabel('Reservation Date')
plt.ylabel('Number of visitors')
plt.legend()
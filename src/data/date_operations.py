from datetime import date, datetime
import numpy as np

def get_season(now):
    Y = 2000 # dummy leap year to allow input X-02-29 (leap day)
    seasons = [('winter', (date(Y,  1,  1),  date(Y,  3, 20))),
               ('spring', (date(Y,  3, 21),  date(Y,  6, 20))),
               ('summer', (date(Y,  6, 21),  date(Y,  9, 22))),
               ('autumn', (date(Y,  9, 23),  date(Y, 12, 20))),
               ('winter', (date(Y, 12, 21),  date(Y, 12, 31)))]

    if isinstance(now, datetime):
        now = now.date()
    now = now.replace(year=Y)
    return next(season for season, (start, end) in seasons
                if start <= now <= end)


def findZeroTempDay(arr):
  csd = 0 #consecutive subzero days
  fsdos = np.nan #first subzero day of seq
  for t in range(int(np.round(arr.shape[0]/2)), arr.shape[0]):
      if arr[t] < 0:
          if csd == 0:
              fsdos = t
          csd += 1
      else:
          fsdos = np.nan
          csd = 0
      if csd >= 5:
          return fsdos 
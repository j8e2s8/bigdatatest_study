# 출동시간과 도착시간 차이가 평균적으로 가장 오래 걸린 소방서의 시간을 분으로 변환해 출력하시오. (반올림 후 정수 출력)
import pandas as pd
df = pd.read_csv('data6-1-1.csv')
df.head()

df['차이'] = pd.to_datetime(df['도착시간'])-pd.to_datetime(df['출동시간'])
df.head()

df['차이'] = df['차이'].dt.total_seconds()/60

group_df = df.groupby('소방서').agg(mean_min = ('차이', 'mean'))

group_df.sort_values('mean_min',ascending=False).head(1)




# 학교에서 교사 한 명당 맡은 학생 수가 가장 많은 학교를 찾고, 그 학교의 전체 교사의 수를 구하여라.(정수 출력)



# 연도별로 총 범죄 건수(범죄유형의 총합)의 월평균 값을 구한 후 그 값이 가장 큰 연도를 찾아, 해당 연도의 총 범죄 건수의 월평균 값을 출력하시오. (반올림하여 정수로 출력)
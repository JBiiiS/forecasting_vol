import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import FinanceDataReader as fdr  # type: ignore



def clean_price_data(price_series, no_minus = False):
    # 1. Pandas Series로 변환 (아직 아니라면)
    if not isinstance(price_series, (pd.DataFrame,pd.Series)):
        try:
            s = pd.Series(price_series)
        
        
        except:
        
            try:
                s = pd.DataFrame(price_series)  
        
            except Exception as e:
                raise ValueError(f"cannot be pd: {e} // {type(e)}")
            
    
    s = price_series
    

    # 2. 어떤 데이터 중 0 또는 음수값이 있다면 NaN으로 처리 (비정상 데이터 제거)
    if no_minus == True:
        s[s <= 0] = np.nan
        print(f'The number of <= 0 data : {s.isnull().sum()}')

    # 3. 선형 보간 (Linear Interpolation)
    # 주변 가격의 평균으로 메꿈으로써 수익률의 급격한 왜곡 방지
    s = s.interpolate(method='linear')

    # 4. 양 끝단에 남은 결측치 처리 (보통 bfill/ffill로 마무리)
    s = s.bfill().ffill()

    
    # 5. data 개수: 3000개 이하
    # print(f'The number of data points : {len(s)}')
    # if len(s) > 3000:
    #     s = s.tail(3000)

    return s


#%%
import os
import json
import warnings
import argparse

import numpy as np
import pandas as pd
import geopandas as gpd
import yaml
from typing import Dict

from tabulate import tabulate
# from IPython.display import set_matplotlib_formats
# set_matplotlib_formats('png', 'pdf')
from shapely.geometry import LineString

from model.pymcdm import methods as mcdm_methods
from model.pymcdm import weights as mcdm_weights
from model.pymcdm import normalizations as norm
from model.pymcdm import correlations as corr
from model.pymcdm.helpers import rankdata, rrankdata


warnings.simplefilter('ignore')
warnings.filterwarnings(action='ignore')

#%%

def load_config(config_filename: str, CONFIG_PATH=os.path.join('config'), **kwargs) -> Dict:
    print(f'config_file directoty: {os.path.join(CONFIG_PATH, config_filename)}')
    with open(os.path.join(CONFIG_PATH, config_filename), encoding='utf-8') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg


# 주의 대상 "휠체어"가 포함된 행 반환 함수
def data_load(cfg):
    DATASET_PATH =  os.path.join('dataset')
    EU_dataset_nm = cfg['EU_dataset_nm']
    df_barr = pd.read_csv(os.path.join(DATASET_PATH, EU_dataset_nm))

    # 위치 좌표를 담을 리스트 초기화
    barrier_point = []
    for i in range(len(df_barr)):
        barrier_point += [[df_barr['위도'][i]] + [df_barr['경도'][i]]]

    # 데이터프레임의 모든 요인을 저장할 집합 초기화
    factor = set()

    # '요인' 열을 반복하며 요인을 추출하고 집합에 추가
    for i in df_barr['요인']:
        if i == i:
            factor_list = i.split(',')
            for j in factor_list:
                factor.add(j)

    # 모든 요인을 열로 추가하고, 해당 요인이 있는 경우 해당 열 값을 1로 설정
    for i in factor:
        df_barr[i] = 0
        
    #요인이 'nan'인 경우 삭제
    df_barr = df_barr.dropna(subset=['요인'])

    # 각 요인이 있는 경우 해당 열 값을 1로 설정
    for i in factor:
        df_barr.loc[df_barr['요인'].str.contains(i), i] = 1

     # '주의대상' 열에 '휠체어'가 포함된 행만 선택하여 반환
    df_barr_target = df_barr[df_barr['주의대상'].str.contains('휠체어')]
    return df_barr_target


# 경로 주변의 장애물 요인계산. 요인들에 대한 가중치를 적용하여 계산된 값을 반환
def route2factor(cfg, df_barr_target, route_):#route를 geopandas로 변경 후, 해당 route에 있는 factor개수 counting
    # 입력 매개변수:
    # - cfg: 설정 및 가중치 정보를 담은 딕셔너리
    # - df_barr_target: 휠체어 관련 요인 데이터를 담은 데이터프레임
    # - route_: 경로 좌표 정보
    
    df_barr = df_barr_target
    weight = cfg['Weight']
    coor_1m = (1/88.74/1000) #약 1m

    # 경로 좌표를 이용하여 GeoPandas GeoDataFrame 생성
    df = pd.DataFrame(route_)
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df[0], df[1]),  crs="EPSG:4326")
    gdf['route'] = 1
    gdf_1 = gdf.groupby(['route'])['geometry'].apply(lambda x:LineString(x.tolist()))
    gdf_1 = gpd.GeoDataFrame(gdf_1, geometry='geometry')
    # gdf_1.buffer(0.0001).plot()


     # 요인 집합 초기화 및 요인 데이터프레임 전처리
    factor = set()    
    for i in df_barr['요인']:
        if i == i:
            factor_list = i.split(',')
            for j in factor_list:
                factor.add(j)
    for i in factor:
        df_barr[i] = 0
    df_barr = df_barr.dropna(subset=['요인'])#요인이 'nan'인 경우 삭제

    for i in factor:
        df_barr.loc[df_barr['요인'].str.contains(i), i] = 1

    # 휠체어 관련 요인 데이터프레임을 GeoPandas GeoDataFrame으로 변환
    gdf_barr = gpd.GeoDataFrame(
        df_barr, geometry=gpd.points_from_xy(df_barr['위도'], df_barr['경도']),  crs="EPSG:4326")

    # 경로 주변의 10미터 반경 버퍼 생성. 반경 약 10m의 장애 요인들 고려
    gdf_buffer = gpd.GeoDataFrame(gdf_1.buffer(coor_1m*10), geometry=0, crs="EPSG:4326") 

    # 버퍼 내 포함된 장애물 계산
    polygons_contains = gpd.sjoin(gdf_buffer, gdf_barr , op='contains')     

    # 장애물 요인을 요약하여 데이터프레임 생성
    df_factor = pd.DataFrame(polygons_contains[['안내시설',	'경사',	'통행폭',	'높이',	'돌출물',	'단차',	'마감']].sum()).T

    # 가중치를 곱하여 요인 값 조정  (weight 정의 dir : model/pymcdm/methods/weight.py)
    for key in list(weight.keys()):
        df_factor[key] = df_factor[key].apply(lambda x: weight[key]*x)

     # 요인 데이터프레임 출력 후 반환
    print(df_factor)
    return df_factor


# route2factor 함수를 사용하여 각 경로에 대한 장애물 요인을 계산하고 데이터프레임으로 반환
def route_load(cfg, route_path):
    # 입력 매개변수:
    # - cfg: 설정 및 가중치 정보를 담은 딕셔너리
    # - route_path: 경로 데이터 JSON 파일의 경로

    with open(os.path.join(route_path), 'r') as f:
        json_data = json.load(f)
    routes = json_data['routes']
    for idx, _ in enumerate(routes):
        globals()[f'route_{idx}'] = []
        for i in range(len(json_data["routes"][idx]["routeItems"])):
            globals()[f'route_{idx}'] += json_data["routes"][idx]["routeItems"][i]["path"]
        print(f'route preprocessing for route_{idx}')
        if idx == 0:
            check_df_0 = route2factor(cfg, df_barr_target, globals()[f'route_{idx}'])
        else:
            check_df_1 = route2factor(cfg, df_barr_target, globals()[f'route_{idx}'])
            check_df_0 = pd.concat([check_df_0, check_df_1], axis=0)
    df_route2factor = check_df_0.reset_index(drop=True)
    return df_route2factor


# 다중기준의사결정(MCDM) 수행
def mcdm_run(data, save_name):
    data = data+0.001
    matrix = data[data.columns].to_numpy()
    
    weights = mcdm_weights.equal_weights(matrix)
    # print(f'weights: {weights}')
    # print(mcdm_weights.entropy_weights(matrix))
    # print(mcdm_weights.standard_deviation_weights(matrix))
    types = np.array([1, -1, -1, -1, -1, -1, -1])
    topsis = mcdm_methods.TOPSIS()
    # print(topsis(matrix, weights, types))
    # print(topsis(matrix, weights, types).shape)
    topsis_methods = {
        'minmax': mcdm_methods.TOPSIS(norm.minmax_normalization),
        # 'max': mcdm_methods.TOPSIS(norm.max_normalization),
        # 'sum': mcdm_methods.TOPSIS(norm.sum_normalization),
        # 'vector': mcdm_methods.TOPSIS(norm.vector_normalization),
        # 'log': mcdm_methods.TOPSIS(norm.logaritmic_normalization),
    }
    results = {}
    for name, function in topsis_methods.items():
        results[name] = function(matrix, weights, types)
    display_result = tabulate([[name, *rankdata(pref, reverse=True)] for name, pref in results.items()],
                headers=['Method'] + [f'route{i+1}' for i in range(10)])
    print(display_result)
    text_file=open(save_name,"w")
    text_file.write(display_result)
    text_file.close()

    # 'minmax' 정규화 방법으로 얻은 결과 반환
    return results['minmax']


parser = argparse.ArgumentParser()
parser.add_argument('--config_filename', default='ver_1.yaml')  # --config_filename: 설정 파일의 경로 (기본값으로 'ver_1.yaml'를 사용)
parser.add_argument('--route_path', default='Path_API_Response_Example.json')  # --route_path: 경로 데이터 파일의 경로 (기본값으로 'Path_API_Response_Example.json'를 사용)
parser.add_argument('--SAVE_PATH', default='SAVE_PATH')  # --SAVE_PATH: 결과를 저장할 디렉토리의 경로 (기본값으로 'SAVE_PATH'를 사용)

args = parser.parse_args()
config_filename = args.config_filename  # 설정 파일 경로
route_path = args.route_path  # 경로 데이터 파일 경로
SAVE_PATH = args.SAVE_PATH  # 결과 저장 디렉토리 경로

cfg = load_config(config_filename)
df_barr_target = data_load(cfg)
df_route2factor = route_load(cfg, route_path)
result = mcdm_run(df_route2factor, save_name=SAVE_PATH) # 다중 기준 의사결정 실행 및 결과 저장

print(1) # 결과가 완료되었음을 나타내는 메시지 출력
print(result) # MCDM 결과 출력

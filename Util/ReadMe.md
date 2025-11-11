데이터 preprocessing 과정
1. mittocsv.py : mit raw data를 z-score 정규화 후 csv로 변환
2. patchingdata.py: csv 파일로 변환된 데이터를 1024 크기의 블록으로 splitting (Overlapping 비율: 50%)
3. predict.py: 모델의 weight를 불러와서 결과 예측, 이 과정에서는 단순히 csv 데이터를 모델에 입력하는 방식

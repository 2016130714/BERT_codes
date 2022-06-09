# 버트 미세조정 - 슬롯 태깅  
  
1. pretrained BERT 모델을 모듈로 export  
    - ETRI에서 사전훈련한 BERT의 체크포인트를 가지고 BERT 모듈을 만드는 과정.  
    - `python export_korbert/bert_to_module.py -i {체크포인트 디렉토리} -o {output 디렉토리}`   
    - 예시: `python export_korbert/bert_to_module.py -i /content/drive/MyDrive/004_bert_eojeol_tensorflow -o /content/drive/MyDrive/bert-module`  
  
2. 데이터 준비  
    - 모델을 훈련하기 위해 필요한 seq.in, seq.out이라는 2가지 파일을 만드는 과정.  
    - `python prepare_data.py -i {input파일} -o {output 디렉토리}`   
    - 예시: `python prepare_data.py -i /content/drive/MyDrive/data/sample_data.txt -o /content/drive/MyDrive/data/sample/`  
  
3. Fine-tuing 훈련  
    - TODO - 위의 내용처럼 어떻게 하면 `train_bert_finetuning.py` 코드를 실행할 수 있는지 코드 내부의 parser을 참조하여 작성하세요.  
  
4. 모델 평가  
    - TODO - 위의 내용처럼 어떻게 하면 `eval_bert_finetuned.py` 코드를 실행할 수 있는지 코드 내부의 parser을 참조하여 작성하세요.  
    - 테스트의 결과는 --model에 넣어준 모델 경로 아래의 `test_results`에 저장된다.  
  
3. Inference (임의의 문장을 모델에 넣어보기)  
    - TODO - `eval_bert_finetuned.py`를 참고하여 한 문장씩 넣어서 모델이 내뱉는 결과물을 볼 수 있도록 inference.py 코드를 완성하세요.  
    - `python inference.py --model {훈련된 모델이 저장된 경로}`   
    - 예시: `python inference.py --model saved_model/`   
    - 모델 자체가 용량이 커서 불러오는 데까지 시간이 걸림  
    - "Enter your sentence:"라는 문구가 나오면 모델에 넣어보고 싶은 문장을 넣어 주면 됨  
    - quit라는 입력을 넣어 주면 종료  

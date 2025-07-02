# 한국어 일상 대화 연결 Baseline
본 리포지토리는 '2025년 국립국어원 인공지능의 한국어 능력 평가' 상시 과제 중 '한국어 일상 대화 연결'에 대한 베이스라인 모델의 추론을 재현하기 위한 코드를 포함하고 있습니다.  

추론의 실행 방법(How to Run)은 아래에서 확인하실 수 있습니다.

### Baseline
|           Model           | Accuracy(%) |
| :-----------------------: | :---------: |
|        **Qwen3-8B**        |    0.479    |
| **HyperCLOVAX Text 1.5B** |    0.418    |

## 리포지토리 구조 (Repository Structure)
```
# 추론에 필요한 리소스들을 보관하는 디렉토리
resource
└── data

# 실행 가능한 python 스크립트를 보관하는 디렉토리
run
└── test.py

# 추론에 사용될 함수들을 보관하는 디렉토리
src
└── data.py
```

## 데이터 형태 (Data Format)
```
[
   {
      "id": "nikluge-2025-일상_대화_연결-dev-1",
      "input": {
         "대상발화": "긍게 이제 이거는 그래도 서민들을 위한 거니까",
         "발화위치": "선행문",
         "발화1": "서민을 위하는 정치인이 누구인지 알고 싶어.",
         "발화2": "이걸 위해 그래도 어떤 노력을 해왔는지 궁금해.",
         "발화3": "평범한 사람들을 위해 이런 제도가 있었으면 해."
      },
      "output": "발화3"
   }
]
```

## 실행 방법 (How to Run)
### 추론 (Inference)
```
CUDA_VISIBLE_DEVICES=0 python -m run.test \
    --input resource/data/sample.json \
    --output result.json \
    --model_id Qwen/Qwen3-8B \
    --debug True
```

## Reference
국립국어원 인공지능 (AI)말평 (https://kli.korean.go.kr/benchmark)  
transformers (https://github.com/huggingface/transformers)  
Qwen3-8B (https://huggingface.co/Qwen/Qwen3-8B)  
HyperCLOVAX-SEED-Text-Instruct-1.5B (https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B)

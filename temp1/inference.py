import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import textwrap

print("🚀 모델 추론 및 테스트 스크립트 시작")

# 1. 훈련된 모델 및 토크나이저 로드
# train_model.py에서 최종 저장한 모델 폴더명을 사용합니다.
model_path = "./koconovel_t5_final_model"
try:
    trained_model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    trained_tokenizer = AutoTokenizer.from_pretrained(model_path)
    print(f"'{model_path}'에서 훈련된 모델을 성공적으로 로드했습니다.")
except OSError:
    print(f"오류: '{model_path}' 폴더를 찾을 수 없습니다.")
    print("먼저 train_model.py를 실행하여 모델 훈련을 완료해주세요.")
    exit()

# 2. 추론을 위한 파이프라인 생성
coref_resolver = pipeline(
    "text2text-generation",
    model=trained_model,
    tokenizer=trained_tokenizer,
    device=0 if torch.cuda.is_available() else -1 # GPU 사용
)
print("추론 파이프라인 준비 완료.")

# 3. 테스트할 길고 다양한 문단 목록
test_paragraphs = [
    {
        "title": "📚 문학 소설 스타일",
        "text": "정수와 민영은 오랫동안 잊고 있던 오래된 다리 위에서 다시 만났다. 그는 그녀를 보자마자 심장이 멎는 듯했다. 하지만 그녀는 아무 말 없이 그를 지나쳐 다리를 건넜다. 그의 눈에는 실망감이 가득했다."
    },
    {
        "title": "📰 뉴스 기사 스타일",
        "text": "삼성전자는 어제 새로운 AI 칩셋인 '퓨전'을 공개했다. 이 칩셋은 기존 제품보다 2배 빠른 연산 속도를 자랑한다. 이재용 회장은 행사에 직접 참석하여 칩셋의 비전을 설명했으며, 그는 이 기술이 미래 산업의 판도를 바꿀 것이라고 강조했다. 이 회사는 연말까지 '퓨전'의 대량 생산을 시작할 계획이다."
    },
    {
        "title": "📜 역사적 인물 스타일",
        "text": "세종대왕은 조선의 4대 임금으로, 한글 창제를 주도했다. 그는 백성을 위한 정치를 펼쳤으며, 그의 통치 아래 조선은 과학과 문화의 황금기를 맞이했다. 집현전 학자들은 임금의 뜻을 받들어 다양한 연구를 수행했고, 그들의 노력은 훈민정음 해례본이라는 결실을 낳았다."
    },
    {
        "title": "🧩 복잡한 관계 스타일",
        "text": "연구팀은 프로젝트의 핵심 기술을 개발한 김 박사에게 모든 공을 돌렸다. 그녀가 없었다면 프로젝트는 시작조차 못 했을 것이다. 김 박사는 자신의 동료인 이 연구원에게 감사를 표했고, 그가 밤낮으로 도와준 덕분에 기술적 난관을 극복할 수 있었다고 말했다."
    }
]

# 4. 각 문단에 대해 추론 실행 및 결과 출력
print("\n" + "="*50)
print("              추론 테스트 시작")
print("="*50 + "\n")

for i, item in enumerate(test_paragraphs):
    title = item["title"]
    original_text = item["text"]
    
    # 모델 입력 형식에 맞게 접두사 추가
    input_text_with_prefix = f"상호참조해결: {original_text}"
    
    # 추론 실행
    result = coref_resolver(
        input_text_with_prefix,
        max_length=512,  # 긴 문장을 처리하기 위해 충분한 길이 설정
        num_beams=5,     # 더 품질 좋은 문장을 생성하기 위한 빔 서치
        early_stopping=True
    )
    
    generated_text = result[0]['generated_text']
    
    # 결과 출력
    print(f"---------- [테스트 {i+1}: {title}] ----------")
    # textwrap을 사용하여 긴 텍스트를 자동으로 줄 바꿈
    print("INPUT (원본 문단):")
    print(textwrap.fill(original_text, width=80))
    print("\nOUTPUT (모델 예측):")
    print(textwrap.fill(generated_text, width=80))
    print("\n" + "="*50 + "\n")

print("✅ 모든 테스트 완료.")

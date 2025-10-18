import torch
from kiwipiepy import Kiwi
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. 모델 및 도구 로드 --- (기존과 동일)
model_name = "klue/roberta-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
kiwi = Kiwi()

# --- 2. 텍스트 분석 및 '언급' 탐지 (⭐️ 이 부분을 수정!) ---
text = "어제 이순신 장군 동상을 봤다. 그는 우리나라 최고의 영웅이다."

# 연속된 명사(NNG, NNP)를 하나의 명사구로 묶는 로직
mentions = []
current_phrase = []
for token in kiwi.tokenize(text):
    # 토큰이 일반명사(NNG) 또는 고유명사(NNP)인 경우
    if token.tag in ['NNG', 'NNP']:
        current_phrase.append(token.form)
    else:
        # 더 이상 명사가 아니면, 지금까지 쌓인 명사구를 mentions에 추가
        if current_phrase:
            mentions.append(" ".join(current_phrase))
            current_phrase = []
        # 대명사(NP)나 다른 중요한 품사도 개별적으로 추가할 수 있음
        if token.tag == 'NP':
            mentions.append(token.form)

# 마지막 토큰이 명사구인 경우를 대비해 루프 종료 후 한 번 더 확인
if current_phrase:
    mentions.append(" ".join(current_phrase))

# 결과 (예시): mentions = ['이순신 장군', '동상', '그', '우리나라', '최고', '영웅']
print(f"탐지된 후보: {mentions}")


# --- 3. 각 '언급'의 의미 벡터 추출 --- (기존과 거의 동일)
mention_embeddings = {}
for mention in mentions:
    # 문맥을 함께 넣어주는 것이 성능에 더 좋습니다. (개선점)
    # 여기서는 간단하게 단어 자체만 사용합니다.
    inputs = tokenizer(mention, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state[0, 0, :].numpy()
    mention_embeddings[mention] = embedding

print("\n벡터 추출 완료!")

# --- 4. 유사도 계산 및 '연결' --- (이제 에러 없이 작동!)
he_vec = mention_embeddings['그'].reshape(1, -1)
lee_vec = mention_embeddings['이순신 장군'].reshape(1, -1)
dong상_vec = mention_embeddings['동상'].reshape(1, -1)

sim_lee = cosine_similarity(he_vec, lee_vec)
sim_dong상 = cosine_similarity(he_vec, dong상_vec)

print(f"\n'그' vs '이순신 장군' 유사도: {sim_lee[0][0]:.4f}")
print(f"'그' vs '동상' 유사도: {sim_dong상[0][0]:.4f}")

if sim_lee > sim_dong상:
    print("\n✅ 결론: '그'는 '이순신 장군'을 가리킵니다.")
else:
    print("\n✅ 결론: '그'는 '동상'을 가리킵니다.")
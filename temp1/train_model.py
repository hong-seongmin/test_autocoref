import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)

print("STEP 2: 모델 훈련 시작")

# 1. 사전 가공된 데이터셋 로드
processed_dataset_path = "./koconovel_processed_dataset"
try:
    raw_datasets = load_from_disk(processed_dataset_path)
    print(f"'{processed_dataset_path}'에서 사전 가공된 데이터셋을 성공적으로 로드했습니다.")
except FileNotFoundError:
    print(f"오류: '{processed_dataset_path}' 폴더를 찾을 수 없습니다.")
    print("먼저 prepare_data.py 스크립트를 실행하여 데이터를 가공해주세요.")
    exit()

# 2. 모델 및 토크나이저 로드
model_name = "KETI-AIR/ke-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# 3. 데이터 전처리 (토큰화)
def preprocess_function(examples):
    model_inputs = tokenizer(examples["input_text"], max_length=512, truncation=True)
    labels = tokenizer(text_target=examples["target_text"], max_length=512, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = raw_datasets.map(preprocess_function, batched=True, remove_columns=["input_text", "target_text"])

# 4. 훈련 인자(Arguments) 설정
per_device_batch_size = 8
num_train_samples = len(tokenized_datasets["train"])
steps_per_epoch = (num_train_samples + per_device_batch_size - 1) // per_device_batch_size
save_and_eval_steps = int(steps_per_epoch * 0.5)
print(f"0.5 에폭은 약 {save_and_eval_steps} steps에 해당합니다. 이 간격으로 모델을 저장하고 평가합니다.")

training_args = Seq2SeqTrainingArguments(
    output_dir="./results_coref_model",
    # 훈련 옵션
    num_train_epochs=5,
    per_device_train_batch_size=per_device_batch_size,
    per_device_eval_batch_size=per_device_batch_size,
    learning_rate=3e-5,
    weight_decay=0.01,
    
    # [수정된 부분] 충돌을 일으키는 모든 인자를 제거하고 핵심만 남깁니다.
    save_steps=save_and_eval_steps,
    eval_steps=save_and_eval_steps,
    save_total_limit=10, # 5 에폭 * 2 = 10개의 체크포인트를 모두 저장
    
    # 기타
    predict_with_generate=True,
    logging_dir='./logs_coref',
    logging_steps=100,
    fp16=torch.cuda.is_available(),
)

# 5. 데이터 콜레이터 및 트레이너 정의
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# 6. 훈련 시작
print("모델 훈련을 시작합니다...")
trainer.train()

# 7. 최종 모델 저장 (훈련의 마지막 상태를 저장)
final_model_path = "./koconovel_t5_final_model"
trainer.save_model(final_model_path)
print(f"\n✅ STEP 2 완료: 모델 훈련 성공. 최종 모델이 '{final_model_path}'에 저장되었습니다.")
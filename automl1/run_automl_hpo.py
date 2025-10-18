import torch
import optuna
import json
from datasets import load_from_disk
from transformers import (
    T5Tokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)

print("STEP 2: AutoML 하이퍼파라미터 최적화 시작")

# 1. 사전 가공된 데이터셋 로드
processed_dataset_path = "./koconovel_processed_dataset"
try:
    raw_datasets = load_from_disk(processed_dataset_path)
    print(f"'{processed_dataset_path}'에서 데이터셋 로드 완료.")
except FileNotFoundError:
    print(f"오류: '{processed_dataset_path}' 폴더를 찾을 수 없습니다. prepare_data.py를 먼저 실행해주세요.")
    exit()

# 2. AutoML을 위한 Objective 함수 정의
def objective(trial, model_name, train_dataset, eval_dataset):
    """Optuna가 최적화할 목표 함수"""
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    
    batch_sizes_to_try = [32, 16, 8, 4, 2] 
    successful_batch_size = None

    for batch_size in batch_sizes_to_try:
        try:
            print(f"\n[Trial #{trial.number}] 배치 사이즈 {batch_size}(으)로 훈련 시도...")
            
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            
            def preprocess_function(examples):
                model_inputs = tokenizer(examples["input_text"], max_length=512, truncation=True)
                labels = tokenizer(text_target=examples["target_text"], max_length=512, truncation=True)
                model_inputs["labels"] = labels["input_ids"]
                return model_inputs
            
            tokenized_train = train_dataset.map(preprocess_function, batched=True)
            tokenized_eval = eval_dataset.map(preprocess_function, batched=True)

            training_args = Seq2SeqTrainingArguments(
                output_dir=f"./hpo_results/{model_name.replace('/', '_')}_{trial.number}",
                learning_rate=trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
                per_device_train_batch_size=batch_size,
                num_train_epochs=trial.suggest_int("num_train_epochs", 3, 6),
                weight_decay=trial.suggest_float("weight_decay", 0.01, 0.1, log=True),
                
                gradient_accumulation_steps=2,
                per_device_eval_batch_size=batch_size,
                
                # [수정] 충돌을 일으키는 strategy 인자들을 모두 제거합니다.
                # evaluation_strategy="epoch",  <- 삭제
                # save_strategy="epoch",        <- 삭제

                logging_steps=100,
                save_total_limit=1,
                fp16=torch.cuda.is_available(),
                predict_with_generate=True,
            )
            
            data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
            trainer = Seq2SeqTrainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_train,
                eval_dataset=tokenized_eval,
                tokenizer=tokenizer,
                data_collator=data_collator,
            )

            trainer.train()
            successful_batch_size = batch_size
            print(f"배치 사이즈 {batch_size} 훈련 성공!")
            break 

        except torch.cuda.OutOfMemoryError:
            print(f"배치 사이즈 {batch_size}에서 CUDA 메모리 부족(OOM) 발생. 더 작은 사이즈로 재시도합니다.")
            torch.cuda.empty_cache()
            continue
        except Exception as e:
            print(f"예상치 못한 오류 발생: {e}")
            raise e

    if successful_batch_size is None:
        raise RuntimeError(f"모든 배치 사이즈 {batch_sizes_to_try} 시도에 실패했습니다. GPU 메모리를 확인해주세요.")

    eval_results = trainer.evaluate()
    trial.set_user_attr("successful_batch_size", successful_batch_size)
    return eval_results["eval_loss"]

# 3. 메인 실행 루프
models_to_compare = [
    "KETI-AIR/ke-t5-large",
    "google/mt5-large"
]
best_params_per_model = {}

for model_name in models_to_compare:
    print("\n" + "="*50)
    print(f"모델 [{model_name}]에 대한 HPO 시작...")
    print("="*50)

    study = optuna.create_study(direction="minimize", study_name=f"hpo_{model_name.replace('/', '_')}")
    study.optimize(
        lambda trial: objective(trial, model_name, raw_datasets['train'], raw_datasets['validation']), 
        n_trials=120
    )

    print(f"모델 [{model_name}] HPO 완료!")
    print(f"최적의 검증 손실 (Best Validation Loss): {study.best_value}")
    print(f"최적 하이퍼파라미터 (Best Hyperparameters): {study.best_params}")
    best_params_per_model[model_name] = study.best_params

print("\n\n" + "="*50)
print("모든 모델에 대한 HPO 완료")
print("="*50)
for model, params in best_params_per_model.items():
    print(f"\n모델: {model}")
    print(f"최적 파라미터: {params}")

with open("best_hyperparameters.json", "w", encoding="utf-8") as f:
    json.dump(best_params_per_model, f, ensure_ascii=False, indent=4)
print("\n'best_hyperparameters.json' 파일에 최적 파라미터를 저장했습니다.")
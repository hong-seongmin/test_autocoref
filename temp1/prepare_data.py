import json
import glob
import os
from tqdm.auto import tqdm
from datasets import Dataset, DatasetDict

print("STEP 1: 데이터 준비 시작")

if not os.path.exists("KoCoNovel"):
    print("KoCoNovel 데이터셋을 다운로드합니다...")
    os.system("git clone https://github.com/storidient/KoCoNovel.git")
else:
    print("KoCoNovel 데이터셋이 이미 존재합니다.")

def convert_koconovel_to_t5_format(data):
    """
    T5 학습 형식으로 변환하는 함수.
    이 함수는 이제 [{'start_offset': ..., 'end_offset': ..., 'text': ...}] 형태의
    데이터를 받을 준비가 되어 있습니다.
    """
    text = data['text']
    clusters = data.get('coreference_clusters', [])
    
    if not clusters:
        return None

    insertions = []
    for i, cluster in enumerate(clusters):
        cluster_id = i + 1
        for mention in cluster:
            start = mention['start_offset']
            end = mention['end_offset']
            insertions.append((start, f"<coref id={cluster_id}> "))
            insertions.append((end, f" </coref>"))

    insertions.sort(key=lambda x: x[0], reverse=True)
    
    tagged_text_list = list(text)
    for pos, tag in insertions:
        tagged_text_list.insert(pos, tag)
    
    target_text = "".join(tagged_text_list)
    
    return {
        "input_text": f"상호참조해결: {text}",
        "target_text": target_text
    }

print("모든 텍스트 데이터를 로드합니다...")
texts_by_id = {}
text_files = glob.glob("./KoCoNovel/data/jsonl/**/text.jsonl", recursive=True)
for file_path in tqdm(text_files, desc="텍스트 파일 로딩 중"):
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            texts_by_id[record['doc_id']] = record['text']
print(f"총 {len(texts_by_id)}개의 텍스트 문서를 로드했습니다.")

print("\n상호참조 데이터를 로드하고 텍스트와 결합합니다...")
processed_data = []
coref_files = glob.glob("./KoCoNovel/data/jsonl/**/coref.jsonl", recursive=True)
for file_path in tqdm(coref_files, desc="상호참조 데이터 가공 중"):
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            coref_record = json.loads(line)
            doc_id = coref_record['doc_id']
            text = texts_by_id.get(doc_id)
            
            if text:
                # --- [핵심 수정 부분] ---
                # 1. 올바른 키('omniscent_separate')로 데이터를 가져옵니다.
                original_clusters = coref_record.get('omniscent_separate', [])
                reformatted_clusters = []

                # 2. 데이터 구조를 변환합니다.
                # [['577', '578', '그', 'C']] -> [{'start_offset': 577, ...}]
                for cluster in original_clusters:
                    new_cluster = []
                    for mention in cluster:
                        new_mention = {
                            'start_offset': int(mention[0]),
                            'end_offset': int(mention[1]),
                            'text': mention[2]
                        }
                        new_cluster.append(new_mention)
                    reformatted_clusters.append(new_cluster)

                combined_data = {
                    'text': text,
                    'coreference_clusters': reformatted_clusters
                }
                # --- [수정 끝] ---
                
                t5_formatted = convert_koconovel_to_t5_format(combined_data)
                if t5_formatted:
                    processed_data.append(t5_formatted)

if not processed_data:
    print("\n오류: 가공된 데이터가 없습니다. 스크립트를 종료합니다.")
    exit()

print(f"\n가공 완료! 총 {len(processed_data)}개의 유효한 학습 데이터를 생성했습니다.")

dataset = Dataset.from_list(processed_data)
train_test_split = dataset.train_test_split(test_size=0.1, seed=42)
final_datasets = DatasetDict({
    'train': train_test_split['train'],
    'validation': train_test_split['test']
})

save_path = "./koconovel_processed_dataset"
final_datasets.save_to_disk(save_path)

print("\n✅ STEP 1 완료: 데이터 준비 및 가공 성공")
print(f"가공된 데이터셋이 '{save_path}' 폴더에 저장되었습니다.")
print(final_datasets)
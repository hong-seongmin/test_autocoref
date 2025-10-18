import re
import math
import warnings
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

import torch
import torch.nn.functional as F
from kiwipiepy import Kiwi
from transformers import AutoTokenizer, AutoModelForMaskedLM

# --------------------------------------------------------------
# 불필요한 경고 억제
warnings.filterwarnings("ignore", category=UserWarning)

# --------------------------------------------------------------
# 전역 리소스 로드(한 번만)
MODEL_NAME = "klue/roberta-large"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
MLM = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
MLM.eval()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MLM.to(DEVICE)
KIWI = Kiwi()

# --------------------------------------------------------------
# 간단한 문장 분리
SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')
def split_sentences(text: str) -> List[Tuple[int, int]]:
    spans = []
    start = 0
    for m in SENT_SPLIT_RE.finditer(text):
        end = m.start()
        spans.append((start, end))
        start = m.end()
    spans.append((start, len(text)))
    return spans

# --------------------------------------------------------------
# Kiwi 토큰 with 오프셋
@dataclass
class KToken:
    form: str
    tag: str
    start: int
    end: int

def kiwi_tokenize_with_offsets(text: str) -> List[KToken]:
    raw = KIWI.tokenize(text)
    tokens: List[KToken] = []
    pos = 0
    for t in raw:
        idx = text.find(t.form, pos)
        if idx < 0:
            idx = text.find(t.form, 0)
            if idx < 0:
                idx = pos
        start = idx
        end = start + len(t.form)
        pos = end
        tokens.append(KToken(t.form, t.tag, start, end))
    return tokens

# --------------------------------------------------------------
# 스팬 구조
@dataclass
class Span:
    text: str
    start: int
    end: int
    sent_idx: int

@dataclass
class PronSpan(Span):
    base: str       # NP 본체(예: '그', '그녀', '그것')
    particle: str   # 뒤따르는 조사(예: '은','는','이','가','을','를' 등 + 꼬리)

NOUN_TAGS = {"NNG", "NNP"}    # 일반/고유 명사
PRON_TAG = "NP"               # 대명사
PARTICLE_PREFIX = "J"         # 조사 품사 접두

FIRST_SECOND_PRON = {
    "나", "저", "내", "제", "우린", "우리", "우리가", "우리를",
    "너", "네", "당신", "너희", "당신들"
}

def build_sentence_index(text: str, sent_spans: List[Tuple[int, int]], pos: int) -> int:
    for i, (s, e) in enumerate(sent_spans):
        if s <= pos < e:
            return i
    return len(sent_spans) - 1

def extract_mentions_and_pronouns(text: str) -> Tuple[List[Span], List[PronSpan], List[KToken]]:
    tokens = kiwi_tokenize_with_offsets(text)
    sent_spans = split_sentences(text)

    # ----- 멘션(연속 명사구; 중첩 포함) -----
    mentions: List[Span] = []
    i, n = 0, len(tokens)
    while i < n:
        if tokens[i].tag in NOUN_TAGS:
            j = i
            while j + 1 < n and tokens[j + 1].tag in NOUN_TAGS:
                j += 1
            for k in range(i, j + 1):
                start = tokens[i].start
                end = tokens[k].end
                mtxt = text[start:end]
                sent_idx = build_sentence_index(text, sent_spans, start)
                mentions.append(Span(mtxt, start, end, sent_idx))
            i = j + 1
        else:
            i += 1

    # ----- 대명사(NP) + 뒤따르는 조사(J*) -----
    pronouns: List[PronSpan] = []
    i = 0
    while i < n:
        if tokens[i].tag == PRON_TAG:
            base_form = tokens[i].form
            if base_form in FIRST_SECOND_PRON:
                i += 1
                continue
            j = i
            while j + 1 < n and tokens[j + 1].tag.startswith(PARTICLE_PREFIX):
                j += 1
            start = tokens[i].start
            end = tokens[j].end
            sent_idx = build_sentence_index(text, sent_spans, start)
            particle = ""
            if j > i:
                particle = text[tokens[i + 1].start: tokens[j].end]
            pron_text = text[start:end]
            pronouns.append(PronSpan(pron_text, start, end, sent_idx, base_form, particle))
            i = j + 1
        else:
            i += 1

    return mentions, pronouns, tokens

# --------------------------------------------------------------
# 한글 종성 기반 조사 교정 규칙(은/는, 이/가, 을/를) — (허용된 규칙)
HANGUL_BASE = 0xAC00
HANGUL_END = 0xD7A3
def last_korean_char(s: str) -> Optional[str]:
    for ch in reversed(s.strip()):
        if HANGUL_BASE <= ord(ch) <= HANGUL_END:
            return ch
    return None
def has_jongseong(ch: Optional[str]) -> bool:
    if ch is None:
        return False
    code = ord(ch)
    if not (HANGUL_BASE <= code <= HANGUL_END):
        return False
    return ((code - HANGUL_BASE) % 28) != 0
def fix_particle(cand_text: str, particle: str) -> str:
    if not particle:
        return ""
    first = particle[:1]
    tail = particle[1:]
    jong = has_jongseong(last_korean_char(cand_text))
    if first in ("은", "는"):
        return ("은" if jong else "는") + tail
    if first in ("이", "가"):
        return ("이" if jong else "가") + tail
    if first in ("을", "를"):
        return ("을" if jong else "를") + tail
    return particle  # 나머지 조사들은 그대로

# --------------------------------------------------------------
# Masked LM 치환 점수
def masked_logprob_score(text: str,
                         replace_start: int,
                         replace_end: int,
                         replacement: str) -> float:
    with torch.no_grad():
        base = TOKENIZER(text, return_offsets_mapping=True, return_tensors="pt")
        offsets = base["offset_mapping"][0].tolist()
        input_ids = base["input_ids"][0].tolist()
        attention = base["attention_mask"][0].tolist()

        mask_indices = [i for i, (s, e) in enumerate(offsets)
                        if not (s == 0 and e == 0) and (s < replace_end and e > replace_start)]
        if not mask_indices:
            return -1e9

        cand_enc = TOKENIZER(replacement, add_special_tokens=False)
        cand_ids = cand_enc["input_ids"]
        if len(cand_ids) == 0:
            return -1e9

        mask_id = TOKENIZER.mask_token_id
        left, right = min(mask_indices), max(mask_indices)

        new_ids = input_ids[:left] + [mask_id] * len(cand_ids) + input_ids[right + 1:]
        new_att = attention[:left] + [1] * len(cand_ids) + attention[right + 1:]

        new_ids_t = torch.tensor([new_ids], device=DEVICE)
        new_att_t = torch.tensor([new_att], device=DEVICE)

        logits = MLM(input_ids=new_ids_t, attention_mask=new_att_t).logits
        mask_pos = list(range(left, left + len(cand_ids)))

        log_probs = F.log_softmax(logits[0, mask_pos, :], dim=-1)
        cand_ids_t = torch.tensor(cand_ids, device=DEVICE).unsqueeze(-1)
        token_logps = torch.gather(log_probs, 1, cand_ids_t).squeeze(-1)
        return float(token_logps.sum().item())

# --------------------------------------------------------------
# 거리(제약) 점수: 가까울수록 페널티 작음
def distance_penalty(text: str, pron: Span, cand: Span) -> float:
    sent_gap = max(0, pron.sent_idx - cand.sent_idx)
    enc = TOKENIZER(text, return_offsets_mapping=True, return_tensors="pt")
    offs = enc["offset_mapping"][0].tolist()

    def char_to_tok(char_ix: int) -> int:
        best_i: Optional[int] = None
        for i, (s, e) in enumerate(offs):
            if s == 0 and e == 0:
                continue
            if s <= char_ix < e:
                return i
            if e <= char_ix:
                best_i = i
        return best_i if best_i is not None else 1

    p_t = char_to_tok(pron.start)
    c_t = char_to_tok(cand.end - 1)
    tok_gap = max(1, p_t - c_t)

    w_sent = 0.6
    w_tok = 0.12
    return w_sent * sent_gap + w_tok * math.log1p(tok_gap)

# --------------------------------------------------------------
# (신규) 후보 애니메이시(사람/사물) 추론 — LM 기반
ANIMATE_CACHE: Dict[str, float] = {}   # candidate text -> score_anim
def animate_score(cand_text: str) -> float:
    """
    score_anim = logP('사람') - max(logP('사물'), logP('물건'))
    템플릿: "{cand}{은/는} XXXX이다."
    """
    if cand_text in ANIMATE_CACHE:
        return ANIMATE_CACHE[cand_text]
    particle = fix_particle(cand_text, "은")  # 주제 표지로 통일
    template = f"{cand_text}{particle} XXXX이다."
    start = template.index("XXXX")
    end = start + 4
    lp_person = masked_logprob_score(template, start, end, "사람")
    lp_obj1 = masked_logprob_score(template, start, end, "사물")
    lp_obj2 = masked_logprob_score(template, start, end, "물건")
    score = lp_person - max(lp_obj1, lp_obj2)
    ANIMATE_CACHE[cand_text] = score
    return score

# --------------------------------------------------------------
# 상호참조 해결(개선+타입 적합성)
TYPE_PENALTY = 1.2  # 타입 불일치 가산 페널티(너무 크지 않게)

def resolve_coreference_v2(text: str) -> List[Dict[str, Any]]:
    """
    - NP(대명사)+조사 스팬을 [MASK]*N 으로 치환
    - (기존) 후보 치환 로그확률 - 거리 페널티
    - (신규) 대명사 유형(그/그녀/그것) vs 후보 애니메이시(사람/사물) 적합성 페널티
      ↳ 애니메이시는 LM(사람/사물 템플릿)으로 추론
    - 조사(은/는/이/가/을/를)는 후보 종성에 맞춰 교정 후 치환
    """
    print(f"\n\n=== 분석 시작: \"{text}\" ===")
    mentions, pronouns, tokens = extract_mentions_and_pronouns(text)

    print("[멘션 후보]")
    print([m.text for m in mentions])
    print("[대명사 후보(+조사)]")
    print([p.text for p in pronouns])

    results: List[Dict[str, Any]] = []
    if not pronouns:
        return results

    for p in pronouns:
        cands = [m for m in mentions if m.end <= p.start]
        if not cands:
            continue

        best = None
        best_score = -1e18

        for c in cands:
            particle = fix_particle(c.text, p.particle)
            replacement = c.text + particle
            lm_score = masked_logprob_score(text, p.start, p.end, replacement)
            pen = distance_penalty(text, p, c)

            # --- 타입 적합성 페널티(루ール 최소: LM 기반 애니메이시) ---
            a = animate_score(c.text)  # >0 사람, <0 사물 경향
            type_pen = 0.0
            if p.base == "그것":
                if a > 0:  # 사람인데 '그것'이면 감점
                    type_pen += TYPE_PENALTY
            elif p.base in {"그", "그녀"}:
                if a < 0:  # 사물인데 '그/그녀'면 감점
                    type_pen += TYPE_PENALTY

            score = lm_score - pen - type_pen

            if score > best_score:
                best_score = score
                best = (c, lm_score, pen, type_pen, replacement, a)

        if best is not None:
            c, lm, pen, tpen, rep, anim = best
            results.append({
                "pronoun": p.text,
                "pronoun_base": p.base,
                "pronoun_span": (p.start, p.end),
                "antecedent": c.text,
                "antecedent_span": (c.start, c.end),
                "score_total": float(best_score),
                "score_lm": float(lm),
                "penalty_dist": float(pen),
                "penalty_type": float(tpen),
                "candidate_animate_score": float(anim),
                "replacement_scored": rep
            })

    return results

# --------------------------------------------------------------
# 간단 평가용 테스트
if __name__ == "__main__":
    test_sentences = [
        # 0) 베이직 리캡 — 단문/양호 케이스
        "어제 이순신 장군 동상을 봤다. 그는 우리나라 최고의 영웅이다.",
        "세종대왕은 훈민정음을 창제했다. 그는 백성을 아끼는 위대한 군주였다.",
        "신사임당은 뛰어난 예술가였다. 그녀는 율곡 이이의 어머니이기도 하다.",
        "나는 어제 새로운 노트북을 샀다. 그것은 정말 빠르고 가벼웠다.",
        "김철수는 영희에게 선물을 주었다. 그는 그녀가 기뻐하는 모습을 보고 행복했다.",

        # 1) 긴 문장 + 삽입구 + 다중 후보(사람만)
        "서울대학교 전기정보공학부 소속 김철수 교수(반도체 소자 전공)는 오늘 강연을 했다. 그는 청중의 질문에 차분히 답했다.",
        # 2) 긴 문장 + 삽입구 + 다중 후보(사람/사물 혼재) → '그것'은 사물로 가야 함
        "현대자동차의 신형 전기차 아이오닉7과 배터리 모듈, 그리고 김정민 엔지니어가 함께 소개됐다. 그것은 주행거리가 크게 향상되었다.",
        # 3) 동격(Apposition) + 재지시
        "시인 윤동주, 그를 기리는 전시가 열렸다. 그는 하늘과 바람과 별을 사랑했다.",
        # 4) 동격(Apposition) 복합
        "삼성전자 부사장 박민수, 메모리 사업부 전략총괄은 계획을 발표했다. 그는 투자 일정을 구체화했다.",
        # 5) 동일 표면형 다른 개체(회사명/사람명 충돌) → 사람/사물 타입 구분
        "영희는 회의에 참석했고, 영희전자는 부스를 설치했다. 그녀는 발표를 맡았다. 그것은 신제품 시연이었다.",
        # 6) 다중 남성 후보 + 근접성/거리 페널티 체크
        "민수는 철수를 초대했다. 그 뒤에 박현우가 도착했다. 그는 인사를 건넸다.",
        # 7) 다중 여성 후보 + 근접성/거리
        "지영은 수진을 만났다. 잠시 후 혜린이 도착했다. 그녀는 먼저 커피를 주문했다.",
        # 8) 소유격1 — '그의' (NP+의)
        "김철수는 오늘 면접을 보았다. 그의 답변은 차분했고, 면접관들은 고개를 끄덕였다.",
        # 9) 소유격2 — '그녀의' (NP+의)
        "신사임당은 그림을 남겼다. 그녀의 작품은 섬세한 필치로 유명하다.",
        # 10) 목적격 — '그를/그녀를' (NP+을/를)
        "박지성은 경기를 지배했다. 관중들은 그를 연호했다.",
        "영희가 논문을 발표했다. 심사위원들은 그녀를 높이 평가했다.",
        # 11) 여럿 등장 + 관계절(내포절) + 소유/주격 혼재
        "김철수는 [영희가 그의 발표를 듣고 감탄했다]고 말했다. 그는 겸손하게 웃었다.",
        # 12) 관형절로 길어지는 명사구 + 주격 '그가'
        "고려대학교에서 인공지능 석사를 마치고 카이스트 로봇공학 박사과정에 진학한 박세준 연구원은 논문을 게재했다. 그가 제시한 방법은 계산량을 크게 줄였다.",
        # 13) 대화/인용 — 인물 참조 유지
        "철수가 말했다. \"영희가 오면 좋겠다.\" 그는 창밖을 바라보았다.",
        # 14) 대화/인용 — 인용 속 대명사가 바깥 인물 참조
        "영희가 속삭였다. \"그는 약속을 잊지 않았을 거야.\" 주변은 조용해졌다.",
        # 15) 타입 충돌 검증(사람/사물) — '그것'은 사물, '그'는 사람
        "문서는 책상 위에 놓였다. 김과장은 서명을 했다. 그것은 최종본이었다. 그는 결재를 올렸다.",
        # 16) 복수 대명사 '그들'
        "팀원들(민수, 지영, 수진)은 밤새 작업했다. 그들은 마침내 빌드를 통과시켰다.",
        # 17) 복수 → 단수 전환(담화 추적)
        "참가자들은 등록을 마쳤다. 그들 중 김현수는 가장 먼저 발표했다. 그는 차분했다.",
        # 18) 비교적 긴 거리(두 문장 사이 후보 다수)
        "민수는 오전에 브리핑을 했다. 회의가 길어지자 지연과 태수가 추가 보고를 했다. 그가 내놓은 결론은 간결했다.",
        # 19) '그것'과 유사 후보가 여러 개 — 가장 최근의 사물
        "보고서, 초안, 그리고 수정본이 차례로 업로드되었다. 그것은 표와 그림이 대폭 보강되었다.",
        # 20) 조사 체인(에게/는) — NP 뒤 다중 조사
        "박민수는 신입에게 조언을 아끼지 않았다. 그에게는 늘 시간이 부족했다.",
        # 21) 직함·호칭 NNG 포함(사람 선호 강화 상황)
        "총장 이정문은 개회사를 했다. 그는 신입생들에게 환영 인사를 전했다.",
        # 22) 복수 후보 + 동격 + 사물 혼재
        "한국은행 총재 이창용, 그리고 금통위는 기준금리를 동결했다. 그는 기자회견에서 전망을 설명했다.",
        # 23) 동일 문자열 반복(중복 표면형, 서로 다른 위치)
        "민수는 민수의 계획을 수정했다. 그는 일정부터 다시 잡았다.",
        # 24) 내포절 속 소유격 — '그의'가 바깥 선행사로 가야 함
        "철수는 [영희가 발표한 그의 연구 결과]를 검토했다. 그는 추가 실험을 제안했다.",
        # 25) '그녀' vs 사물 후보 혼재(타입 페널티로 방지)
        "서류와 계약서가 책상에 놓였다. 수진은 서명을 마쳤다. 그녀는 동료에게 복사본을 전달했다.",
        # 26) 사물→사람 혼재 — '그'는 가장 최근 사람으로
        "새 프로그램이 배포되었고, 민수가 검증을 진행했다. 그는 주요 버그를 찾아냈다.",
        # 27) 긴 수식 명사구(고유명사 연쇄)
        "주식회사 네오바이오테크 임상개발본부 김지후 이사는 계획안을 제출했다. 그는 일정의 현실성을 강조했다.",
        # 28) 모호성 극대화(남성 후보 3개) + 거리
        "진수는 종석을 소개했고, 곧 민호가 합류했다. 그는 먼저 악수를 청했다.",
        # 29) '그것' + 두 사물 후보(가까운 것 선호)
        "장비와 시제품이 동시에 도착했다. 그것은 아직 검수 중이다.",
        # 30) 사물 후보가 먼저, 사람 후보가 나중 — '그'는 사람에게 가야 함
        "초안이 공개되었다. 이후 김성훈이 수정을 제안했다. 그는 서론을 대폭 줄였다.",
        # 31) 길고 복잡한 문장 + 재귀적 내포 + 주격/목적격 교차
        "데이터셋을 정제하던 민준은 [수빈이 그가 어제 만든 스크립트를 고쳤다고] 보고했다. 그는 로그 수집부만 보완하자고 했다.",
        # 32) 관형절 + 소유격 연속
        "지수는 [민호가 그녀의 코드를 리뷰하면서 남긴] 코멘트를 검토했다. 그녀는 일부만 반영했다.",
        # 33) 대화 + 서로 다른 화자 교차(바깥 담화로 해석)
        "경호가 말했다. \"오늘 민지가 발표할 거야.\" 그는 슬라이드를 정리했다.",
        # 34) 사물-사람-사물 혼합 연쇄
        "패치와 릴리스 노트가 게시되었고, 태진이 알림을 보냈다. 그것은 긴급 수정 사항이었다.",
        # 35) 복수 대명사 + 접속조사
        "철수와 민호는 토론을 이어갔다. 그들과 함께 영희도 참여했다.",
        # 36) 긴 담화 후반부 대명사 — 거리 페널티 작동 확인
        "아침 회의에서 태윤은 개요를, 지효는 일정표를 공유했다. 점심 이후에야 민호가 핵심 지표를 설명했다. 그는 이후 질의응답을 진행했다.",
        # 37) 격조사 변형(으로/에서 등은 그대로 유지; 타입만 확인)
        "로봇은 실험실로 들어갔다. 그것으로 테스트가 마무리됐다.",
        # 38) 사물/사람 후보 교대 — '그녀'는 사람
        "서버와 로그가 준비됐고, 지연이 배포를 시작했다. 그녀는 롤백 시나리오도 작성했다.",
        # 39) 명시적 선행사 후행(역지시; 실패가 정상) — 우리 로직은 앞선 후보만 허용
        "그는 철수가 도착한 뒤에야 회의실로 들어갔다.",
        # 40) 동일 담화 내 다중 대명사 연쇄(그/그의/그에게)
        "현우는 초안을 올렸다. 그의 동료들은 검토 의견을 달았다. 그에게는 수정할 시간이 필요했다.",
        # 41) 사물의 소유격 — '그것의'
        "신형 카메라가 출시됐다. 그것의 자동초점 성능은 대폭 개선됐다.",
        # 42) 사람 vs 기관명(사물) 충돌
        "카카오와 민수가 공동 연구를 발표했다. 그는 데이터 파이프라인을 설명했다.",
        # 43) 다대다 후보 + 여성 대명사
        "민지와 해린이 토론했고, 보고서는 잠시 보류됐다. 그녀는 결론을 더 명확히 하자고 했다.",
        # 44) 명사구 중복 + 소유격 내포
        "영준은 영준의 팀이 그의 제안을 지지했다고 밝혔다. 그는 최종 결정을 미뤘다.",
        # 45) 인명+직함 복합, 뒤따르는 사물 대명사
        "한국과학기술원 AI대학원 김도윤 조교수는 새 커리큘럼을 공개했다. 그것은 프로젝트 중심으로 설계되었다.",
        # 46) 동격+복수 → 복수 대명사
        "디자이너 수진과 개발자 하람, 두 사람은 밤새 협업했다. 그들은 마침내 프로토타입을 완성했다.",
        # 47) 관형절 두 겹 + 주어 전환
        "지훈은 [민서가 [그가 정리한 기준]을 검토한 뒤] 메일을 보냈다. 그는 수정본만 공유했다.",
        # 48) 사물 후보가 많은 리스트 + 최근성
        "기획안, 제안서, 예산서, 일정표가 한꺼번에 올라왔다. 그것은 가장 최신의 버전이었다.",
        # 49) 긴 이름 + 조사 교정 확인
        "주식회사 브라이트비전 인더스트리얼솔루션즈 김태완 대표이사는 전략을 발표했다. 그는 해외 진출 계획을 밝혔다.",
        # 50) 인용부호 속 2연속 대명사
        "민아가 말했다. \"그는 준비됐고, 그녀는 자료를 모두 모았어.\" 모두가 고개를 끄덕였다.",
    ]

    for sent in test_sentences:
        out = resolve_coreference_v2(sent)
        print("\n--- 최종 해결 결과 ---")
        if not out:
            print("해결된 대명사-선행사 쌍을 찾지 못했습니다.")
        else:
            for r in out:
                s = r["score_total"]
                print(f"[{r['pronoun_span'][0]}:{r['pronoun_span'][1]}] "
                      f"'{r['pronoun']}' → '{r['antecedent']}' "
                      f"(score={s:.2f}, lm={r['score_lm']:.2f}, "
                      f"dist_pen={r['penalty_dist']:.2f}, type_pen={r['penalty_type']:.2f}, "
                      f"anim={r['candidate_animate_score']:.2f}, rep='{r['replacement_scored']}')")
        print("="*60)

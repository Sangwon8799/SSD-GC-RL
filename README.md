# SSD-GC-RL — SSD 가비지 컬렉션: 언제 지우고 무엇을 지울까

강화학습개론(2026년 2학기) 프로젝트. SSD 의 가비지 컬렉션(GC)에서 **어느 블록을 지울지(희생 블록)** 와 **언제 지울지(GC 시점)** 를
강화학습으로 배운다. 실제 SSD 대신 직접 지은 페이지 매핑 FTL 시뮬레이터(Python)를 환경으로 쓰고, 그 시뮬레이터가 맞게 동작한다는 것을
손계산 · 이론식 · 발표된 수치로 먼저 보였다.

- 팀: 20232195 지상원 · [팀원 학번 · 이름]
- 저장소: <GITHUB_URL>

## 무엇을 하나

| 문제 | 상태 | 행동 | 보상 | 알고리즘 |
|---|---|---|---|---|
| 무엇을 지울까 — **최종 설계** | 장치 통계 7(닫힌 블록 유효 비율의 평균 · 흩어짐 · 거의 꽉 찬 블록의 몫 · GC 블록의 몫 · 직전 64번의 희생 u · WAF · 젊은 블록에 몰린 무효화의 몫) + 직전에 고른 β | 점수 (1 − u) × 나이^β ÷ u 의 β ∈ {0, 0.5, 1, 2} 하나를 골라 **다음 GC 64번** 동안 쓴다(β 0 = greedy · β 1 = cost-benefit) | −(64번 희생 블록 u 의 평균) | Double DQN · REINFORCE + 가치 기준선 |
| 무엇을 지울까 — 첫 설계(배우지 못함 · 진단용으로 남김) | 후보 블록 8개 × 특징 8 + 장치 특징 4 | GC 마다 후보 8개 가운데 하나(greedy 순위 4 + cost-benefit 순위 4) | −(그 GC 의 희생 블록 u) | DQN · Double DQN · Dueling DQN · REINFORCE + 가치 기준선 |
| 언제 지울까 | 빈 블록 수 · 유휴 경과 · 도착 간격 · 다음 희생 블록의 유효 비율 · 최근 포그라운드 GC | 기다린다 / 지금 한 블록 GC | −(요청 지연 합 + λ × 옮긴 블록 분량) | 표 Q-learning · DQN(시간 간격만큼 할인) |

비교한 기준: 규칙 다섯(greedy · cost-benefit · FIFO · 무작위 · d-choices(8)) · 고정 β 넷 · 64번마다 무작위 β · 고정 번갈음(β 0 ↔ 1 · 0 ↔ 2) ·
「후보 중 무작위」(첫 설계) · 시점 규칙 (W, τ, W_e) 64개.

## 설치

```bash
python3 -m venv venv && . venv/bin/activate          # Python 3.12
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

GPU 는 필요 없다(신경망 가중치 수천 개 · 규칙 고르기 한 판이 CPU 한 코어로 1–2분).

## 실행

```bash
python -m tests.test_ftl                 # 손계산 검사 다섯(몇 초)
python -m tests.test_prep                # 트레이스 변환기 손계산 검사
python scripts/validate_sim.py           # 시뮬레이터 검증 E0(이론식 · 발표된 수 · 불변식, 약 3분)
python scripts/validate_timing.py        # 시간 모형 검증(M/D/1)
python scripts/run_baselines.py          # 규칙 정책 E1
python scripts/run_cand_baselines.py     # 후보 중 무작위 · 학습 전 신경망

python scripts/run_rule_baselines.py     # 고정 β · 무작위 β · 번갈음

# 한 판 학습 — runs/<exp-id>/ 에 config · 학습 기록 · 검증 기록 · best.pt · model.pt · done.json
python scripts/train_rule.py --exp-id demo --algo dqn --workload hc90-10                     # 최종 설계(약 1–2분)
python scripts/train.py --exp-id demo-d --algo reinforce --workload hc80-20 --continuing 1 --gamma 0.999 --bootstrap 1 --eval-stochastic 1   # 첫 설계
python scripts/train_timing.py --exp-id demo-t --algo tabq --lam 3000 --gamma 0.9995 --steps 2000000                                          # 시점
python scripts/trace_rule.py runs/demo   # 학습한 에이전트가 64번마다 고른 β 와 희생 블록
```

## 전체 실험 다시 내기

```bash
python scripts/experiments.py                                  # queues/rule.json · queues/direct.json · queues/timing.json
python scripts/run_queue.py --queue queues/rule.json --parallel 2     # E2 · E3 · E4 · E7 용(규칙 고르기)
python scripts/run_queue.py --queue queues/direct.json --parallel 1   # E2a 첫 설계(할인율 · 다섯 워크로드)
python scripts/run_queue.py --queue queues/timing.json --parallel 1   # E6 시점
python scripts/eval_transfer_rule.py                           # E5 일반화
python scripts/eval_combined.py                                # E7 합치기
python scripts/switch_horizon.py hc80-20                       # 규칙을 바꾼 이득이 오는 시간
python scripts/pergc_rules.py hc90-10                          # 고정 규칙의 희생 블록 통계
python scripts/plot_results.py                                 # results/figures/*.png · results/summary/*.csv
```

실행기는 같은 판을 두 번 돌리지 않고(끝난 판은 건너뛴다), 끊긴 판은 체크포인트에서 이어 간다. 시드: 학습 0–4 · 검증 9000+s · 9500+s(체크포인트 고르기) ·
시험 100–104(성적 — 학습에도 고르기에도 쓰지 않는다). 신뢰구간은 시드 다섯 평균의 t 분포 95% 구간이다.

## 결과

모든 수는 `results/` 와 `runs/*/done.json` 에서 나온다(시험 시드 100–104 · 에이전트는 학습 시드 다섯의 평균 ± 95% 신뢰구간).

### E2 — 규칙의 성향 β 를 64번마다 고르는 에이전트(OP 14%)

| 워크로드 | β 0 = greedy | β 1 = cost-benefit | 가장 좋은 고정 규칙 | 같은 비율 무작위 | 고정 번갈음(나은 쪽) | DQN | REINFORCE | DQN(다섯을 섞어 배움) |
|---|---|---|---|---|---|---|---|---|
| uniform | 4.309 | 4.353 | β 0 = greedy 4.309 | 4.302 | 4.325 | 4.302 ± 0.003 | 4.303 ± 0.003 | 4.339 ± 0.046 |
| hc80-20 | 4.445 | 3.934 | β 1 = cost-benefit 3.934 | 3.857 | 3.775 | 3.786 ± 0.029 | 3.855 ± 0.014 | 3.824 ± 0.045 |
| hc90-10 | 4.547 | 2.862 | β 1 = cost-benefit 2.862 | 2.835 | 2.629 | 2.629 ± 0.009 | 2.733 ± 0.028 | 2.635 ± 0.022 |
| zipf1.0 | 4.328 | 3.357 | β 0.5 3.351 | 3.279 | 3.343 | 3.215 ± 0.015 | 3.300 ± 0.029 | 3.388 ± 0.246 |
| shift90-10 | 4.514 | 3.026 | β 1 = cost-benefit 3.026 | 2.944 | 2.829 | 2.784 ± 0.025 | 2.912 ± 0.022 | 2.815 ± 0.014 |

- 쏠린 워크로드 넷에서 DQN 이 가장 좋은 고정 규칙보다 WAF 를 3.8–8.1% 낮췄다. uniform 에서는 greedy 를 골라 greedy 와 같다.
- 에이전트는 「값싼 수확(β 0)」과 「차가운 블록 정리(β 1 · 2)」를 64번씩 번갈아 쓴다(`scripts/trace_rule.py`). 같은 β 비율을 순서 없이 섞으면(`results/baselines/mix.csv`) 그 이득의 47–88% 가 사라진다 — 핵심은 차례다. 그 차례를 옮겨 적은 고정 번갈음은 에이전트와 비슷하다.
- 첫 설계(GC 마다 후보 8개 중 하나를 고르기 · `scripts/train.py`)는 배우지 못했다 — 학습한 에이전트가 「후보 중 무작위」보다도 나빴다.

### E3 · E4 — 하이퍼파라미터와 절제(다섯을 섞어 배운 DQN · 학습 시드 셋 · 다섯 워크로드 평균 WAF)

| 설정 | 평균 WAF |
|---|---|
| 기본(γ 0.9 · M 64 · 상태 11 · β 넷 · 시드 다섯) | 3.400 |
| 할인율 0 | 3.643 |
| 할인율 0.5 | 3.543 |
| 할인율 0.99 | 3.385 |
| 묶음 4번(M = 4) | 3.551 |
| 묶음 8번(M = 8) | 3.466 |
| 묶음 16번(M = 16) | 3.325 |
| M = 16 · 지평 맞춤(γ 0.974) | 3.343 |
| 묶음 256번(M = 256) | 3.497 |
| 상태 없음(모두 0) | 3.567 |
| 직전 β 만 | 3.385 |
| − 유효 비율 분포 셋 | 3.375 |
| − 무효화 쏠림 | 3.389 |
| 규칙 둘(β 0 · 1)만 | 3.368 |

- 할인율 0 · 0.5 는 「지금 비싼 정리」를 고르지 못해 나쁘다. 묶음 길이는 M = 16 이 64 보다 낫고 256 은 나쁘다.
- 상태를 모두 지우면 한 규칙에 굳어 무너지지만, 직전 β 하나만 남겨도 전체 상태와 거의 같다 — 상태 특징의 몫은 확인되지 않았다.

### E5 — 다시 학습하지 않은 일반화

| 평가 | β 0 | β 1(CB) | 무작위 바꾸기 | 0 ↔ 1 번갈음 | 0 ↔ 2 번갈음 | DQN(hc90-10 · OP 14% 에서 배움) | DQN(다섯을 섞어 배움) |
|---|---|---|---|---|---|---|---|
| zipf0.8 — 안 본 워크로드 | 4.447 ± 0.016 | 4.121 ± 0.019 | 4.132 ± 0.031 | 4.120 ± 0.022 | 4.147 ± 0.023 | **4.100 ± 0.049** | 4.103 ± 0.020 |
| hc95-5 — 안 본 워크로드 | 4.383 ± 0.053 | 2.022 ± 0.047 | 1.874 ± 0.064 | **1.808 ± 0.023** | 1.934 ± 0.012 | 1.816 ± 0.016 | 2.016 ± 0.307 |
| hc90-10 · OP 7% | 7.735 ± 0.043 | 5.219 ± 0.022 | 4.863 ± 0.089 | 4.602 ± 0.051 | **4.531 ± 0.038** | 4.830 ± 0.343 | 4.837 ± 0.491 |
| hc90-10 · OP 28% | 2.809 ± 0.025 | 1.833 ± 0.011 | 1.796 ± 0.079 | 1.760 ± 0.008 | **1.750 ± 0.011** | 1.778 ± 0.039 | 1.759 ± 0.022 |

### E6 — GC 시점(OP 28% · hc80-20 · 평균 지연 / 99 백분위 / WAF)

| 정책 | 평균 지연(ms) | 99 백분위(ms) | 99.9 백분위(ms) | WAF | 시드 |
|---|---|---|---|---|---|
| 포그라운드만 | 21.94 | 77.8 | 100.2 | 2.711 | 평가 5 |
| 규칙 W 8 · τ 2 · W_e 4 | 2.09 | 14.8 | 36.3 | 2.767 | 평가 5 |
| 규칙 W 12 · τ 2 · W_e 5 | 1.34 | 8.0 | 23.7 | 2.874 | 평가 5 |
| 규칙 W 6 · τ 2 · W_e 3 | 3.90 | 46.4 | 72.3 | 2.727 | 평가 5 |
| DQN · λ 300 | 4.08 ± 3.20 | 42.1 ± 18.1 | 65.0 ± 22.1 | 3.072 ± 0.880 | 학습 5 |
| DQN · λ 3,000 | 16.03 ± 3.74 | 71.3 ± 7.6 | 97.0 ± 9.7 | 2.711 ± 0.005 | 학습 5 |
| 표 Q · λ 300 | 3.92 ± 3.72 | 38.5 ± 17.1 | 63.0 ± 18.2 | 2.753 ± 0.039 | 학습 5 |
| 표 Q · λ 1,000 | 4.80 ± 1.76 | 46.5 ± 10.4 | 70.8 ± 9.8 | 2.742 ± 0.039 | 학습 5 |
| 표 Q · λ 3,000 | 6.16 ± 2.06 | 50.9 ± 8.5 | 76.3 ± 9.9 | 2.724 ± 0.024 | 학습 5 |
| 표 Q · λ 10,000 | 5.30 ± 1.25 | 47.8 ± 6.0 | 71.2 ± 8.6 | 2.754 ± 0.103 | 학습 5 |

- 표 Q-learning · DQN 모두 유휴 시간에 미리 지우는 법을 배웠지만 잘 고른 규칙(W · τ · W_e 64개)의 경계선을 넘지 못했다. 시점 DQN 은 보상 배율 0.01 에서 「지우지 않기」로 무너져 0.00001 로 고쳤다.

### E7 — 희생 블록 정책 × 시점 정책

| 무엇을 지울까 | 언제 지울까 | 평균 지연(ms) | 99 백분위(ms) | WAF |
|---|---|---|---|---|
| greedy | 포그라운드만 | 21.94 ± 1.81 | 77.8 ± 7.6 | 2.711 ± 0.008 |
| greedy | 규칙 W 8 · τ 2 · W_e 4 | 2.09 ± 0.24 | 14.8 ± 8.7 | 2.767 ± 0.008 |
| greedy | 표 Q(λ 3,000) | 6.22 ± 2.24 | 48.1 ± 8.5 | 2.724 ± 0.016 |
| cost-benefit | 포그라운드만 | 11.22 ± 0.20 | 41.8 ± 1.4 | 2.325 ± 0.012 |
| cost-benefit | 규칙 W 8 · τ 2 · W_e 4 | 1.40 ± 0.13 | 8.3 ± 0.2 | 2.430 ± 0.013 |
| cost-benefit | 표 Q(λ 3,000) | 11.22 ± 0.20 | 41.8 ± 1.4 | 2.325 ± 0.012 |
| 규칙 고르는 DQN | 포그라운드만 | 16.14 ± 2.76 | 116.1 ± 44.1 | 2.288 ± 0.008 |
| 규칙 고르는 DQN | 규칙 W 8 · τ 2 · W_e 4 | 2.26 ± 1.04 | 36.0 ± 30.5 | 2.376 ± 0.009 |
| 규칙 고르는 DQN | 표 Q(λ 3,000) | 16.15 ± 3.05 | 126.0 ± 55.8 | 2.286 ± 0.019 |

- greedy 와 함께 배운 표 Q 시점 에이전트는 희생 블록 정책이 바뀌면 유휴 GC 를 멈춘다 — 「다음 희생 블록 u」가 학습 때 한 번도 가지 않은 칸(u < 0.5)으로 내려간다(`scripts/timing_state_shift.py` · `results/combined/state_shift.csv`).

### 학습 시간

학습 판 160개 + 시범 27개 · CPU 코어 하나 기준 합 약 10.7시간 · 규칙 고르기 한 판 1–3분 · 시점 한 판 4–12분 · GPU 없음.

그래프는 `python scripts/plot_results.py` 가 `results/figures/` 에 낸다.

## 학습된 모델

`models/` 에 판마다 검증 시드로 고른 체크포인트(`best.pt`)와 그 설정(`config.json`)이 있다. 불러오는 법:

```python
import json, torch, sys
sys.path.insert(0, "scripts")
from train_rule import build                      # 첫 설계의 판은 train.build_agent
from ssdgc.rule_env import RuleAgentPolicy
from ssdgc.env import run_policy
cfg = json.load(open("models/<판>/config.json"))
agent = build(cfg)
agent.load_state_dict(torch.load("models/<판>/best.pt", weights_only=False))
pol = RuleAgentPolicy(agent, cfg["M"], stochastic=bool(cfg["eval_stochastic"]), acts=cfg["acts"], drop=cfg["drop"])
print(run_policy(pol, "hc90-10", seed=100)["waf"])
```

## 데이터 출처

합성 워크로드 다섯(uniform · hc80-20 · hc90-10 · zipf1.0 · shift90-10)을 코드가 난수 시드로 만든다 — 외부 데이터를 쓰지 않았다.
공개 블록 트레이스(MSR Cambridge, SNIA IOTTA 저장소)를 넣는 길(`scripts/prep_msr.py` → `ssdgc.workloads.Replay`)은 남겨 두었지만,
내려받기에 개인 정보 등록이 필요해 이번 실험에는 쓰지 않았고 트레이스 파일도 이 저장소에 넣지 않았다.
시뮬레이터 검증에 쓴 발표된 수치는 Xiang · Kurkoski, 「An Improved Analytical Expression for Write Amplification in NAND Flash」(arXiv:1110.4245)의 greedy 시뮬레이션 값이다.

## 생성형 AI 활용 범위

- 명세서 해석 · 문제 설계 · 실험 계획 · 시뮬레이터 · 환경 · 에이전트 · 실험 실행 · 그래프 · 보고서(PPT) 초안은 AI(Claude)가 했다.
- 강화학습 부분(재현 버퍼 · DQN 손실 · REINFORCE · 표 Q-learning)은 PyTorch 텐서 · NumPy 배열 연산 위에 직접 짰고 Stable-Baselines 같은
  강화학습 라이브러리나 공개 저장소의 코드를 쓰지 않았다(명세서 8쪽 「코드는 직접 구현하여 표절이 되지 않도록」).
- 팀은 결과를 검토하고 코드를 직접 읽어 검증했다(명세서 8쪽 「AI 로 한 코딩은 … 반드시 코드를 직접 보고 검증」). 검증할 자리:

| 무엇 | 어디(파일:줄) | 확인할 것 |
|---|---|---|
| 쓰기 한 번 · 무효화 | `ssdgc/ftl.py:128` | 옛 물리 페이지를 무효로 돌리고 새 자리에 쓴 뒤 빈 블록이 2개보다 적으면 GC 를 부르는가 |
| GC 한 번 | `ssdgc/ftl.py:146` | 닫힌 블록만 고르는가 · 유효 페이지만 옮기는가 · 지운 뒤 빈 목록에 넣는가(5절 코드 설명) |
| 일관성 검사 | `ssdgc/ftl.py:190` | 매핑 ↔ 역매핑 ↔ 유효 수 ↔ 빈 목록이 서로 맞는지 무엇을 재는가 |
| WAF 계산 | `ssdgc/env.py:237` | 1 + (측정 구간에 옮긴 페이지) ÷ (측정 구간의 호스트 쓰기) — 예열 구간을 빼는가 |
| greedy · cost-benefit 식 | `ssdgc/policies.py:18` | (1 − u) × 나이 ÷ 2u 의 나이가 「닫힌 뒤 지난 시간」인가 |
| 최종 설계 — β 규칙 | `ssdgc/rule_env.py:30` | β 0 이 greedy(같으면 오래된 것) · 그 밖은 (1 − u) × 나이^β ÷ u 의 최댓값인가(6절 코드 설명) |
| 최종 설계 — 64번 묶음 · 보상 | `ssdgc/rule_env.py:72` | 고른 β 를 GC 64번 동안 쓰고 옮긴 페이지 · 호스트 쓰기를 세는가 |
| 최종 설계 — 상태 11 | `ssdgc/rule_env.py:41` | 닫힌 블록 통계 · 직전 64번의 u · WAF · 무효화 쏠림 · 직전 β 표시가 6절과 같은가 |
| 최종 설계 — 환경 한 걸음 | `ssdgc/rule_env.py:133` | 보상 = −(64번 평균 u) · 에피소드 50걸음에서 「자름」(장치는 이어 씀) |
| 최종 설계 — Double DQN | `ssdgc/agents/rule_agents.py:61` | 6절 식 ↔ 코드 줄(고르기는 메인 · 값은 타깃 · Huber · 200번마다 복사) |
| 최종 설계 — REINFORCE | `ssdgc/agents/rule_agents.py:122` | 잘린 꼬리 부트스트랩 · reward-to-go · 기준선 detach · 엔트로피 |
| 최종 설계 — 학습 · 평가 | `scripts/train_rule.py:93` | 검증 시드로 best.pt 를 고르고 시험 시드 100–104 로 재는가 · REINFORCE 의 평가 방식 |
| 첫 설계 — 후보 집합 | `ssdgc/env.py:36` | greedy 순위 넷 + cost-benefit 순위 넷 · 겹치면 다음 순위로 채우는가 |
| 첫 설계 — 상태 특징 12 | `ssdgc/env.py:69` | 후보 특징 여덟과 장치 특징 넷이 3절 표와 같은가 |
| 첫 설계 — 보상과 결정 시점 | `ssdgc/env.py:198` | 보상 = −(옮긴 페이지 ÷ 64) · 다음 GC 가 필요할 때까지 쓰기를 이어 가는가 |
| 첫 설계 — 보상 가운데 맞추기 | `scripts/train.py:103` | greedy 300번의 평균 u 를 상수로 빼는가(최적 정책 불변) |
| 첫 설계 — 점수 신경망 | `ssdgc/agents/dqn.py:26` | 후보마다 같은 MLP(12 → 64 → 64 → 1)인가 |
| 첫 설계 — Dueling 신경망 | `ssdgc/agents/dqn.py:44` | Q = V + A − 평균 A · V 가 장치 특징과 후보의 평균 · 최댓값을 보는가 |
| 첫 설계 — DQN 손실 | `ssdgc/agents/dqn.py:169` | 6절 식 ↔ 코드 줄(고르기는 메인 · 값은 타깃 · Huber · 500번마다 복사) |
| 첫 설계 — REINFORCE | `ssdgc/agents/reinforce.py:71` | reward-to-go · 기준선 detach · 어드밴티지 정규화 · 엔트로피 항 |
| 첫 설계 — 체크포인트 고르기 | `scripts/train.py:132` | 검증 시드(9000+ · 9500+)와 시험 시드(100–104)가 겹치지 않는가 |
| 시점 환경 · 보상 | `ssdgc/timing_env.py:92` | 기다림 · 지금 GC 두 갈래 · 보상 = −(지연 합 + λ × 복사 ÷ 64) × 0.01 · dt_ms |
| 시간 간격 할인 | `ssdgc/agents/timing_agents.py:54` | 타깃 = r + γ^Δt × max Q(s′) |
| 규칙 정책(시점) | `ssdgc/timing_env.py:135` | 같은 관측으로 (W, τ, W_e) 규칙을 쓰는가 — 에이전트와 같은 결정 시점 |
| 이론식 | `ssdgc/theory.py:19` | u = exp(−α(1 − u)) 의 0 이 아닌 근을 이분법으로 찾는가 · 순수 무작위 α/(α − 1) |
| 손계산 검사 | `tests/test_ftl.py:59` | 5절 단계 그림의 수(WAF = 13/12)가 단언으로 박혀 있는가 |

## 디렉터리

```
ssdgc/            시뮬레이터(ftl.py) · 규칙 정책 · 워크로드 · 이론식 · 환경(rule_env.py 최종 · env.py 첫 설계 · timing_env.py) · 시간 모형
ssdgc/agents/     규칙 고르기(rule_agents.py) · 첫 설계 DQN 계열(dqn.py) · REINFORCE(reinforce.py) · 시점 에이전트(timing_agents.py)
scripts/          검증 · 기준 · 학습 · 실험 대기열 · 평가 · 그래프
tests/            손계산 검사
results/          검증 · 기준 · 요약 CSV · 그래프
models/           학습된 모델(best.pt · config.json)
```

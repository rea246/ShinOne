# 그래프 학습 기반 대표패턴 선정 및 커버리지 분석 기술보고서

> 작성 상태: 기술요약 초안 작성. 각 항목을 순서대로 검토·보완한다.
>
> 구현 근거: `agent/topology-clustering-experimental` 브랜치의
> `EXPERIMENTAL_TOPOLOGY_CLUSTERING.md`, `EXPERIMENTAL_KPCA_ANALYSIS.md`,
> `EXPERIMENTAL_COVERAGE_COMPARISON.md` 및 관련 코드.
> 초안 작성 시 확인한 구현 기준 커밋: `ee091cd9c62ddc721f8c6162cd34ab9abe5386d1`.

## 1. 기술요약

본 기술은 반도체 레이아웃의 중심 패턴과 주변 형상 간 구조적 관계를 그래프로 표현하고, 그래프 신경망으로 학습한 특징을 이용하여 대규모 패턴 집합에서 구조적 다양성을 대표하는 실제 패턴을 자동 선정하는 기술이다. 중심 형상과 단계별 주변 구조의 특징을 함께 활용하여 유사한 패턴을 군집화하고, 각 군집의 중심에 가까운 실제 패턴을 대표로 추출한다. 이때 표본 기반 군집화와 전체 패턴의 후속 배정을 결합하여 대규모 데이터의 분류에 필요한 계산 부담을 줄인다.

또한, 선정된 대표군의 적정성을 평가하기 위해 공통 기준 표본으로 정의한 kPCA 공간에서 기존 선정 방법과 커버리지를 비교한다. 거리 반경에 따른 커버리지 변화와 구간별 미커버 비율을 분석하고, 기존 방법이 놓치지만 제안 방법이 포함하는 영역의 실제 패턴을 추출한다. 이를 통해 대표패턴 선정 결과를 정량적으로 비교하고, 모델 구축 및 검증에 필요한 패턴의 보완 후보를 제시하는 것을 목적으로 한다.

## 2. 기존기술

작성 예정. 공개 문헌의 대표패턴 선정 방법과 본 실험의 실제 비교 방법을 구분하여 정리한다.

## 3. 제안기술

작성 예정. 그래프 표현·특징 학습, 표본 기반 군집화·전체 배정, 실제 대표 추출 및 공통 기준의 커버리지 비교 과정을 정리한다.

## 4. 평가과정

작성 예정. 실제 적용 대상, 입력 데이터, 비교군, 평가 조건 및 지표를 확인하여 작성한다.

## 5. 평가결과

작성 예정. 실제 실행 및 측정으로 확인한 결과를 기재하고, 미확인 결과는 보강 후 반영한다.

## 6. 보호대상 핵심 기술 권리

작성 예정. 구현의 구체적인 차별 요소와 결합 구성을 정리하고 관련 선행기술과 비교한다.

## 7. 개선효과(기술적/경제적)

### 7.1 기술적 개선효과

작성 예정. 확인된 개선효과와 추가 평가가 필요한 기대효과를 구분하여 작성한다.

### 7.2 경제적 개선효과

작성 예정. 계측·연산·모델 구축에 필요한 시간과 비용의 근거를 확인하여 작성한다.

## 참고자료 및 배경 근거

아래 자료는 기술의 필요성과 기존 접근을 설명하기 위한 공개 자료이며, 본 실험의 성능을 입증하는 결과로 사용하지 않는다.

1. **The feasibility of using image parameters for test pattern selection during OPC model calibration.** SPIE Advanced Lithography, 2010.
   [IBM Research 공개 초록](https://research.ibm.com/publications/the-feasibility-of-using-image-parameters-for-test-pattern-selection-during-opc-model-calibration)
   - 배경 근거: OPC 모델 정확도 확보와 계측·모델 구축 시간 사이의 절충 및 이미지 파라미터를 이용한 시험 패턴 선정.
2. **Rehab Kotb Ali and Le Hong, “Smart down-sampling using SONR.”** Proc. SPIE 12052, *DTCO and Computational Patterning*, 120520N, 2022. DOI: [10.1117/12.2614609](https://doi.org/10.1117/12.2614609).
   [Siemens 공개 기술자료](https://resources.sw.siemens.com/nl-NL/technical-paper-smart-down-sampling-using-sonr/)
   - 배경 근거: 대규모 설계에서의 대표패턴 추출 필요성과 패턴 커버리지·모델 커버리지 평가의 활용.

## 작성 확인 사항

- 직접적인 적용 목적이 **MPC 모델 구축용 게이지 선정**인지 확인한 뒤 기술요약의 적용 대상과 목적을 구체화한다. 현재 초안에서는 모델 구축 및 검증을 위한 대표패턴 선정으로 기술하였다.

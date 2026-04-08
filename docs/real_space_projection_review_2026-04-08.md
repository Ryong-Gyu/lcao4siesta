# Real-space projection 검토 (electron density)

검토 대상:
- `ref/atmfuncs.f`
- `ref/spher_harm.f`
- `ref/write_orb_indx.f90`
- `ref/rhoofd.F90`
- Python 구현: `lcao/core/model.py`, `lcao/compute/density.py`

## 1) 핵심 식 비교

SIESTA 참조 구현(`all_phi`)은 각 오비탈을

\[
\phi_i(\mathbf r) = \text{phir}_i(r) \cdot \underbrace{\left(r^{l_i} Y_{l_i m_i}(\hat r)\right)}_{\text{rlylm}}
\]

형태로 계산한다. (`phi(i) = phir * rly(jlm)`)  
즉 `r^l`은 각도함수 쪽(`rlylm`)에 포함된다.

현재 Python 구현은

\[
\phi_i(\mathbf r) = \underbrace{\left(\text{phir}_i(r)\, r^{l_i}\right)}_{Rnl} \cdot \underbrace{Y_{l_i m_i}(\hat r)}_{Yml}
\]

로 계산한다. (`Rnl` 내부에서 `* r**l`, `Yml`은 순수 각도함수)

=> 두 구현은 곱의 분해 위치만 다를 뿐 최종 \(\phi_i\)는 동등하다.

## 2) density 누적 식 비교

SIESTA `rhoofd`는 삼각행렬 루프를 사용해
- off-diagonal(\(i \neq j\)): 기여를 한 번만 더하되, 대칭성 때문에 실질적으로 2배 효과
- diagonal(\(i=j\)): 1/2 계수와 `sqrt(2)` 스케일링 조합으로 최종 1배

를 만족하도록 구성되어 있다.

현재 Python은
- `io1 <= io2`만 순회
- `factor = 2.0 (off-diag), 1.0 (diag)`
- `rho += DM * factor * (phi1*phi2).real`

로 구현되어 있어 수학적으로 SIESTA 누적과 동치이다.

## 3) m(자기양자수) / 실수 조화함수 컨벤션

`write_orb_indx.f90` 설명과 `spher_harm.f`의 저차 explicit 식(예: p_x, p_y 부호)을 기준으로 보면,
현재 `Yml`의
- m>0: `sqrt(2) * Re(Y_l^{|m|})`
- m<0: `sqrt(2) * Im(Y_l^{|m|})`

선택은 SIESTA의 실수 조화함수 부호 컨벤션과 일치한다(최소 l<=2에서 부호 일치 확인).

## 4) 결론

- 질문하신 `r^l` 위치 변경(원본: ylm 쪽, 현재: Rnl 쪽)은 **최종 식 관점에서 문제 없음**.
- density real-space projection 식(활성 오비탈 선택 + sparse DM 누적)에서 **구조적 오류는 발견되지 않음**.
- 다만 향후 회귀 방지를 위해 `l<=2` 분석식 기반의 자동 검증 테스트(특히 m<0 부호)를 추가하는 것을 권장.

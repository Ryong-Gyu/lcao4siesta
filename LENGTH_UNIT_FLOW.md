# Length Unit Flow (Internal Unit = Bohr)

## 기준 경로

아래 표는 `siesta_io.readStruct`, `siesta_io.readIon`, `lcao/core/model.py:Rnl`, `unit_cell_grid`, `_build_supercell_vectors`를 따라 길이 단위가 어떻게 흐르는지 정리한 것입니다.

| 경로 | 입력 단위(원본 파일) | 내부 변환 | 내부 저장 단위 | 사용 지점 |
|---|---|---|---|---|
| `readStruct()` → `CELL` | `LatticeConstant`와 `LatticeVectors` 조합 (`Ang` 또는 `Bohr`) | `LatticeVectors * LatticeConstant` 후 `Ang`이면 `Ang → Bohr` 변환 | **Bohr** | `_build_supercell_vectors()`, `unit_cell_grid()` |
| `readStruct()` → `ATOMS` (`ScaledCartesian`) | 좌표는 `LatticeConstant` 단위 | `ATOMS * LatticeConstant` 후 `LatticeConstant` 단위 기준으로 Bohr 변환 | **Bohr** | PLDOS 위치 벡터 계산, `Rnl()` 반경 계산 |
| `readStruct()` → `ATOMS` (`Ang`) | Ang | `Ang → Bohr` | **Bohr** | 동일 |
| `readStruct()` → `ATOMS` (`Bohr`) | Bohr | 변환 없음 | **Bohr** | 동일 |
| `readStruct()` → `ATOMS` (`Fractional`/`ScaledByLatticeVectors`) | 격자 분수 좌표 | `ATOMS @ CELL` (CELL이 이미 Bohr) | **Bohr** | 동일 |
| `readIon()` → PAO `r`, `cutoff` | SIESTA `.ion` 기본 Bohr | 변환 없음 (`length_unit='bohr'` 메타데이터 부여) | **Bohr** | `Rnl()` 보간/컷오프 판정 |
| `_build_supercell_vectors()` | `cell`, `cutoff` | 별도 변환 없음 | **Bohr** | 중첩 슈퍼셀 벡터 생성 |
| `unit_cell_grid()` | `cell` | 별도 변환 없음 | **Bohr** | 실공간 격자 포인트 생성 |
| `Rnl()` | 호출 인자 `r`, PAO 테이블 `r/cutoff` | 단위 불일치 시 예외 | **Bohr 일치 강제** | 반경 보간 안전성 확보 |

## 정책 요약

- 내부 길이 단위는 **Bohr**로 고정.
- `readStruct()`에서 `LatticeConstant` 단위 + `AtomicCoordinatesFormat` 조합별 변환을 명시적으로 수행.
- `readIon()`은 `r`, `cutoff`를 Bohr 그대로 유지하고 `length_unit='bohr'` 메타데이터를 보존.
- `Rnl()`에서 projector 내부 단위와 PAO 테이블 단위가 다르거나 메타데이터가 없으면 즉시 예외를 발생.

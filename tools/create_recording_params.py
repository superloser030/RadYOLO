"""
AWR1642 RecordingParameters.mat 생성 스크립트.
한 번만 실행하면 됨. data/radar/iqData_RecordingParameters.mat 에 저장.
"""
import scipy.io
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = PROJECT_ROOT / "data" / "radar" / "iqData_RecordingParameters.mat"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# AWR1642 awr1642_raw_data.cfg 기반 파라미터
params = {
    "ADCSampleRate":  np.float64(5209),    # kHz
    "SweepSlope":     np.float64(70),      # MHz/us
    "SamplesPerChirp":np.float64(256),
    "CenterFrequency":np.float64(77),      # GHz
    "ChirpCycleTime": np.float64(486.14),  # us (idleTime 429 + rampEndTime 57.14)
    "NumReceivers":   np.float64(4),
    "NumChirps":      np.float64(32),      # 2 chirps/loop × 16 loops
}

scipy.io.savemat(str(OUT_PATH), {"RecordingParameters": params})
print(f"저장 완료: {OUT_PATH}")
for k, v in params.items():
    print(f"  {k}: {v}")

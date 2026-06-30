import scipy.io
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = PROJECT_ROOT / "data" / "radar" / "iqData_RecordingParameters.mat"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

params = {
    "ADCSampleRate":  np.float64(5209),
    "SweepSlope":     np.float64(70),
    "SamplesPerChirp":np.float64(256),
    "CenterFrequency":np.float64(77),
    "ChirpCycleTime": np.float64(486.14),
    "NumReceivers":   np.float64(4),
    "NumChirps":      np.float64(32),
}

scipy.io.savemat(str(OUT_PATH), {"RecordingParameters": params})
print(f"저장 완료: {OUT_PATH}")
for k, v in params.items():
    print(f"  {k}: {v}")

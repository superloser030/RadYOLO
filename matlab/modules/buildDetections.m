function dets = buildDetections(time, azList, rngList, rrList, p)
% ── [Convert 예제] spherical 자식 프레임 (레이더 원점, azimuth-only) ──
mp = struct();
mp.Frame         = 'spherical';
mp.OriginPosition = [0; 0; 0];      % 레이더를 원점에 둔다 (X축 정렬)
mp.Orientation   = eye(3);
mp.HasAzimuth    = true;
mp.HasElevation  = false;           % ← 우리 수정: AWR1642 수평 RX → 고도각 없음
mp.HasRange      = true;
mp.HasVelocity   = true;

% ── [ParkingLot 예제 line 54-60] 측정 노이즈 (대각 공분산) ──
mn = diag([p.azStd^2, p.rangeStd^2, p.rrStd^2]);   % [36, 0.36, rrStd^2]

n = numel(azList);
dets = cell(1, n);
for i = 1:n
    meas = [azList(i); rngList(i); rrList(i)];
    dets{i} = objectDetection(time, meas, ...
        'MeasurementNoise',      mn, ...
        'MeasurementParameters', mp);
end
end

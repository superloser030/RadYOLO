function xrv = readRadarCube(fr, frameIdx)
% ── [RecordProcess 예제 원문] ────────────────────────────────────────────
% The function read() returns a cell array of IQ data cubes, indexed as
% (fast-time/samples, rx-channel, slow-time/chirps).
% CurrentPosition 은 읽기 전용 → 현재 위치(폴링마다 reader 재생성으로 1)부터
% frameIdx 개를 읽어 마지막(최신) 큐브 사용
cubes  = read(fr, frameIdx);
iqData = cubes{end};                % cell → 최신 큐브 [samples x 4(rx) x chirps]

% ── [MIMO 가상배열 예제 원문 line 124-130] ───────────────────────────────
% For the TDM-MIMO radar system used in this example, the measurements
% corresponding to the two transmit antenna elements can be recovered from
% two consecutive sweeps by taking every other page of the data cube.
xr1 = iqData(:, :, 1:2:end);        % TX1 (홀수 sweep)
xr2 = iqData(:, :, 2:2:end);        % TX2 (짝수 sweep)

% TX 별 sweep 수가 다르면(홀수 chirp) 짧은 쪽에 맞춤 — cat 차원 일치 보장
m = min(size(xr1, 3), size(xr2, 3));
xr1 = xr1(:, :, 1:m);
xr2 = xr2(:, :, 1:m);

% Hence, the data cube from the virtual array can be formed as:
xrv = cat(2, xr1, xr2);             % [samples x 8(가상 rx) x chirps/2]
end

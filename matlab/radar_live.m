clear; clc;
thisDir = fileparts(mfilename('fullpath'));
addpath(fullfile(thisDir, 'modules'));

recordLocation = "D:\projects\RadYOLO\data\radar";
outputFile = fullfile(recordLocation, "targets.json");
tmpFile    = fullfile(recordLocation, "targets.json.tmp");
tsFile     = fullfile(recordLocation, "iqData_timestamps.csv");

POLL_SEC = 0.2;

% ── 레이더 파라미터 (.mat 1회 로드) ──────────────────────────────────────
temp = load(fullfile(recordLocation, 'iqData_RecordingParameters.mat'));
rp = temp.RecordingParameters;
c          = 3e8;
fs         = rp.ADCSampleRate * 1e3;
sweepSlope = rp.SweepSlope * 1e12;
nr         = rp.SamplesPerChirp;
fc         = rp.CenterFrequency * 1e9;
lambda     = c / fc;
chirpCycle = rp.ChirpCycleTime * 1e-6;
Nt = 2;
Nr = rp.NumReceivers;
Nvr = Nt * Nr;

% ── 시스템 객체 (1회 생성) — 전역 클러스터/추적 없음 ─────────────────────
% 검출점(az, range, doppler)을 그대로 내보내고, 클러스터링/물체 분리는
% Python(radar_fusion)이 YOLO bbox + DA3 depth 게이트로 bbox 단위로 처리한다.
varray = phased.ULA('NumElements', Nvr, 'ElementSpacing', lambda/2);

prf = 1 / (Nt * chirpCycle);
rdresp = phased.RangeDopplerResponse('PropagationSpeed', c, ...
    'DopplerOutput', 'Speed', 'OperatingFrequency', fc, 'SampleRate', fs, ...
    'RangeMethod', 'FFT', 'PRFSource', 'Property', 'PRF', prf, ...
    'SweepSlope', sweepSlope, ...
    'RangeFFTLengthSource', 'Property', 'RangeFFTLength', nr, ...
    'DopplerFFTLengthSource', 'Auto', ...
    'RangeWindow', 'Hann', 'DopplerWindow', 'Hann');

cfar = phased.CFARDetector2D('Method', 'CA', ...
    'TrainingBandSize', [8 4], 'GuardBandSize', [4 2], ...
    'ProbabilityFalseAlarm', 1e-4);

ang = -60:0.5:60;
doa = phased.BeamscanEstimator('SensorArray', varray, 'PropagationSpeed', c, ...
    'OperatingFrequency', fc, 'DOAOutputPort', true, ...
    'NumSignals', 1, 'ScanAngles', ang);

% ── CFAR/게이트 파라미터 (클러스터 파라미터는 Python 쪽으로 이동) ──
p = struct();
p.cfarMarginR = 8 + 4;
p.cfarMarginD = 4 + 2;
p.rGateMin  = 0.3;  p.rGateMax  = 6.0;
p.topN      = 50;                          % bbox 매칭용이라 다소 넉넉히

fprintf('실시간 레이더 검출 시작 (검출점 출력, Ctrl+C 로 종료)\n');

while true
    t_poll = tic;

    % 폴링마다 reader 재생성 → 새로 추가된 .bin/프레임 반영
    try
        fr = dca1000FileReader(recordLocation = recordLocation);
    catch
        pause(POLL_SEC); continue;
    end
    N = fr.NumDataCubes;
    if N < 2
        clear fr; pause(POLL_SEC); continue;
    end
    lastIdx = N - 1;

    % ── 타임스탬프 (frame_idx 는 0-based) ──
    frame_ts = 0;
    try
        tsTable = readtable(tsFile);
        if any(strcmp('frame_idx', tsTable.Properties.VariableNames))
            mask = tsTable.frame_idx == (lastIdx - 1);
            if any(mask)
                v = tsTable.ts_ms(mask);
                frame_ts = double(v(1));
            end
        end
    catch
    end

    % ── 파이프라인 (검출점까지) ──────────────────────────────────────────
    try
        xrv = readRadarCube(fr, lastIdx);
        [resp, rngGrid, dopGrid] = rangeDopplerMap(xrv, rdresp);
        [ri, di, ~] = cfarDetect(resp, cfar, p, rngGrid);
    catch ME
        warning('파이프라인 오류: %s', ME.message);
        clear fr; pause(POLL_SEC); continue;
    end

    % ── 검출점 → targets.json (클러스터 없이 그대로) ──────────────────────
    % ⚠ 검증필요: az 부호. radar_fusion.NEGATE_AZIMUTH 와 함께 좌우 맞출 것.
    targets = struct('range_m', {}, 'azimuth_deg', {}, 'velocity_mps', {});
    if ~isempty(ri)
        az      = estimateAzimuth(resp, ri, di, doa);
        rngVals = rngGrid(ri);
        dopVals = dopGrid(di);
        for k = 1:numel(ri)
            targets(end+1).range_m    = rngVals(k);     %#ok<AGROW>
            targets(end).azimuth_deg  = az(k);
            targets(end).velocity_mps = dopVals(k);
        end
    end

    frameData.frame_idx = lastIdx - 1;
    frameData.ts_ms     = frame_ts;
    if isempty(targets)
        frameData.targets = {};
    else
        frameData.targets = targets;
    end
    payload = {frameData};

    % ── atomic write ──
    fid = fopen(tmpFile, 'w');
    fprintf(fid, '%s', jsonencode(payload));
    fclose(fid);
    for attempt = 1:20
        try
            movefile(tmpFile, outputFile);
            break;
        catch
            pause(0.03);
        end
    end

    fprintf('[RADAR] frame %d | %d det | %.2fs\n', ...
        lastIdx - 1, numel(targets), toc(t_poll));

    clear fr
    pause(POLL_SEC);
end

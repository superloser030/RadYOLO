    % AWR1642 레이더 데이터 CFAR 검출 스크립트
% 출력: data/radar/targets.json (프레임별 range_m, velocity_mps, azimuth_deg, ts_ms)

recordLocation = "D:\projects\RadYOLO\data\radar";
outputFile     = "D:\projects\RadYOLO\data\radar\targets.json";

% 파라미터 로드
temp = load(fullfile(recordLocation, 'iqData_RecordingParameters.mat'));
p = temp.RecordingParameters;

fs         = p.ADCSampleRate * 1e3;
sweepSlope = p.SweepSlope * 1e12;
nr         = p.SamplesPerChirp;
fc         = p.CenterFrequency * 1e9;
tpulse     = 2 * p.ChirpCycleTime * 1e-6;
prf        = 1 / tpulse;
ndop       = 64;
nAngle     = 64;

rdresp = phased.RangeDopplerResponse( ...
    'RangeMethod',            'FFT', ...
    'DopplerOutput',          'Speed', ...
    'SampleRate',             fs, ...
    'SweepSlope',             sweepSlope, ...
    'OperatingFrequency',     fc, ...
    'PRFSource',              'Property', ...
    'PRF',                    prf, ...
    'RangeFFTLengthSource',   'Property', ...
    'RangeFFTLength',         nr, ...
    'DopplerFFTLengthSource', 'Property', ...
    'DopplerFFTLength',       ndop, ...
    'ReferenceRangeCentered', false);

cfar2D = phased.CFARDetector2D('GuardBandSize', 5, 'TrainingBandSize', 10, ...
    'ProbabilityFalseAlarm', 1e-5);

% 타임스탬프 로드
tsFile  = fullfile(recordLocation, 'iqData_timestamps.csv');
tsTable = readtable(tsFile);

fr = dca1000FileReader(recordLocation = recordLocation);

frameResults = cell(fr.NumDataCubes, 1);
frameIdx     = 0;
gb = 5; tb = 10; margin = gb + tb;

while fr.CurrentPosition <= fr.NumDataCubes
    iqData   = read(fr, 1);
    iqData   = iqData{1};
    frameIdx = frameIdx + 1;

    % 타임스탬프
    mask = tsTable.frame_idx == (frameIdx - 1);
    if any(mask)
        ts_vals  = tsTable.ts_ms(mask);
        frame_ts = double(ts_vals(1));
    else
        frame_ts = 0;
    end

    % 모든 RX에 대해 RD 맵 계산 (TX1: 홀수 chirp)
    rdMaps = zeros(nr, ndop, 4);
    for rx = 1:4
        iqRxTx1 = squeeze(iqData(:, rx, 1:2:end));
        if rx == 1
            [rdMap, rangeGrid, speedGrid] = rdresp(iqRxTx1);
        else
            rdMap = rdresp(iqRxTx1);
        end
        rdMaps(:, :, rx) = rdMap;
    end

    % CFAR (RX1 기준)
    resp = abs(rdMaps(:,:,1)).^2;
    [nR, nD] = size(resp);
    [columnInds, rowInds] = meshgrid(margin+1:nD-margin, margin+1:nR-margin);
    CUTIdx     = [rowInds(:) columnInds(:)]';
    detections = cfar2D(resp, CUTIdx);

    targets = struct('range_m', {}, 'velocity_mps', {}, 'azimuth_deg', {});
    for k = 1:length(detections)
        if detections(k)
            rIdx = CUTIdx(1, k);
            dIdx = CUTIdx(2, k);

            % 방위각 추정: 4 RX 스티어링 벡터 → angle FFT
            sv = zeros(4, 1);
            for rx = 1:4
                sv(rx) = rdMaps(rIdx, dIdx, rx);
            end
            angleSpec       = abs(fftshift(fft(sv, nAngle)));
            [~, peakIdx]    = max(angleSpec);
            sin_theta       = (peakIdx - nAngle/2 - 1) / (nAngle/2);
            sin_theta       = max(-1, min(1, sin_theta));
            theta_deg       = asind(sin_theta);

            targets(end+1).range_m      = rangeGrid(rIdx);
            targets(end).velocity_mps   = speedGrid(dIdx);
            targets(end).azimuth_deg    = theta_deg;
        end
    end

    frameData.frame_idx = frameIdx - 1;
    frameData.ts_ms     = frame_ts;
    if isempty(targets)
        frameData.targets = {};
    else
        frameData.targets = targets;
    end
    frameResults{frameIdx} = frameData;

    fprintf('프레임 %d / %d : %d개 검출\n', frameIdx, fr.NumDataCubes, length(targets));
end

fid = fopen(outputFile, 'w');
fprintf(fid, '%s', jsonencode(frameResults));
fclose(fid);
fprintf('\n저장 완료: %s\n', outputFile);

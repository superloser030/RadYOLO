% AWR1642 실시간 CFAR 검출 — .bin 을 폴링하며 targets.json 을 계속 갱신.
% 배치판(cfar_detect.m)과 달리 무한 루프로 최근 프레임만 처리해 덮어쓴다.
% targets.json 은 .bin 에서 재생성 가능한 가공물이라 덮어쓰기만 함(학습은 .bin 사용).
%
% 사용: 데스크톱 MATLAB 에서 실행 (main_r.py 와 병행). Ctrl+C 로 종료.
% ⚠ RecordingParameters(.mat, chirp 등)는 시작 시 1회 로드. 레벨(chirp)을 바꾸면
%   이 스크립트를 재시작할 것.

recordLocation = "D:\projects\RadYOLO\data\radar";
outputFile     = "D:\projects\RadYOLO\data\radar\targets.json";
tmpFile        = "D:\projects\RadYOLO\data\radar\targets.json.tmp";
tsFile         = fullfile(recordLocation, 'iqData_timestamps.csv');

K_RECENT = 10;    % 매 폴링마다 처리할 최근 프레임 수
POLL_SEC = 0.3;   % 폴링 간격

% ── 파라미터 로드 (1회) ─────────────────────────────
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
gb = 5; tb = 10; margin = gb + tb;

fprintf('실시간 CFAR 시작 (Ctrl+C 로 종료)\n');

while true
    % 매 폴링마다 reader 재생성 → 새로 추가된 .bin/프레임 반영
    try
        fr = dca1000FileReader(recordLocation = recordLocation);
    catch
        pause(POLL_SEC); continue;
    end
    N = fr.NumDataCubes;
    if N < 2
        pause(POLL_SEC); continue;   % 아직 프레임 부족
    end

    % 마지막 프레임은 receiver 가 쓰는 중일 수 있어 N-1 까지. 최근 K개만.
    lastIdx  = N - 1;
    startIdx = max(1, lastIdx - K_RECENT + 1);

    % 타임스탬프 (매번 새로 읽음 — 계속 추가되므로)
    try
        tsTable = readtable(tsFile);
    catch
        tsTable = table();
    end

    % 최근 프레임으로 점프
    try
        fr.CurrentPosition = startIdx;
    catch
        % CurrentPosition 설정 미지원 시: startIdx 전까지 순차 스킵
        for s = 1:(startIdx-1)
            read(fr, 1);
        end
    end

    frameResults = {};
    fIdx = startIdx;
    while fr.CurrentPosition <= lastIdx
        iqData = read(fr, 1);
        iqData = iqData{1};

        % 타임스탬프 (frame_idx 는 0-based)
        frame_ts = 0;
        if ~isempty(tsTable) && any(strcmp('frame_idx', tsTable.Properties.VariableNames))
            mask = tsTable.frame_idx == (fIdx - 1);
            if any(mask)
                ts_vals  = tsTable.ts_ms(mask);
                frame_ts = double(ts_vals(1));
            end
        end

        % 모든 RX 에 대해 RD 맵 (TX1: 홀수 chirp)
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
                sv = zeros(4, 1);
                for rx = 1:4
                    sv(rx) = rdMaps(rIdx, dIdx, rx);
                end
                angleSpec    = abs(fftshift(fft(sv, nAngle)));
                [~, peakIdx] = max(angleSpec);
                sin_theta    = (peakIdx - nAngle/2 - 1) / (nAngle/2);
                sin_theta    = max(-1, min(1, sin_theta));
                theta_deg    = asind(sin_theta);
                targets(end+1).range_m    = rangeGrid(rIdx);
                targets(end).velocity_mps = speedGrid(dIdx);
                targets(end).azimuth_deg  = theta_deg;
            end
        end

        frameData.frame_idx = fIdx - 1;
        frameData.ts_ms     = frame_ts;
        if isempty(targets)
            frameData.targets = {};
        else
            frameData.targets = targets;
        end
        frameResults{end+1} = frameData;
        fIdx = fIdx + 1;
    end

    % atomic write: tmp 에 쓰고 rename (live loop 이 깨진 파일 읽는 것 방지)
    % Windows 는 main_r 이 targets.json 을 읽는 순간 movefile 이 막히므로 재시도.
    fid = fopen(tmpFile, 'w');
    fprintf(fid, '%s', jsonencode(frameResults));
    fclose(fid);
    moved = false;
    for attempt = 1:20
        try
            movefile(tmpFile, outputFile);
            moved = true;
            break;
        catch
            pause(0.03);   % main_r 이 파일을 놓을 때까지 잠깐 대기 후 재시도
        end
    end
    if ~moved
        warning('targets.json movefile 실패 (계속 잠김). 다음 폴링에 재시도.');
    end

    pause(POLL_SEC);
end

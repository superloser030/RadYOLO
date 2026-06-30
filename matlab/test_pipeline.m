% test_pipeline.m — 1프레임으로 모듈 파이프라인 단계별 검증 (radar_live 전체 전에)
clear; clc;
thisDir = fileparts(mfilename('fullpath'));
addpath(fullfile(thisDir, 'modules'));

recordLocation = "D:\projects\RadYOLO\data\radar";
temp = load(fullfile(recordLocation, 'iqData_RecordingParameters.mat'));
rp = temp.RecordingParameters;
c          = 3e8;
fs         = rp.ADCSampleRate * 1e3;
sweepSlope = rp.SweepSlope * 1e12;
nr         = rp.SamplesPerChirp;
fc         = rp.CenterFrequency * 1e9;
lambda     = c / fc;
chirpCycle = rp.ChirpCycleTime * 1e-6;
Nt = 2; Nr = rp.NumReceivers; Nvr = Nt * Nr;

fprintf('파라미터: fs=%.0fkHz nr=%d fc=%.1fGHz Nr=%d NumChirps(meta)=%d\n', ...
    fs/1e3, nr, fc/1e9, Nr, rp.NumChirps);

% ── 시스템 객체 ──
varray = phased.ULA('NumElements', Nvr, 'ElementSpacing', lambda/2);
prf = 1 / (Nt * chirpCycle);
rdresp = phased.RangeDopplerResponse('PropagationSpeed', c, 'DopplerOutput', 'Speed', ...
    'OperatingFrequency', fc, 'SampleRate', fs, 'RangeMethod', 'FFT', ...
    'PRFSource', 'Property', 'PRF', prf, 'SweepSlope', sweepSlope, ...
    'RangeFFTLengthSource', 'Property', 'RangeFFTLength', nr, ...
    'DopplerFFTLengthSource', 'Auto', 'RangeWindow', 'Hann', 'DopplerWindow', 'Hann');
cfar = phased.CFARDetector2D('Method', 'CA', 'TrainingBandSize', [8 4], ...
    'GuardBandSize', [4 2], 'ProbabilityFalseAlarm', 1e-4, ...
    'ThresholdOutputPort', true);
ang = -60:0.5:60;
doa = phased.BeamscanEstimator('SensorArray', varray, 'PropagationSpeed', c, ...
    'OperatingFrequency', fc, 'DOAOutputPort', true, 'NumSignals', 1, 'ScanAngles', ang);
tracker = trackerJPDA(TrackLogic = "Integrated");
tracker.FilterInitializationFcn = @initParkingLotFilter;
V = 60 * 25 * 5;
tracker.ClutterDensity = 5/V; tracker.NewTargetDensity = 0.1/V;
tracker.DetectionProbability = 0.9; tracker.ConfirmationThreshold = 0.9;
tracker.DeletionThreshold = 1e-4;

p = struct('epsilon',2,'minNumPts',1,'azStd',6,'rangeStd',0.6,'rrStd',0.5, ...
    'cfarMarginR',12,'cfarMarginD',6,'rGateMin',0.3,'rGateMax',6.0,'topN',30);

% ── 1프레임 파이프라인 ──
fr = dca1000FileReader(recordLocation = recordLocation);
fprintf('NumDataCubes = %d\n', fr.NumDataCubes);

xrv = readRadarCube(fr, 1);
fprintf('① readRadarCube 가상큐브: %s  (기대 [256 x 8 x 64])\n', mat2str(size(xrv)));

[resp, rg, dg] = rangeDopplerMap(xrv, rdresp);
fprintf('② rangeDopplerMap: %s  range %.2f~%.2fm, speed %.1f~%.1f m/s\n', ...
    mat2str(size(resp)), rg(1), rg(end), dg(1), dg(end));

[ri, di, ~] = cfarDetect(resp, cfar, p, rg);
fprintf('③ cfarDetect 검출: %d개\n', numel(ri));

if ~isempty(ri)
    az = estimateAzimuth(resp, ri, di, doa);
    fprintf('④ estimateAzimuth 방위각(deg): %s\n', mat2str(round(az(:)', 1)));
    fprintf('   해당 거리(m): %s\n', mat2str(round(rg(ri(:))', 2)));
    dets = buildDetections(0.1, az, rg(ri), dg(di), p);
    fprintf('   buildDetections: objectDetection %d개 생성\n', numel(dets));

    % ── 클러스터 상세 (DBSCAN epsilon 적정성 확인) ──
    clusters = partitionDetections(dets, p.epsilon, p.minNumPts, 'Algorithm', 'DBSCAN');
    rngVals  = rg(ri);
    azCol    = az(:);
    fprintf('   클러스터 분포(점별 ID): %s\n', mat2str(double(clusters(:)')));
    uc = unique(clusters);
    for k = 1:numel(uc)
        sel = clusters == uc(k);
        fprintf('   ▶ 클러스터 %d: %2d점 | 거리 %.2f±%.2fm | 방위 %+.1f±%.1f°\n', ...
            double(uc(k)), sum(sel), mean(rngVals(sel)), std(rngVals(sel)), ...
            mean(azCol(sel)), std(azCol(sel)));
    end

    [tracks, cdets] = trackObjects(dets, tracker, 0.1, p);
    fprintf('⑤ trackObjects: 클러스터 centroid %d개, 트랙 %d개\n', numel(cdets), numel(tracks));

    % ── 파라미터 스윕: azStd/epsilon 별 클러스터 수 (촘촘하게) ──
    azList  = [6 5 4 3 2.5 2 1.5 1 0.5];
    epsList = [3 2.5 2 1.5 1 0.7 0.5 0.3];
    fprintf('\n=== 클러스터 수 스윕 (행=azStd°, 열=eps) ===\n');
    fprintf('azStd\\eps');
    for e = epsList, fprintf('%6.1f', e); end
    fprintf('\n');
    for a = azList
        fprintf('%7.1f  ', a);
        p2 = p; p2.azStd = a;
        d2 = buildDetections(0.1, az, rg(ri), dg(di), p2);
        for e = epsList
            cl = partitionDetections(d2, e, 1, 'Algorithm', 'DBSCAN');
            fprintf('%6d', numel(unique(cl)));
        end
        fprintf('\n');
    end
else
    fprintf('④⑤ 검출 0개 — CFAR Pfa 완화(1e-4→1e-3) 또는 거리게이트 확대 필요\n');
end

fprintf('=== 파이프라인 통과 (에러 없으면 OK) ===\n');

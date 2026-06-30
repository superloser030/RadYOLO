function [riList, diList, powList, snrList] = cfarDetect(resp, cfar, p, rngGrid)
% 참조 채널(가상 1번)의 전력 맵으로 검출 (위상은 estimateAzimuth 에서 사용)
% powList = 검출 셀 강도(power=|반사|²), snrList = power / noise_threshold (신뢰도)
powmap = abs(squeeze(resp(:, 1, :))).^2;       % [range x doppler]
[nR, nD] = size(powmap);

% ── 가장자리(밴드 부족) 제외한 안쪽 셀을 CUT 으로 ──
mr = p.cfarMarginR; md = p.cfarMarginD;
if nR <= 2*mr || nD <= 2*md
    riList = []; diList = []; powList = []; snrList = []; return;   % 맵이 너무 작음
end
[cutR, cutD] = meshgrid((1+mr):(nR-mr), (1+md):(nD-md));
cutidx = [cutR(:)'; cutD(:)'];

[dets, th] = cfar(powmap, cutidx);             % dets=검출여부, th=noise threshold(CUT별)
keepDet = logical(dets);
sel     = cutidx(:, keepDet);
thDet   = th(keepDet);                         % 검출된 CUT 의 threshold
riList = sel(1, :)';
diList = sel(2, :)';

% ── 우리 수정: 실내 거리 게이트 (rGateMin~Max m) ──
if ~isempty(riList)
    rg   = rngGrid(riList);
    keep = rg >= p.rGateMin & rg <= p.rGateMax;
    riList = riList(keep);
    diList = diList(keep);
    thDet  = thDet(keep);
end

% ── power(강도) + SNR(power/threshold) ──
powList = zeros(numel(riList), 1);
snrList = zeros(numel(riList), 1);
if ~isempty(riList)
    powList = powmap(sub2ind([nR, nD], riList, diList));
    snrList = powList ./ max(thDet(:), eps);   % 잡음 대비 신뢰도
    % ── 우리 수정: 상위 topN 강한 검출만 (angle 처리 비용/노이즈 절감) ──
    if numel(powList) > p.topN
        [~, ord] = sort(snrList, 'descend');   % SNR 기준 강한 순
        ord    = ord(1:p.topN);
        riList = riList(ord);
        diList = diList(ord);
        powList = powList(ord);
        snrList = snrList(ord);
    end
end
end

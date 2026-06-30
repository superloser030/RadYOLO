function [riList, diList, snr] = cfarDetect(resp, cfar, p, rngGrid)
% 참조 채널(가상 1번)의 전력 맵으로 검출 (위상은 estimateAzimuth 에서 사용)
powmap = abs(squeeze(resp(:, 1, :))).^2;       % [range x doppler]
[nR, nD] = size(powmap);

% ── 가장자리(밴드 부족) 제외한 안쪽 셀을 CUT 으로 ──
mr = p.cfarMarginR; md = p.cfarMarginD;
if nR <= 2*mr || nD <= 2*md
    riList = []; diList = []; snr = []; return;   % 맵이 너무 작음
end
[cutR, cutD] = meshgrid((1+mr):(nR-mr), (1+md):(nD-md));
cutidx = [cutR(:)'; cutD(:)'];

dets = cfar(powmap, cutidx);                   % logical 벡터 (CUT 별 검출 여부)
sel  = cutidx(:, logical(dets));
riList = sel(1, :)';
diList = sel(2, :)';

% ── 우리 수정: 실내 거리 게이트 (rGateMin~Max m) ──
if ~isempty(riList)
    rg   = rngGrid(riList);
    keep = rg >= p.rGateMin & rg <= p.rGateMax;
    riList = riList(keep);
    diList = diList(keep);
end

% ── 우리 수정: 상위 topN 강한 검출만 (angle 처리 비용/노이즈 절감) ──
snr = zeros(numel(riList), 1);
if ~isempty(riList)
    snr = powmap(sub2ind([nR, nD], riList, diList));
    if numel(snr) > p.topN
        [~, ord] = sort(snr, 'descend');
        ord    = ord(1:p.topN);
        riList = riList(ord);
        diList = diList(ord);
        snr    = snr(ord);
    end
end
end

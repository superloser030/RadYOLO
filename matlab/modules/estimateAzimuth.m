function azList = estimateAzimuth(resp, riList, diList, doa)
n = numel(riList);
azList = zeros(n, 1);
for i = 1:n
    % BeamscanEstimator 입력은 [스냅샷 x 채널] → 가상배열 8채널을 [1 x 8] 로
    sv = reshape(resp(riList(i), :, diList(i)), 1, []);   % [1 x 8] (1 snapshot x 8 ch)
    [~, az] = doa(sv);                             % beamscan → DOA(deg)
    azList(i) = az;
end
end

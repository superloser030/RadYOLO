function [tracks, clusteredDets] = trackObjects(dets, tracker, time, p)
% ── [ParkingLot 예제 line 110-118] DBSCAN 클러스터 + centroid ──
if isempty(dets)
    clusters = zeros(0, 1, 'uint32');
else
    clusters = partitionDetections(dets, p.epsilon, p.minNumPts, 'Algorithm', 'DBSCAN');
end
clusteredDets = mergeDetections(dets, clusters);

% ── [ParkingLot 예제 line 120-123] JIPDA 추적 ──
tracks = objectTrack.empty(0, 1);
if isLocked(tracker) || ~isempty(clusteredDets)
    tracks = tracker(clusteredDets, time);
end
end

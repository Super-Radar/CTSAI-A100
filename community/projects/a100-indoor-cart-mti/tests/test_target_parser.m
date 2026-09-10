function test_target_parser()
%TEST_TARGET_PARSER Verify mixed metadata/target row parsing.

fixture = [tempname '.csv'];
cleanup = onCleanup(@() delete_if_exists(fixture));
fid = fopen(fixture, 'w');
assert(fid >= 0);
fprintf(fid, ['ObjId,range,speed,angle,snr,Kstate,OboCarSpeed,heigh,heighAngle,' ...
              'X,Y,EstimateCarSpeed,EstimateMagCarSpeed,RCS,trkReiablility,ObstacleProb\n']);
fprintf(fid, 'START,FrameNb:7,CameraNb:0\n');
fprintf(fid, '2026/08/02 16:27:16:939---->\n');
fprintf(fid, 'heightOffset:0,metadata\n');
fprintf(fid, '4,7.28,-0.25,-2.07,12,0,0,0,0,0.263,7.275,-0.25,0,-6.3,98,99,\n');
fprintf(fid, 'START,FrameNb:8,CameraNb:0\n');
fprintf(fid, '2026/08/02 16:27:16:999---->\n');
fprintf(fid, '4,7.20,-0.25,-2.10,13,0,0,0,0,0.264,7.195,-0.25,0,-6.1,98,99,\n');
fclose(fid);

targets = read_radartools_targets(fixture);
assert(targets.meta.record_count == 2);
assert(targets.meta.frame_count == 2);
assert(targets.meta.target_frame_count == 2);
assert(isequal(targets.frame_nb, [7; 8]));
assert(abs(targets.elapsed_s(2) - 0.060) < 1e-9);
assert(all(targets.obj_id == 4));
assert(abs(targets.polar_y_m(1) - 7.275) < 0.01);
assert(targets.reliability(1) == 98);

% 完整日期必须保证跨午夜时间仍单调；空帧也应计入检测率分母。
midnight_fixture = [tempname '.csv'];
midnight_cleanup = onCleanup(@() delete_if_exists(midnight_fixture));
fid = fopen(midnight_fixture, 'w');
assert(fid >= 0);
fprintf(fid, ['ObjId,range,speed,angle,snr,Kstate,OboCarSpeed,heigh,heighAngle,' ...
              'X,Y,EstimateCarSpeed,EstimateMagCarSpeed,RCS,trkReiablility,ObstacleProb\n']);
fprintf(fid, 'START,FrameNb:20,CameraNb:0\n');
fprintf(fid, '2026/08/02 23:59:59:900---->\n');
fprintf(fid, '4,7.28,-0.25,-2.07,12,0,0,0,0,0.263,7.275,-0.25,0,-6.3,98,99\n');
fprintf(fid, 'START,FrameNb:21,CameraNb:0\n');
fprintf(fid, '2026/08/03 00:00:00:100---->\n');
fprintf(fid, '4,7.20,-0.25,-2.10,13,0,0,0,0,0.264,7.195,-0.25,0,-6.1,98,99\n');
fprintf(fid, 'START,FrameNb:22,CameraNb:0\n');
fprintf(fid, 'START,FrameNb:23,CameraNb:0\n');
fprintf(fid, '4,7.10,-0.25,-2.10,13,0,0,0,0,0.264,7.095,-0.25,0,-6.1,98,99\n');
fclose(fid);

midnight_targets = read_radartools_targets(midnight_fixture);
assert(midnight_targets.meta.record_count == 3);
assert(midnight_targets.meta.frame_count == 4);
assert(midnight_targets.meta.target_frame_count == 3);
assert(abs(midnight_targets.elapsed_s(2) - 0.200) < 1e-4);
assert(isnan(midnight_targets.elapsed_s(3)));

synthetic = targets;
synthetic.speed_mps = [-0.2; -0.2];
synthetic.snr_db = [20; 22];
synthetic.range_m = [2.5; 2.4];
synthetic.polar_x_m = -synthetic.range_m .* sin(synthetic.angle_deg * pi / 180);
synthetic.polar_y_m = synthetic.range_m .* cos(synthetic.angle_deg * pi / 180);
track = extract_motion_track(synthetic);
assert(track.point_count == 2);
assert(track.end_range_m < track.start_range_m);
assert(strcmp(track.inferred_direction, 'approaching'));

synthetic.speed_mps = [0.2; 0.2];
synthetic.range_m = [2.4; 2.5];
synthetic.polar_x_m = -synthetic.range_m .* sin(synthetic.angle_deg * pi / 180);
synthetic.polar_y_m = synthetic.range_m .* cos(synthetic.angle_deg * pi / 180);
track = extract_motion_track(synthetic);
assert(track.point_count == 2);
assert(track.end_range_m > track.start_range_m);
assert(strcmp(track.inferred_direction, 'receding'));

synthetic.speed_mps = [0; 0];
track = extract_motion_track(synthetic);
assert(track.point_count == 0);
assert(strcmp(track.inferred_direction, 'unknown'));

clear cleanup;
delete_if_exists(fixture);
clear midnight_cleanup;
delete_if_exists(midnight_fixture);
end

function delete_if_exists(file_path)
if exist(file_path, 'file')
    delete(file_path);
end
end

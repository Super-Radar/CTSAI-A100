function targets = read_radartools_targets(file_path)
%READ_RADARTOOLS_TARGETS Parse RadarTools CSV files containing metadata rows.
%   START, timestamp, and status rows are retained as frame context but are
%   not treated as targets. A target row must match the header after removal
%   of trailing empty fields and must contain finite key numeric values.

if ~exist(file_path, 'file')
    error('a100:TargetFileNotFound', 'Target CSV not found: %s', file_path);
end
fid = fopen(file_path, 'r');
if fid < 0
    error('a100:TargetFileOpenFailed', 'Could not open target CSV: %s', file_path);
end
cleanup = onCleanup(@() fclose(fid));

header_line = fgetl(fid);
if ~ischar(header_line)
    error('a100:EmptyTargetFile', 'Target CSV is empty: %s', file_path);
end
headers = split_csv(header_line);
required = {'ObjId', 'range', 'speed', 'angle', 'snr', 'X', 'Y', ...
            'EstimateCarSpeed', 'RCS'};
indices = struct();
for idx = 1:numel(required)
    match = find(strcmp(headers, required{idx}), 1);
    if isempty(match)
        error('a100:MissingTargetColumn', ...
            'CSV %s is missing required column %s.', file_path, required{idx});
    end
    indices.(required{idx}) = match;
end
optional = {'trkReiablility', 'ObstacleProb'};
for idx = 1:numel(optional)
    match = find(strcmp(headers, optional{idx}), 1);
    if isempty(match)
        indices.(optional{idx}) = NaN;
    else
        indices.(optional{idx}) = match;
    end
end

capacity = 4096;
frame_nb = nan(capacity, 1);
frame_sequence = nan(capacity, 1);
timestamp_ms = nan(capacity, 1);
obj_id = nan(capacity, 1);
range_m = nan(capacity, 1);
speed_mps = nan(capacity, 1);
angle_deg = nan(capacity, 1);
snr_db = nan(capacity, 1);
x_m = nan(capacity, 1);
y_m = nan(capacity, 1);
estimated_speed_mps = nan(capacity, 1);
rcs_dbsm = nan(capacity, 1);
reliability = nan(capacity, 1);
obstacle_probability = nan(capacity, 1);

count = 0;
ignored_lines = 0;
current_frame = NaN;
current_frame_sequence = -1;
current_timestamp_ms = NaN;
total_frame_count = 0;
line = fgetl(fid);
while ischar(line)
    start_token = regexp(line, '^START,FrameNb:([0-9]+)', 'tokens', 'once');
    if ~isempty(start_token)
        current_frame = str2double(start_token{1});
        current_frame_sequence = current_frame_sequence + 1;
        current_timestamp_ms = NaN;
        total_frame_count = total_frame_count + 1;
        line = fgetl(fid);
        continue;
    end

    timestamp_token = regexp(line, ...
        '^([0-9]{4})/([0-9]{2})/([0-9]{2}) ([0-9]{2}):([0-9]{2}):([0-9]{2}):([0-9]{3})', ...
        'tokens', 'once');
    if ~isempty(timestamp_token)
        timestamp_values = cellfun(@str2double, timestamp_token);
        day_index = round(datenum(timestamp_values(1), timestamp_values(2), ...
            timestamp_values(3)) - datenum(1970, 1, 1));
        current_timestamp_ms = (((day_index * 24 + timestamp_values(4)) * 60 + ...
            timestamp_values(5)) * 60 + timestamp_values(6)) * 1000 + ...
            timestamp_values(7);
        line = fgetl(fid);
        continue;
    end

    fields = split_csv(line);
    while numel(fields) > numel(headers) && isempty(fields{end})
        fields(end) = [];
    end
    if numel(fields) ~= numel(headers) || isempty(regexp(fields{1}, '^-?[0-9]+$', 'once'))
        ignored_lines = ignored_lines + 1;
        line = fgetl(fid);
        continue;
    end

    key_values = [to_number(fields{indices.ObjId}), ...
                  to_number(fields{indices.range}), ...
                  to_number(fields{indices.speed}), ...
                  to_number(fields{indices.angle})];
    if any(~isfinite(key_values)) || key_values(2) < 0
        ignored_lines = ignored_lines + 1;
        line = fgetl(fid);
        continue;
    end

    count = count + 1;
    if count > capacity
        old_capacity = capacity;
        capacity = capacity * 2;
        frame_nb(old_capacity + 1:capacity, 1) = NaN;
        frame_sequence(old_capacity + 1:capacity, 1) = NaN;
        timestamp_ms(old_capacity + 1:capacity, 1) = NaN;
        obj_id(old_capacity + 1:capacity, 1) = NaN;
        range_m(old_capacity + 1:capacity, 1) = NaN;
        speed_mps(old_capacity + 1:capacity, 1) = NaN;
        angle_deg(old_capacity + 1:capacity, 1) = NaN;
        snr_db(old_capacity + 1:capacity, 1) = NaN;
        x_m(old_capacity + 1:capacity, 1) = NaN;
        y_m(old_capacity + 1:capacity, 1) = NaN;
        estimated_speed_mps(old_capacity + 1:capacity, 1) = NaN;
        rcs_dbsm(old_capacity + 1:capacity, 1) = NaN;
        reliability(old_capacity + 1:capacity, 1) = NaN;
        obstacle_probability(old_capacity + 1:capacity, 1) = NaN;
    end

    frame_nb(count) = current_frame;
    frame_sequence(count) = current_frame_sequence;
    timestamp_ms(count) = current_timestamp_ms;
    obj_id(count) = key_values(1);
    range_m(count) = key_values(2);
    speed_mps(count) = key_values(3);
    angle_deg(count) = key_values(4);
    snr_db(count) = to_number(fields{indices.snr});
    x_m(count) = to_number(fields{indices.X});
    y_m(count) = to_number(fields{indices.Y});
    estimated_speed_mps(count) = to_number(fields{indices.EstimateCarSpeed});
    rcs_dbsm(count) = to_number(fields{indices.RCS});
    reliability(count) = optional_number(fields, indices.trkReiablility);
    obstacle_probability(count) = optional_number(fields, indices.ObstacleProb);

    line = fgetl(fid);
end
clear cleanup;

fields_to_trim = {'frame_nb', 'frame_sequence', 'timestamp_ms', 'obj_id', ...
                  'range_m', 'speed_mps', 'angle_deg', 'snr_db', 'x_m', 'y_m', ...
                  'estimated_speed_mps', 'rcs_dbsm', 'reliability', ...
                  'obstacle_probability'};
values = {frame_nb, frame_sequence, timestamp_ms, obj_id, range_m, speed_mps, ...
          angle_deg, snr_db, x_m, y_m, estimated_speed_mps, rcs_dbsm, ...
          reliability, obstacle_probability};
targets = struct();
for idx = 1:numel(fields_to_trim)
    targets.(fields_to_trim{idx}) = values{idx}(1:count);
end
% 毫秒整数先做差，避免大绝对时间戳相减造成亚秒精度损失。
targets.timestamp_s = targets.timestamp_ms / 1000;
targets.time_of_day_s = mod(targets.timestamp_ms, 86400000) / 1000;
targets.elapsed_s = elapsed_seconds(targets.timestamp_ms);
targets.polar_x_m = -targets.range_m .* sin(targets.angle_deg * pi / 180);
targets.polar_y_m = targets.range_m .* cos(targets.angle_deg * pi / 180);
targets.meta = struct('file', file_path, ...
                      'record_count', count, ...
                      'ignored_line_count', ignored_lines, ...
                      'frame_count', total_frame_count, ...
                      'target_frame_count', ...
                          numel(unique(targets.frame_sequence)), ...
                      'headers', {headers});
end

function fields = split_csv(line)
fields = regexp(line, ',', 'split');
end

function value = to_number(text)
if isempty(text)
    value = NaN;
else
    value = str2double(text);
end
end

function value = optional_number(fields, index)
if isnan(index)
    value = NaN;
else
    value = to_number(fields{index});
end
end

function elapsed = elapsed_seconds(timestamp_ms)
elapsed = nan(size(timestamp_ms));
valid = isfinite(timestamp_ms);
if any(valid)
    valid_indices = find(valid);
    start_time_ms = timestamp_ms(valid_indices(1));
    elapsed(valid) = (timestamp_ms(valid) - start_time_ms) / 1000;
end
end

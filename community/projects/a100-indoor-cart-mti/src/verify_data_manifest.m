function report = verify_data_manifest(project_root)
%VERIFY_DATA_MANIFEST Check every declared acquisition file before analysis.

if nargin < 1 || isempty(project_root)
    project_root = fileparts(fileparts(mfilename('fullpath')));
end
manifest_path = fullfile(project_root, 'data', 'manifest.csv');
if ~exist(manifest_path, 'file')
    error('a100:ManifestNotFound', 'Data manifest not found: %s', manifest_path);
end

fid = fopen(manifest_path, 'r');
if fid < 0
    error('a100:ManifestOpenFailed', 'Could not open data manifest: %s', manifest_path);
end
cleanup = onCleanup(@() fclose(fid));
header_line = fgetl(fid);
if ~ischar(header_line)
    error('a100:EmptyManifest', 'Data manifest is empty: %s', manifest_path);
end
headers = regexp(header_line, ',', 'split');
path_column = required_column(headers, 'relative_path');
size_column = required_column(headers, 'size_bytes');
hash_column = required_column(headers, 'sha256');

entry_count = 0;
total_bytes = 0;
seen_paths = {};
line_number = 1;
line = fgetl(fid);
while ischar(line)
    line_number = line_number + 1;
    if isempty(strtrim(line))
        line = fgetl(fid);
        continue;
    end
    fields = regexp(line, ',', 'split');
    if numel(fields) ~= numel(headers)
        error('a100:MalformedManifestRow', ...
            'Manifest line %d has %d fields; expected %d.', ...
            line_number, numel(fields), numel(headers));
    end

    relative_path = strrep(strtrim(fields{path_column}), '\', '/');
    expected_size = str2double(strtrim(fields{size_column}));
    expected_hash = lower(strtrim(fields{hash_column}));
    validate_relative_path(relative_path, line_number);
    if any(strcmp(seen_paths, relative_path))
        error('a100:DuplicateManifestPath', ...
            'Manifest line %d repeats path "%s".', line_number, relative_path);
    end
    seen_paths{end + 1} = relative_path; %#ok<AGROW>
    if ~isfinite(expected_size) || expected_size < 0 || expected_size ~= floor(expected_size)
        error('a100:InvalidManifestSize', ...
            'Manifest line %d has invalid size "%s".', line_number, fields{size_column});
    end
    if isempty(regexp(expected_hash, '^[0-9a-f]{64}$', 'once'))
        error('a100:InvalidManifestHash', ...
            'Manifest line %d has an invalid SHA-256 value.', line_number);
    end

    file_path = fullfile(project_root, strrep(relative_path, '/', filesep));
    if ~exist(file_path, 'file')
        error('a100:ManifestFileMissing', ...
            'Manifest entry is missing: %s', relative_path);
    end
    info = dir(file_path);
    if info.bytes ~= expected_size
        error('a100:ManifestSizeMismatch', ...
            'Manifest size mismatch for %s: expected %d, found %d.', ...
            relative_path, expected_size, info.bytes);
    end
    actual_hash = sha256_file(file_path);
    if ~strcmp(actual_hash, expected_hash)
        error('a100:ManifestHashMismatch', ...
            'Manifest SHA-256 mismatch for %s.', relative_path);
    end

    entry_count = entry_count + 1;
    total_bytes = total_bytes + expected_size;
    line = fgetl(fid);
end
clear cleanup;

if entry_count == 0
    error('a100:EmptyManifest', 'Data manifest has no file entries: %s', manifest_path);
end
report = struct('manifest_file', manifest_path, ...
                'entry_count', entry_count, ...
                'total_bytes', total_bytes, ...
                'sha256_verified', true);
end

function index = required_column(headers, name)
index = find(strcmp(headers, name), 1);
if isempty(index)
    error('a100:MissingManifestColumn', ...
        'Data manifest is missing required column %s.', name);
end
end

function validate_relative_path(relative_path, line_number)
parts = regexp(relative_path, '/', 'split');
is_absolute = isempty(relative_path) || relative_path(1) == '/' || ...
              ~isempty(regexp(relative_path, '^[A-Za-z]:', 'once'));
if is_absolute || any(strcmp(parts, '..')) || ~strncmp(relative_path, 'data/', 5)
    error('a100:UnsafeManifestPath', ...
        'Manifest line %d has unsafe path "%s".', line_number, relative_path);
end
end

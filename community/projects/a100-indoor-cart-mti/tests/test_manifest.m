function test_manifest(project_root)
%TEST_MANIFEST Verify real-data provenance and corruption detection.

real_report = verify_data_manifest(project_root);
assert(real_report.entry_count == 31);
assert(real_report.total_bytes == 51666387);
assert(real_report.sha256_verified);

fixture_root = tempname;
mkdir(fixture_root);
mkdir(fullfile(fixture_root, 'data'));
cleanup = onCleanup(@() remove_fixture(fixture_root));
data_path = fullfile(fixture_root, 'data', 'sample.txt');
write_bytes(data_path, uint8('abc'));
manifest_path = fullfile(fixture_root, 'data', 'manifest.csv');
write_manifest(manifest_path, 3, ...
    'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad');

fixture_report = verify_data_manifest(fixture_root);
assert(fixture_report.entry_count == 1);
assert(fixture_report.total_bytes == 3);

write_bytes(data_path, uint8('abd'));
assert_error_id(@() verify_data_manifest(fixture_root), 'a100:ManifestHashMismatch');

write_bytes(data_path, uint8('abc'));
write_duplicate_manifest(manifest_path, 3, ...
    'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad');
assert_error_id(@() verify_data_manifest(fixture_root), 'a100:DuplicateManifestPath');

clear cleanup;
remove_fixture(fixture_root);
end

function write_duplicate_manifest(file_path, size_bytes, digest)
fid = fopen(file_path, 'w');
assert(fid >= 0);
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, 'relative_path,size_bytes,sha256\n');
fprintf(fid, 'data/sample.txt,%d,%s\n', size_bytes, digest);
fprintf(fid, 'data/sample.txt,%d,%s\n', size_bytes, digest);
clear cleanup;
end

function write_manifest(file_path, size_bytes, digest)
fid = fopen(file_path, 'w');
assert(fid >= 0);
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, 'relative_path,size_bytes,sha256\n');
fprintf(fid, 'data/sample.txt,%d,%s\n', size_bytes, digest);
clear cleanup;
end

function write_bytes(file_path, bytes)
fid = fopen(file_path, 'wb');
assert(fid >= 0);
cleanup = onCleanup(@() fclose(fid));
fwrite(fid, bytes, 'uint8');
clear cleanup;
end

function remove_fixture(directory)
if exist(directory, 'dir')
    rmdir(directory, 's');
end
end

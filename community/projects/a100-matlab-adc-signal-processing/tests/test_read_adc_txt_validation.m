function test_read_adc_txt_validation()
%TEST_READ_ADC_TXT_VALIDATION 验证 ADC token、头部与尾部填充的严格校验。

projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(projectRoot, 'functions'));
fixture = [tempname '.txt'];
cleanupObj = onCleanup(@() delete_if_exists(fixture)); %#ok<NASGU>
ioOpts.allow_zero_pad = false;

valid = [0, 4, 2, 1, 2, 3, double(intmax('uint32')), 0, 0];
write_values(fixture, valid);
[words, header, trailer] = read_adc_txt(fixture, ioOpts);
assert(isequal(words(:), uint32(valid(4:7)).'));
assert(header.rx_index == 0 && header.samples_per_chirp == 4 && header.chirp_count == 2);
assert(isequal(trailer(:), [0; 0]));

cases = {
    [Inf, 4, 2, 1, 2, 3, 4], 'read_adc_txt:InvalidHeader';
    [0.5, 4, 2, 1, 2, 3, 4], 'read_adc_txt:InvalidHeader';
    [4, 4, 2, 1, 2, 3, 4], 'read_adc_txt:InvalidRxIndex';
    [0, 3, 2, 1, 2, 3, 4], 'read_adc_txt:InvalidSamplesPerChirp';
    [0, 4, 0, 1], 'read_adc_txt:InvalidChirpCount';
    [0, 4294967294, 4294967295, 1], 'read_adc_txt:InvalidDimensions';
    [0, 4, 2, 1, 2, NaN, 4], 'read_adc_txt:InvalidPayload';
    [0, 4, 2, 1, 2, 3.5, 4], 'read_adc_txt:InvalidPayload';
    [0, 4, 2, 1, 2, -1, 4], 'read_adc_txt:InvalidPayload';
    [0, 4, 2, 1, 2, 3, 4294967296], 'read_adc_txt:InvalidPayload';
    [0, 4, 2, 1, 2, 3, 4, 0, 7], 'read_adc_txt:NonZeroTrailer'
};
for iCase = 1:size(cases, 1)
    write_values(fixture, cases{iCase, 1});
    assert_error_id(@() read_adc_txt(fixture, ioOpts), cases{iCase, 2});
end
write_text(fixture, '0,4,2,1,2,3,4,bad');
assert_error_id(@() read_adc_txt(fixture, ioOpts), 'read_adc_txt:InvalidPayload');
end

function assert_error_id(callback, expectedId)
try
    callback();
catch err
    assert(strcmp(err.identifier, expectedId), ...
        'Expected %s, received %s.', expectedId, err.identifier);
    return;
end
error('test_read_adc_txt_validation:MissingError', 'Expected error %s.', expectedId);
end

function write_values(filePath, values)
fid = fopen(filePath, 'w');
assert(fid >= 0, 'Could not create fixture.');
cleanupObj = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '%.17g', values(1));
fprintf(fid, ',%.17g', values(2:end));
fprintf(fid, '\n');
end

function write_text(filePath, contents)
fid = fopen(filePath, 'w');
assert(fid >= 0, 'Could not create fixture.');
cleanupObj = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '%s\n', contents);
end

function delete_if_exists(filePath)
if exist(filePath, 'file')
    delete(filePath);
end
end

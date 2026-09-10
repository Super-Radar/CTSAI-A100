function [packedWords, header, trailer] = read_adc_txt(filePath, ioOpts)
%READ_ADC_TXT Read one CTSAI-A100 RX export file.
% Format: rx_index, samples_per_chirp, chirp_count, packed_uint32...

fid = fopen(filePath, 'r');
if fid < 0
    error('Cannot open ADC file: %s', filePath);
end
cleanupObj = onCleanup(@() fclose(fid)); %#ok<NASGU>
tokenCell = textscan(fid, '%s', 'Delimiter', ',');
values = str2double(tokenCell{1});
if numel(values) < 4
    error('ADC file is too short: %s', filePath);
end

headerValues = values(1:3);
if any(~isfinite(headerValues)) || any(headerValues ~= fix(headerValues)) || ...
        any(abs(headerValues) > flintmax)
    error('read_adc_txt:InvalidHeader', ...
        'ADC header fields must be finite, exactly represented integers.');
end

header.rx_index = headerValues(1);
header.samples_per_chirp = headerValues(2);
header.chirp_count = headerValues(3);
header.file = filePath;
if header.rx_index < 0 || header.rx_index > 3
    error('read_adc_txt:InvalidRxIndex', 'rx_index must be an integer from 0 to 3.');
end
if header.samples_per_chirp <= 0 || mod(header.samples_per_chirp, 2) ~= 0
    error('read_adc_txt:InvalidSamplesPerChirp', ...
        'samples_per_chirp must be a positive even integer.');
end
if header.chirp_count <= 0
    error('read_adc_txt:InvalidChirpCount', 'chirp_count must be a positive integer.');
end

expectedWords = header.samples_per_chirp / 2 * header.chirp_count;
if ~isfinite(expectedWords) || expectedWords ~= fix(expectedWords) || expectedWords > flintmax
    error('read_adc_txt:InvalidDimensions', ...
        'Header dimensions produce an invalid packed-word count.');
end
payload = values(4:end);
% uint32 转换会截断或饱和非法输入，必须在转换前拒绝数据损坏。
if any(~isfinite(payload)) || any(payload ~= fix(payload)) || ...
        any(payload < 0) || any(payload > double(intmax('uint32')))
    error('read_adc_txt:InvalidPayload', ...
        'ADC payload values must be finite uint32 integers.');
end
if numel(payload) < expectedWords
    if ioOpts.allow_zero_pad
        warning('ADC payload is short in %s; zero padding %d words.', ...
            filePath, expectedWords - numel(payload));
        payload(end+1:expectedWords, 1) = 0;
    else
        error('ADC payload is short in %s: expected %d words, got %d.', ...
            filePath, expectedWords, numel(payload));
    end
end
trailer = payload(expectedWords+1:end);
if ~isempty(trailer)
    if any(trailer ~= 0)
        error('read_adc_txt:NonZeroTrailer', ...
            'ADC trailer contains non-zero data in %s.', filePath);
    end
    fprintf('  %s: verified %d zero padding value(s).\n', ...
        char(filePath), numel(trailer));
end
packedWords = uint32(payload(1:expectedWords));
end

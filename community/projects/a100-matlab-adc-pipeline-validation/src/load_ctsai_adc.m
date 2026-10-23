function adc = load_ctsai_adc(files, cfg)
%LOAD_CTSAI_ADC Read four text files and unpack two signed int16 samples
%from each stored uint32 word. Output: [sample, chirp, rx].
% Public f0/f1 captures are headerless, with at most one extra trailing zero.

adc = zeros(cfg.numAdcSamples, cfg.numChirps, cfg.numRx);
expectedWords = cfg.numPackedSamples * cfg.numChirps;

for rx = 1:cfg.numRx
    if ~isfile(files{rx})
        error('CTSAI:MissingData', 'ADC file not found: %s', files{rx});
    end
    % Parse the exported comma-separated record without inferring header rows
    % or dropping invalid fields: either would shift subsequent ADC samples.
    tokens = strsplit(strtrim(fileread(files{rx})), ',', ...
        'CollapseDelimiters', false);
    values = str2double(tokens(:));
    if ~isreal(values) || any(~isfinite(values)) || ...
            any(values < 0 | values > double(intmax('uint32'))) || ...
            any(values ~= fix(values))
        error('CTSAI:DataValue', ...
            'RX%d must contain finite unsigned 32-bit integer ADC words.', rx-1);
    end

    % Keep the first payload word and all zeros within the expected payload.
    % The public exports have ONE extra zero at the end, not a leading header.
    if numel(values) == expectedWords + 1
        if values(end) ~= 0
            error('CTSAI:DataTrailer', ...
                'RX%d has a nonzero extra word; only one trailing zero is supported.', rx-1);
        end
        values = values(1:expectedWords);
    elseif numel(values) ~= expectedWords
        error('CTSAI:DataLength', ...
            'RX%d contains %d values; expected %d headerless words (+ optional trailing zero).', ...
            rx-1, numel(values), expectedWords);
    end

    words = uint32(values);
    high = typecast(uint16(bitshift(words, -16)), 'int16');
    low = typecast(uint16(bitand(words, uint32(65535))), 'int16');
    unpacked = zeros(2 * expectedWords, 1);
    unpacked(1:2:end) = double(high);
    unpacked(2:2:end) = double(low);
    adc(:,:,rx) = reshape(unpacked, cfg.numAdcSamples, cfg.numChirps);
end

% Remove per-chirp DC bias before windowing.
adc = adc - mean(adc, 1);
end

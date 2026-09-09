function adc = load_ctsai_adc(files, cfg)
%LOAD_CTSAI_ADC Read four text files and unpack two signed int16 samples
%from each stored uint32 word. Output: [sample, chirp, rx].

adc = zeros(cfg.numAdcSamples, cfg.numChirps, cfg.numRx);
expectedWords = cfg.numPackedSamples * cfg.numChirps;

for rx = 1:cfg.numRx
    if ~isfile(files{rx})
        error('CTSAI:MissingData', 'ADC file not found: %s', files{rx});
    end
    values = readmatrix(files{rx});
    values = values(:);

    % Public files contain one leading metadata value followed by payload.
    if numel(values) == expectedWords + 1
        values = values(2:end);
    elseif numel(values) ~= expectedWords
        error('CTSAI:DataLength', ...
            'RX%d contains %d values; expected %d payload values (+ optional header).', ...
            rx-1, numel(values), expectedWords);
    end
    if any(~isfinite(values))
        error('CTSAI:DataValue', ...
            'RX%d contains a non-finite payload value.', rx-1);
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

function run_adc_loading_tests(repoRoot)
%RUN_ADC_LOADING_TESTS Exact sample-order checks, including public captures.
projectDir = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(projectDir, 'src'));
if nargin < 1
    repoRoot = fileparts(fileparts(fileparts(projectDir)));
end
folder = tempname;
mkdir(folder);
cleanup = onCleanup(@() rmdir(folder, 's')); %#ok<NASGU>
cfg = struct('numAdcSamples',4,'numPackedSamples',2,'numChirps',2,'numRx',1);
words = [2147549183, 32767, 655380, 1966120];
samples = [-32768,10; -1,20; 0,30; 32767,40];
expected = samples - mean(samples,1);
path = fullfile(folder, 'adc.txt');

for padded = 0:1
    write_tokens(path, [words, zeros(1,padded)]);
    assert(isequal(load_ctsai_adc({path},cfg), expected));
end
samples(3:4,2) = 0;
for padded = 0:1
    write_tokens(path, [words(1:3),0,zeros(1,padded)]);
    assert(isequal(load_ctsai_adc({path},cfg), samples-mean(samples,1)));
end
write_tokens(path, zeros(1,5));
assert(isequal(load_ctsai_adc({path},cfg),zeros(4,2)));

badValues = {'NaN','Inf','-Inf','-1','4294967296','1.5','','bad','1+2i'};
for k = 1:numel(badValues)
    for padded = 0:1
        text = sprintf('2147549183,%s,655380,1966120',badValues{k});
        if padded, text = [text,',0']; end %#ok<AGROW>
        write_text(path,text);
        expect_error(@() load_ctsai_adc({path},cfg),'CTSAI:DataValue');
    end
end
write_tokens(path,[words,1]);
expect_error(@() load_ctsai_adc({path},cfg),'CTSAI:DataTrailer');
for tokens = {words(1:3),[words,0,0],[0,4,2,words]}
    write_tokens(path,tokens{1});
    expect_error(@() load_ctsai_adc({path},cfg),'CTSAI:DataLength');
end

% Test four channels with distinct sample sequences, independent of filenames.
cfg.numRx = 4;
files = cell(1,4);
expected = zeros(4,2,4);
for rx = 1:4
    x = reshape(1:8,4,2) .* rx;
    packed = x(1:2:end,:) * 65536 + x(2:2:end,:);
    files{rx} = fullfile(folder,sprintf('rx%d.txt',rx));
    write_tokens(files{rx},[packed(:).',0]);
    expected(:,:,rx) = x - mean(x,1);
end
assert(isequal(load_ctsai_adc(files,cfg),expected));
fprintf('PASS: hand-decoded samples, chirp/channel order and invalid inputs\n');

% Independent floor/mod two's-complement oracle for every public sample.
dataDir = fullfile(repoRoot,'ADC数据采集','示例adc数据和结果');
for profile = 0:1
    cfg.numAdcSamples = 1024 * (profile+1);
    cfg.numPackedSamples = cfg.numAdcSamples / 2;
    cfg.numChirps = 256 / (profile+1);
    matches = dir(fullfile(dataDir,sprintf('*_f%d_*.txt',profile)));
    assert(numel(matches)==4,'Expected the four public RX captures.');
    [~,order] = sort({matches.name});
    files = cell(1,4);
    expected = zeros(cfg.numAdcSamples,cfg.numChirps,4);
    for rx = 1:4
        files{rx} = fullfile(dataDir,matches(order(rx)).name);
        raw = sscanf(strrep(fileread(files{rx}),',',' '),'%f');
        count = cfg.numPackedSamples * cfg.numChirps;
        assert(numel(raw)==count+1 && raw(end)==0);
        payload = raw(1:count);
        high = floor(payload/65536);
        low = mod(payload,65536);
        high(high>=32768) = high(high>=32768)-65536;
        low(low>=32768) = low(low>=32768)-65536;
        pairs = [high.';low.'];
        x = reshape(pairs,cfg.numAdcSamples,cfg.numChirps);
        expected(:,:,rx) = x-mean(x,1);
    end
    actual = load_ctsai_adc(files,cfg);
    assert(isequal(actual,expected));
    fprintf('PASS: f%d, %d samples, all four RX match independent unpacking exactly\n', ...
        profile,numel(actual));
end
end

function write_tokens(path,values)
text = sprintf('%.0f,',values);
write_text(path,text(1:end-1));
end

function write_text(path,text)
fid = fopen(path,'w');
assert(fid>=0);
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid,'%s\n',text);
end

function expect_error(call,identifier)
try
    call();
catch err
    assert(strcmp(err.identifier,identifier),err.message);
    return;
end
error('Expected %s.',identifier);
end

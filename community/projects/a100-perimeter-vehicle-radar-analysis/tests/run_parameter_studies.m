clear;
clc;
close all;

%% ============================================================
% CTSAI-A100 Issue #13
% LFMCW Radar Parameter Studies
%
% Studies:
%   A. LFM bandwidth
%   B. LFM chirp duration
%   C. Chirp repetition period / PRF
%   D. Number of targets
%   E. Target velocity
%   F. MTI order
%% ============================================================

%% Project

thisFile = mfilename('fullpath');
testDir = fileparts(thisFile);
projectRoot = fileparts(testDir);

addpath(genpath(fullfile(projectRoot, 'functions')));
addpath(fullfile(projectRoot, 'config'));

resultDir = fullfile( ...
    projectRoot, ...
    'results', ...
    'simulation', ...
    'parameter_study');

if ~exist(resultDir, 'dir')
    mkdir(resultDir);
end

%% Base configuration

cfgBase = simulation_config();

dynamicRangeDb = 60;
maxPlotRangeM = 80;

fprintf('\n');
fprintf('=====================================================\n');
fprintf(' CTSAI-A100 Issue #13 Parameter Studies\n');
fprintf('=====================================================\n\n');

%% ============================================================
% STUDY A
% LFM BANDWIDTH
%
% Two targets separated by 1 m are used so the effect on
% range resolving capability is visible.
%% ============================================================

fprintf('Study A - LFM bandwidth\n');

bandwidthValues = [150e6 300e6 450e6];

fig = figure( ...
    'Name', ...
    'Bandwidth Study', ...
    'Position', ...
    [100 100 1500 420]);

tiledlayout(1, 3);

for i = 1:numel(bandwidthValues)

    cfg = cfgBase;

    cfg.bandwidth = bandwidthValues(i);

    cfg.slope = ...
        cfg.bandwidth / ...
        cfg.chirp_duration;

    % Remove static clutter for clean resolution comparison.
    cfg.clutter = [];

    cfg.noise_sigma = 0.01;

    % Two closely spaced targets.
    cfg.targets(1).range_m = 30;
    cfg.targets(1).velocity_mps = -5;
    cfg.targets(1).amplitude = 1;

    cfg.targets(2).range_m = 31;
    cfg.targets(2).velocity_mps = -5;
    cfg.targets(2).amplitude = 0.9;

    sim = simulate_lfmcw_frame(cfg);

    rd = process_simulated_range_doppler( ...
        sim.beat, ...
        cfg);

    % Collapse Doppler dimension to obtain range profile.
    rangeProfile = ...
        max(rd.power, [], 2);

    rangeProfileDb = ...
        10 * log10( ...
        (rangeProfile + eps) / ...
        (max(rangeProfile) + eps));

    nexttile;

    plot( ...
        rd.range_axis_m, ...
        rangeProfileDb, ...
        'LineWidth', ...
        1.2);

    grid on;

    xlim([25 36]);
    ylim([-50 0]);

    xlabel('Range (m)');
    ylabel('Normalized Power (dB)');

    theoreticalResolution = ...
        cfg.c / ...
        (2 * cfg.bandwidth);

    title(sprintf( ...
        'B = %.0f MHz, \\DeltaR = %.2f m', ...
        cfg.bandwidth / 1e6, ...
        theoreticalResolution));

end

saveas( ...
    fig, ...
    fullfile( ...
        resultDir, ...
        'bandwidth_study.png'));

close(fig);

fprintf('  Saved bandwidth_study.png\n');

%% ============================================================
% STUDY B
% LFM CHIRP DURATION
%
% Bandwidth remains constant.
% Changing chirp duration changes chirp slope and therefore
% the beat frequency produced by the same target range.
%% ============================================================

fprintf('Study B - LFM chirp duration\n');

durationValues = [30e-6 36e-6 43e-6];

fig = figure( ...
    'Name', ...
    'Chirp Duration Study', ...
    'Position', ...
    [100 100 1500 420]);

tiledlayout(1, 3);

for i = 1:numel(durationValues)

    cfg = cfgBase;

    cfg.chirp_duration = ...
        durationValues(i);

    % Keep the ADC rate fixed and sample only within this chirp ramp.
    % These durations give 750, 900 and 1075 samples at 25 MHz.
    cfg.num_samples = floor(cfg.sample_rate * cfg.chirp_duration);

    cfg.slope = ...
        cfg.bandwidth / ...
        cfg.chirp_duration;

    cfg.clutter = [];
    cfg.noise_sigma = 0;

    cfg.target.range_m = 30;
    cfg.target.velocity_mps = 0;
    cfg.target.amplitude = 1;

    sim = simulate_lfmcw_frame(cfg);

    x = sim.beat(:, 1);

    spectrum = abs( ...
        fft(x, cfg.num_samples));

    spectrum = ...
        spectrum(1:floor(cfg.num_samples/2));

    freqAxis = ...
        (0:numel(spectrum)-1) .* ...
        cfg.sample_rate / ...
        cfg.num_samples;

    spectrumDb = ...
        20 * log10( ...
        (spectrum + eps) / ...
        (max(spectrum) + eps));

    nexttile;

    plot( ...
        freqAxis / 1e6, ...
        spectrumDb, ...
        'LineWidth', ...
        1.2);

    grid on;

    xlabel('Beat Frequency (MHz)');
    ylabel('Normalized Magnitude (dB)');

    ylim([-60 0]);

    expectedBeat = ...
        cfg.slope * ...
        2 * cfg.target.range_m / ...
        cfg.c;

    title(sprintf( ...
        'T = %.0f us, f_b \\approx %.2f MHz', ...
        cfg.chirp_duration * 1e6, ...
        expectedBeat / 1e6));

end

saveas( ...
    fig, ...
    fullfile( ...
        resultDir, ...
        'chirp_duration_study.png'));

close(fig);

fprintf('  Saved chirp_duration_study.png\n');

%% ============================================================
% STUDY C
% CHIRP PERIOD / PRF
%
% Chirp period must remain longer than the 43 us ramp.
%% ============================================================

fprintf('Study C - Chirp period / PRF\n');

periodValues = [48e-6 60e-6 80e-6];

fig = figure( ...
    'Name', ...
    'PRF Study', ...
    'Position', ...
    [100 100 1500 420]);

tiledlayout(1, 3);

for i = 1:numel(periodValues)

    cfg = cfgBase;

    cfg.chirp_period = ...
        periodValues(i);

    cfg.target.range_m = 30;
    cfg.target.velocity_mps = -5;
    cfg.target.amplitude = 1;

    sim = simulate_lfmcw_frame(cfg);

    rd = process_simulated_range_doppler( ...
        sim.beat, ...
        cfg);

    rdDb = ...
        normalize_power_db( ...
        rd.power);

    nexttile;

    imagesc( ...
        rd.velocity_axis_mps, ...
        rd.range_axis_m, ...
        rdDb);

    axis xy;

    xlabel('Velocity (m/s)');
    ylabel('Range (m)');

    ylim([0 maxPlotRangeM]);

    caxis([-dynamicRangeDb 0]);

    prfKhz = ...
        1 / cfg.chirp_period / 1e3;

    vmax = ...
        cfg.lambda / ...
        (4 * cfg.chirp_period);

    title(sprintf( ...
        'T_r = %.0f us, PRF = %.1f kHz, V_{max}=%.1f', ...
        cfg.chirp_period * 1e6, ...
        prfKhz, ...
        vmax));

end

saveas( ...
    fig, ...
    fullfile( ...
        resultDir, ...
        'prf_study.png'));

close(fig);

fprintf('  Saved prf_study.png\n');

%% ============================================================
% STUDY D
% TARGET NUMBER
%% ============================================================

fprintf('Study D - Target number\n');

fig = figure( ...
    'Name', ...
    'Target Number Study', ...
    'Position', ...
    [100 100 1500 420]);

tiledlayout(1, 3);

for targetCount = 1:3

    cfg = cfgBase;

    cfg.clutter = [];

    targets = struct([]);

    targets(1).range_m = 20;
    targets(1).velocity_mps = -3;
    targets(1).amplitude = 1;

    if targetCount >= 2

        targets(2).range_m = 35;
        targets(2).velocity_mps = 4;
        targets(2).amplitude = 0.8;

    end

    if targetCount >= 3

        targets(3).range_m = 50;
        targets(3).velocity_mps = -8;
        targets(3).amplitude = 0.7;

    end

    cfg.targets = targets;

    sim = simulate_lfmcw_frame(cfg);

    rd = process_simulated_range_doppler( ...
        sim.beat, ...
        cfg);

    rdDb = ...
        normalize_power_db(rd.power);

    nexttile;

    imagesc( ...
        rd.velocity_axis_mps, ...
        rd.range_axis_m, ...
        rdDb);

    axis xy;

    xlabel('Velocity (m/s)');
    ylabel('Range (m)');

    ylim([0 maxPlotRangeM]);

    caxis([-dynamicRangeDb 0]);

    title(sprintf( ...
        '%d Target(s)', ...
        targetCount));

end

saveas( ...
    fig, ...
    fullfile( ...
        resultDir, ...
        'target_number_study.png'));

close(fig);

fprintf('  Saved target_number_study.png\n');

%% ============================================================
% STUDY E
% TARGET VELOCITY
%% ============================================================

fprintf('Study E - Target velocity\n');

velocityValues = [-2 -5 -10];

fig = figure( ...
    'Name', ...
    'Velocity Study', ...
    'Position', ...
    [100 100 1500 420]);

tiledlayout(1, 3);

for i = 1:numel(velocityValues)

    cfg = cfgBase;

    cfg.clutter = [];

    cfg.target.range_m = 30;
    cfg.target.velocity_mps = ...
        velocityValues(i);

    cfg.target.amplitude = 1;

    sim = simulate_lfmcw_frame(cfg);

    rd = process_simulated_range_doppler( ...
        sim.beat, ...
        cfg);

    rdDb = ...
        normalize_power_db(rd.power);

    nexttile;

    imagesc( ...
        rd.velocity_axis_mps, ...
        rd.range_axis_m, ...
        rdDb);

    axis xy;

    xlabel('Velocity (m/s)');
    ylabel('Range (m)');

    ylim([0 maxPlotRangeM]);

    caxis([-dynamicRangeDb 0]);

    title(sprintf( ...
        'Target v = %.1f m/s', ...
        velocityValues(i)));

end

saveas( ...
    fig, ...
    fullfile( ...
        resultDir, ...
        'target_velocity_study.png'));

close(fig);

fprintf('  Saved target_velocity_study.png\n');

%% ============================================================
% STUDY F
% MTI ORDER
%% ============================================================

fprintf('Study F - MTI order\n');

cfg = cfgBase;

cfg.target.range_m = 30;
cfg.target.velocity_mps = -5;
cfg.target.amplitude = 1;

sim = simulate_lfmcw_frame(cfg);

rdBefore = ...
    process_simulated_range_doppler( ...
        sim.beat, ...
        cfg);

rangeCube = ...
    rdBefore.range_fft;

numRangeBins = size(rangeCube, 1);
numChirps = size(rangeCube, 2);

rangeCube3D = reshape( ...
    rangeCube, ...
    numRangeBins, ...
    numChirps, ...
    1);

%% MTI2

opts = struct();

opts.method = 'MTI2';
opts.normalize_output_power = false;

[mti2Cube, ~] = ...
    suppress_clutter( ...
        rangeCube3D, ...
        opts);

mti2Cube = reshape( ...
    mti2Cube, ...
    numRangeBins, ...
    numChirps);

mti2Power = ...
    doppler_power_from_range_cube( ...
        mti2Cube);

%% MTI3

opts.method = 'MTI3';

[mti3Cube, ~] = ...
    suppress_clutter( ...
        rangeCube3D, ...
        opts);

mti3Cube = reshape( ...
    mti3Cube, ...
    numRangeBins, ...
    numChirps);

mti3Power = ...
    doppler_power_from_range_cube( ...
        mti3Cube);

%% Shared normalization

sharedReference = max([ ...
    rdBefore.power(:);
    mti2Power(:);
    mti3Power(:)]);

beforeDb = ...
    power_db_with_reference( ...
        rdBefore.power, ...
        sharedReference);

mti2Db = ...
    power_db_with_reference( ...
        mti2Power, ...
        sharedReference);

mti3Db = ...
    power_db_with_reference( ...
        mti3Power, ...
        sharedReference);

fig = figure( ...
    'Name', ...
    'MTI Study', ...
    'Position', ...
    [100 100 1500 420]);

tiledlayout(1, 3);

nexttile;

imagesc( ...
    rdBefore.velocity_axis_mps, ...
    rdBefore.range_axis_m, ...
    beforeDb);

axis xy;
xlabel('Velocity (m/s)');
ylabel('Range (m)');
ylim([0 maxPlotRangeM]);
caxis([-dynamicRangeDb 0]);
title('Before MTI');

nexttile;

imagesc( ...
    rdBefore.velocity_axis_mps, ...
    rdBefore.range_axis_m, ...
    mti2Db);

axis xy;
xlabel('Velocity (m/s)');
ylabel('Range (m)');
ylim([0 maxPlotRangeM]);
caxis([-dynamicRangeDb 0]);
title('MTI2');

nexttile;

imagesc( ...
    rdBefore.velocity_axis_mps, ...
    rdBefore.range_axis_m, ...
    mti3Db);

axis xy;
xlabel('Velocity (m/s)');
ylabel('Range (m)');
ylim([0 maxPlotRangeM]);
caxis([-dynamicRangeDb 0]);
title('MTI3');

saveas( ...
    fig, ...
    fullfile( ...
        resultDir, ...
        'mti_order_study.png'));

close(fig);

fprintf('  Saved mti_order_study.png\n');

%% ============================================================
% Done
%% ============================================================

fprintf('\n');
fprintf('=====================================================\n');
fprintf(' Parameter studies completed\n');
fprintf('=====================================================\n');

fprintf('Results:\n%s\n\n', resultDir);

%% ============================================================
% Local helper functions
%% ============================================================

function db = normalize_power_db(power)

reference = max(power(:));

db = 10 * log10( ...
    (power + eps) / ...
    (reference + eps));

end


function db = power_db_with_reference(power, reference)

db = 10 * log10( ...
    (power + eps) / ...
    (reference + eps));

end


function power = doppler_power_from_range_cube(rangeCube)

numRangeBins = size(rangeCube, 1);
numChirps = size(rangeCube, 2);

window = hann(numChirps).';

rangeCube = ...
    rangeCube .* ...
    repmat( ...
        window, ...
        numRangeBins, ...
        1);

rd = fftshift( ...
    fft( ...
        rangeCube, ...
        numChirps, ...
        2), ...
    2);

power = abs(rd).^2;

end
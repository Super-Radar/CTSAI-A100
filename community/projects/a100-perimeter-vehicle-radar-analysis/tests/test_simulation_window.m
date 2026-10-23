function test_simulation_window(projectRoot)
%TEST_SIMULATION_WINDOW Verify finite-ramp sampling and the duration study.
% Optional projectRoot permits running these checks against an older source.

if nargin < 1
    projectRoot = fileparts(fileparts(mfilename('fullpath')));
end
oldPath = path;
cleanup = onCleanup(@() path(oldPath));
addpath(fullfile(projectRoot, 'config'));
addpath(fullfile(projectRoot, 'functions', 'simulation'));
clear simulation_config simulate_lfmcw_frame;
base = simulation_config();

% The original duration study generated samples after both shorter ramps.
for duration = [30e-6, 36e-6]
    cfg = base;
    cfg.chirp_duration = duration;
    cfg.slope = cfg.bandwidth / duration;
    assert_error(@() simulate_lfmcw_frame(cfg), 'a100:SamplesOutsideChirp');
end

% Use exactly representable times to pin down the half-open boundary.
cfg = base;
cfg.num_chirps = 1;
cfg.num_samples = 8;
cfg.sample_rate = 8;
cfg.chirp_duration = 1;
cfg.clutter = [];
cfg.noise_sigma = 0;
cfg.target.range_m = 0;
cfg.target.velocity_mps = 0;
sim = simulate_lfmcw_frame(cfg);
assert(isequal(sim.fast_time_s, (0:7).' / 8));
assert(all(sim.beat(:) == 1));
cfg.num_samples = 9;
assert_error(@() simulate_lfmcw_frame(cfg), 'a100:SamplesOutsideChirp');
cfg.chirp_duration = 17 / 16;
sim = simulate_lfmcw_frame(cfg);
assert(sim.fast_time_s(end) == 1); % Ns/fs > T is nevertheless valid.
cfg.num_samples = 1;
cfg.chirp_duration = 1 / 16;
sim = simulate_lfmcw_frame(cfg);
assert(isequal(sim.fast_time_s, 0));

% Actual frame generation, not a second implementation of the simulator.
durations = [30e-6, 36e-6, 43e-6];
expectedSamples = [750, 900, 1075];
for i = 1:numel(durations)
    cfg = base;
    cfg.chirp_duration = durations(i);
    cfg.num_samples = floor(cfg.sample_rate * cfg.chirp_duration);
    cfg.slope = cfg.bandwidth / cfg.chirp_duration;
    cfg.clutter = [];
    cfg.noise_sigma = 0;
    cfg.target.velocity_mps = 0;
    sim = simulate_lfmcw_frame(cfg);
    assert(isequal(size(sim.beat), [expectedSamples(i), base.num_chirps]));
    assert(all(sim.fast_time_s >= 0 & sim.fast_time_s < durations(i)));
    x = sim.beat(:, 1);
    measuredHz = angle(sum(x(2:end) .* conj(x(1:end-1)))) * ...
        cfg.sample_rate / (2*pi);
    expectedHz = 2 * cfg.slope * cfg.target.range_m / cfg.c;
    assert(abs(measuredHz - expectedHz) < 1e-6);
    spectrum = abs(fft(x));
    [~, iPeak] = max(spectrum(1:floor(numel(x)/2)));
    peakHz = (iPeak - 1) * cfg.sample_rate / numel(x);
    assert(abs(peakHz - expectedHz) <= cfg.sample_rate / (2*numel(x)));
    fprintf('T=%.0f us: N=%d, last=%.8g us, beat=%.9g Hz\n', ...
        durations(i)*1e6, numel(x), sim.fast_time_s(end)*1e6, measuredHz);
end

% The existing 1024-point, 43-us default remains legal.
sim = simulate_lfmcw_frame(base);
assert(isequal(size(sim.beat), [1024, 256]));
assert(sim.fast_time_s(end) < base.chirp_duration);

% Non-finite values must not bypass the time-window contract.
for value = {0, -1, Inf, NaN, 1+1i, [1 2]}
    for field = {'sample_rate', 'chirp_duration', 'num_samples'}
        cfg = base;
        cfg.(field{1}) = value{1};
        assert_error(@() simulate_lfmcw_frame(cfg), '');
    end
end
cfg = base;
cfg.num_samples = 1.5;
assert_error(@() simulate_lfmcw_frame(cfg), '');
fprintf('PASS: finite-ramp sampling and duration-study regression.\n');
clear cleanup;
end

function assert_error(callback, expectedId)
try
    callback();
catch err
    if ~isempty(expectedId)
        assert(strcmp(err.identifier, expectedId), ...
            'Expected %s, received %s.', expectedId, err.identifier);
    end
    return;
end
error('test:ExpectedError', 'Expected the invalid sampling configuration to fail.');
end

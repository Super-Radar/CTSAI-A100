function sim = simulate_lfmcw_frame(cfg)
%SIMULATE_LFMCW_FRAME Generate one synthetic LFMCW beat-signal frame.
%
% Supports:
%   cfg.target   - single target
%   cfg.targets  - multiple targets
%
% Beat matrix:
%   [fast-time samples, slow-time chirps]

Ns = cfg.num_samples;
Nc = cfg.num_chirps;

fs = cfg.sample_rate;

% The analytic IF model samples one half-open ramp: 0 <= tFast < T.
% Reject extrapolated samples instead of silently extending the chirp.
validateattributes(Ns, {'numeric'}, ...
    {'real', 'finite', 'scalar', 'integer', 'positive'}, ...
    mfilename, 'cfg.num_samples');
validateattributes(fs, {'numeric'}, ...
    {'real', 'finite', 'scalar', 'positive'}, ...
    mfilename, 'cfg.sample_rate');
validateattributes(cfg.chirp_duration, {'numeric'}, ...
    {'real', 'finite', 'scalar', 'positive'}, ...
    mfilename, 'cfg.chirp_duration');
if (Ns - 1) / fs >= cfg.chirp_duration
    error('a100:SamplesOutsideChirp', ...
        ['Last ADC sample at %.12g s must be before the chirp end ' ...
         'at %.12g s. Adjust num_samples or sample_rate.'], ...
        (Ns - 1) / fs, cfg.chirp_duration);
end

c0 = cfg.c;
lambda = cfg.lambda;
slope = cfg.slope;

Tchirp = cfg.chirp_period;

%% Time axes

tFast = (0:Ns-1).' / fs;
tSlow = (0:Nc-1) * Tchirp;

%% Target list

if isfield(cfg, 'targets') && ~isempty(cfg.targets)

    targets = cfg.targets;

elseif isfield(cfg, 'target')

    targets = cfg.target;

else

    error('No simulation target is defined.');

end

%% Allocate beat signal

beat = complex(zeros(Ns, Nc));

%% ============================================================
% Moving targets
%% ============================================================

for iTarget = 1:numel(targets)

    R0 = targets(iTarget).range_m;
    v  = targets(iTarget).velocity_mps;
    A  = targets(iTarget).amplitude;

    fd = 2 * v / lambda;

    for m = 1:Nc

        % Target range at this slow-time position
        Rm = R0 + v * tSlow(m);

        tau = 2 * Rm / c0;

        fb = slope * tau;

        rangePhase = ...
            2 * pi * fb * tFast;

        dopplerPhase = ...
            2 * pi * fd * tSlow(m);

        beat(:, m) = beat(:, m) + ...
            A * exp( ...
            1j * (rangePhase + dopplerPhase));

    end

end

%% ============================================================
% Static clutter
%% ============================================================

if isfield(cfg, 'clutter') && ~isempty(cfg.clutter)

    for iClutter = 1:numel(cfg.clutter)

        R = cfg.clutter(iClutter).range_m;
        Aclutter = cfg.clutter(iClutter).amplitude;

        tau = 2 * R / c0;

        fb = slope * tau;

        rangePhase = ...
            2 * pi * fb * tFast;

        clutterSignal = ...
            Aclutter * exp(1j * rangePhase);

        beat = beat + ...
            repmat(clutterSignal, 1, Nc);

    end

end

%% ============================================================
% Complex Gaussian noise
%% ============================================================

if isfield(cfg, 'noise_sigma')
    noiseSigma = cfg.noise_sigma;
else
    noiseSigma = 0;
end

noise = ...
    noiseSigma / sqrt(2) .* ...
    (randn(Ns, Nc) + ...
     1j * randn(Ns, Nc));

beat = beat + noise;

%% Output

sim = struct();

sim.beat = beat;

sim.fast_time_s = tFast;
sim.slow_time_s = tSlow;

sim.target_ranges_m = ...
    [targets.range_m];

sim.target_velocities_mps = ...
    [targets.velocity_mps];

end
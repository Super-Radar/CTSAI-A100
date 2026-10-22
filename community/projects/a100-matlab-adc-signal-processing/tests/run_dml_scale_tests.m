function run_dml_scale_tests
% Native MATLAB/Octave regression, with no table/toolbox adapters.
root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root,'functions'));
grid = -60:0.25:60;
positions = {[0 .5 1 1.5], [0 1.1656 2.5796 3.2420]};
scales = [1e-200 1e-10 1e-8 1e-6 1 1e100 1e200];
phases = [0 .7 2.1];
comparisons = 0;
for geometry = 1:numel(positions)
    dictionary = steering_dictionary(positions{geometry}, grid);
    for truth = [-30 0 30]
        target = dictionary(:,find(grid==truth));
        % Noiseless and a fixed non-collinear perturbation; scale both the
        % signal and disturbance, preserving SNR and the physical direction.
        for noise = [0 .03]
            snapshot = target + noise*[1; -.4i; -.6; .3i];
            reference = estimate_angle_dml(snapshot, dictionary, grid);
            if noise == 0, assert(reference.angle_deg == truth); end
            for scale = scales
                for phase = phases
                    value = estimate_angle_dml(scale*exp(1i*phase)*snapshot, dictionary, grid);
                    assert(value.angle_deg == reference.angle_deg);
                    assert(all(~isnan(value.spectrum) & ~isinf(value.spectrum)));
                    % A published flat-spectrum rejection rule (issue #100)
                    % must still find this informative spectrum usable.
                    peak = max(value.spectrum);
                    assert(peak-min(value.spectrum) > 100*eps(peak));
                    if noise > 0
                        a = value.spectrum/max(value.spectrum);
                        b = reference.spectrum/max(reference.spectrum);
                        assert(max(abs(a-b)) < 1e-9);
                    end
                    comparisons = comparisons + 1;
                end
            end
        end
    end
end
dictionary = steering_dictionary(positions{1},grid);
% Reproduce the original absolute-eps failure with a known 30-degree source.
target = dictionary(:,find(grid==30));
fprintf('amplitude,original_angle_deg,normalized_angle_deg\n');
for amplitude = [1 1e-6 1e-8 1e-10]
    snapshot = amplitude*target;
    originalSpectrum = 1 ./ max(norm(snapshot)^2 - abs(dictionary'*snapshot).^2, eps);
    [~, peakIndex] = max(originalSpectrum);
    originalAngle = grid(peakIndex);
    fixed = estimate_angle_dml(snapshot,dictionary,grid);
    assert(fixed.angle_deg == 30);
    if amplitude <= 1e-8
        assert(originalAngle == grid(1));
        assert(max(originalSpectrum) == min(originalSpectrum));
    end
    fprintf('%.0e,%g,%g\n', amplitude,originalAngle,fixed.angle_deg);
end
zero = estimate_angle_dml(zeros(4,1), dictionary, grid);
assert(isnan(zero.angle_deg) && all(zero.spectrum==0));
assert_error(@() estimate_angle_dml([1;NaN;0;0],dictionary,grid));
assert_error(@() estimate_angle_dml([1;Inf;0;0],dictionary,grid));
assert_error(@() estimate_angle_dml(ones(3,1),dictionary,grid));
assert_error(@() estimate_angle_dml(ones(4,1),dictionary,grid(1:end-1)));
fprintf('PASS: %d amplitude/phase/geometry/SNR comparisons; zero and four invalid-input cases.\n', comparisons);
end

function assert_error(action)
failed = false;
try
    action();
catch exception
    failed = strcmp(exception.identifier,'a100:InvalidDmlInput');
end
assert(failed);
end

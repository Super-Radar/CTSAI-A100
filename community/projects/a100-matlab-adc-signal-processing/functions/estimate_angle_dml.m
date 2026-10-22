function result = estimate_angle_dml(snapshot, dictionary, angleGridDeg)
%ESTIMATE_ANGLE_DML Single-source deterministic maximum likelihood search.
% The dictionary is column-normalized by steering_dictionary. Normalize the
% snapshot before squaring: a fixed absolute residual floor otherwise erases
% directional information for small amplitudes, or overflows for large ones.
% The returned inverse relative-residual spectrum is dimensionless; absolute
% source power is not encoded in this angle diagnostic.
snapshot = double(snapshot(:));
if isempty(snapshot) || size(dictionary,1) ~= numel(snapshot) || ...
        size(dictionary,2) ~= numel(angleGridDeg) || isempty(angleGridDeg) || ...
        any(isnan(snapshot) | isinf(snapshot)) || ...
        any(isnan(dictionary(:)) | isinf(dictionary(:))) || ...
        any(isnan(angleGridDeg(:)) | isinf(angleGridDeg(:)))
    error('a100:InvalidDmlInput', 'Expected finite, dimensionally consistent DML inputs.');
end
result = struct('angle_deg', NaN, 'spectrum', zeros(1,numel(angleGridDeg)));
% Scale real and imaginary components separately to avoid overflow in abs(z).
scale = max([abs(real(snapshot)); abs(imag(snapshot))]);
if scale == 0
    return; % No directional information; normalization is undefined.
end
snapshot = snapshot / scale;
snapshot = snapshot / norm(snapshot);
projectionPower = abs(dictionary' * snapshot).^2;
residualPower = max(1 - projectionPower, eps);
spectrum = 1 ./ residualPower;
[~, iPeak] = max(spectrum);
result.angle_deg = angleGridDeg(iPeak);
result.spectrum = spectrum.';
end

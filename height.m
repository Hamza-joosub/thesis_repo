function h = height(x)
% Octave shim for MATLAB's height(): number of rows. Provided so Matthews's
% es_probs.m runs UNMODIFIED — the function under test is untouched.
h = size(x, 1);
end

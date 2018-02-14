    function [ FC_pred, corr ] = testing_MKL( sCall, fCall, num_scls, pi, epsilon, idx_lam, exp_values )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

% variables
[~, n, num_subjs] = size(sCall);
% K = zeros(n, m * n);
FC_pred = zeros(n, n, num_subjs);
corr = zeros(1, num_subjs);
error = zeros(1, num_subjs);
% for each sample
for l = 1 : num_subjs
    % pre-process
    [MapC, inds] = pre_process(sCall(:,:,l), fCall(:,:,l));
    
    % thresholding SC
    MapC = (MapC > epsilon * max(MapC(:))) .* MapC;
    
    % set of heat kernels
    if (~exist( 'exp_values', 'var'))
        K = Kernels_version3(MapC, num_scls, idx_lam);
    else 
        K = Kernels_version3(MapC, num_scls, idx_lam, exp_values);
    end
    
    % prediction
%     FC_pred(:, :, l) = (K * pi);
%     FC_pred(:, :, l) = (FC_pred(:, :, l) + FC_pred(:, :, l)') / 2;
    for i = 1 : num_scls
        fc_pred = K(:, 1 + (i - 1) * n : (i) * n) * pi(1 + (i - 1) * n : (i) * n, :);
        fc_pred = (fc_pred + fc_pred') / 2;
        FC_pred(:, :, l) = FC_pred(:, :, l) + fc_pred;
    end
    
    % correlation
    c = corrcoef((FC_pred(:, :, l) .* inds), (fc_range(fCall(:, :, l), '') .* inds));
    corr(1,l) = c(1,2);
    % mse
    error(1, l) = sqrt(sum(sum((FC_pred(:, :, l) .* inds - fc_range(fCall(:, :, l), '') .* inds).^(2)))) / (n * n);
end

end


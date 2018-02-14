function [ pi ] = training_version3( sCall, fCall, m, epsilon, idx_lam, exp_values  )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

% initializating variables
L = size(fCall,3); % number of samples
n = size(fCall,1); % number of ROIs
pi = zeros(m * n, n);
Psi = zeros(L * n, m * n);
Phi = zeros(L * n, n);
% one_vec = ones(n, 1);
%-------------------------------
% matrix formation
%-------------------------------
for l = 1 : L % l == subj_idx
    % pre-processing
    [MapC, inds] = pre_process(sCall(:,:,l), fCall(:,:,l));
    
    % thresholding SC
    MapC = (MapC > epsilon * max(MapC(:))) .* MapC;
    
    % set of heat kernels
    if (~exist( 'exp_values', 'var'))
        K = Kernels_version3(MapC, m, idx_lam);
    else
        K = Kernels_version3(MapC, m, idx_lam, exp_values);
    end
    % Psi and Phi formation
    
    Psi((l - 1) * n + 1 : l * n, :) = K;
    Phi((l - 1) * n + 1 : l * n, :) = fc_range(fCall(:,:,l), ' ') .* inds; % fc_range(fCall(:,:,l), 'normalized') .* inds;
end

%--------------------------------
% training
%--------------------------------
% [u, s, v] = svd(Psi);
% s = s .* (s > 0.5);
% Psi = u * s * v';
% % pseudo inverse of Psi
% [u, s, v] = svd(Psi);
% s = s .* (s > 0.2);
% [rows, cols] = find(s > 0);
% inv_s = [[inv(s(rows, cols)), zeros(max(rows), L * n - max(rows))]; zeros(m * n - max(rows), L * n)];
% Psi_inv = v * inv_s * u';
%
% % find pi
% pi = Psi_inv * Phi;

% finding pi for each node
for j = 1 : n
    %     % Quadratic programming
    %     [pi(:, j), fval, exitflag, o_p, lambda] = quadprog(Psi' * Psi, -2 * Phi(:, j)' * Psi, [], [], [], [], double(zeros(m * n, 1)), double(ones(m * n, 1)));
    
    % lasso
    [x, fitinfo] = lasso(Psi, squeeze(Phi(:, j)), 'Alpha', 0.001);
    [~, idx_min] = min(fitinfo.MSE);
    pi(:, j) = x(:, idx_min);
    j
    % pseudo inverse
    % pi(:, j) = (Psi' * Psi) \ (Psi' * Phi(:, j));
end

% % post processing
% for i = 1 : m
%     p = pi(1 + (i - 1) * n : (i) * n, :);
%     p = (p + p') / 2;
%     pi(1 + (i - 1) * n : (i) * n, :) = p;
% end
end


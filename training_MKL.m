function [ pi ] = training_MKL( sCall, fCall, m, epsilon, idx_lam, exp_values  )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

% initializating variables
L = size(fCall,3); % number of samples
n = size(fCall,1); % number of ROIs
pi = zeros(m * n, n); % model parameters, \Pi
Psi = zeros(L * n, m * n); % this matrix comes while optimization formulation
Phi = zeros(L * n, n); % this matrix comes while optimization formulation

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
        K = Kernels(MapC, m, idx_lam);
    else
        K = Kernels(MapC, m, idx_lam, exp_values);
    end
    
    % Psi and Phi formation
    Psi((l - 1) * n + 1 : l * n, :) = K;
    Phi((l - 1) * n + 1 : l * n, :) = fc_range(fCall(:,:,l), ' ') .* inds; 
end

%--------------------------------
% training
%--------------------------------

% finding pi for each node
for j = 1 : n   
    % lasso
    [x, fitinfo] = lasso(Psi, squeeze(Phi(:, j)), 'Alpha', 0.001);
    [~, idx_min] = min(fitinfo.MSE);
    pi(:, j) = x(:, idx_min);
end
end


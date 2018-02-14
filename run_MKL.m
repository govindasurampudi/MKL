%% load the data
data_pth = '.';
load([data_pth, '/sc_all.mat']);
load([data_pth, '/fc_all.mat']);

%% parameters
[~, num_rois, num_subjs] = size(sCall); % or fCall, check for both.
epsilon = 0.0; % value for thresholding SC matrices. preferrably keep it 0.
num_scls = 16; % number of scales of the MKL model.
idx_lam = ceil(num_rois * 1 / 40); % index of eigenvalue on which subject-specific scales are found.
exp_values = linspace(0.01, 0.95, num_scls); % the normalized scale values - between [0, 1].

random_indices = randperm(num_subjs);
train_idx = random_indices(1 : 23);
test_idx  = random_indices(24 : 46);

%% training and testing the model
MKL = struct;
MKL.pi = training_MKL(sCall(:, :, train_idx), fCall(:, :, train_idx), num_scls, epsilon, idx_lam, exp_values);
[MKL.FC_pred, MKL.corr] = testing_MKL(sCall(:, :, test_idx), fCall(:, :, test_idx), num_scls, MKL.pi, epsilon, idx_lam, exp_values);
FC_pred_mean = mean(MKL.FC_pred, 3);
MKL.FC_pred_mean = FC_pred_mean;
MKL.random_indices = random_indices;
MKL.train_idx = train_idx;
MKL.test_idx = test_idx;

%% save the model
save('MKL_final', 'MKL')

%% leave-one-out cross validation
%{
% leave-one-out => 'loo'
MKL_loo_cell = cell(1, size(test_idx, 2));
for validn_subj_idx = 1 : 23
    M = struct;
    M.pi = training_MKL(...
        sCall(:, :, setdiff(random_indices, test_idx(validn_subj_idx))),...
        fCall(:, :, setdiff(random_indices, test_idx(validn_subj_idx))),...
        num_scls, epsilon, idx_lam, exp_values);
    [M.FC_pred, M.corr] = testing_MKL(...
        sCall(:, :, test_idx(validn_subj_idx)),...
        fCall(:, :, test_idx(validn_subj_idx)),...
        num_scls, M.pi, epsilon, idx_lam, exp_values);
    MKL_loo_cell{1, validn_subj_idx} = M;
end
% save
save('MKL_leave_one_out.mat', 'MKL_loo_cell')
%}


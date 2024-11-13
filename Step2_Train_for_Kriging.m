% Case 4.1
%---------------------------------Process parameter domain discretization-----------------------------------------
% In accordance with FDM (ss_step, ap_step) mesh
ap_step  = 30;  % Number of discrete units per cutting depth
ss_step  = 40; % Number of discrete units per spindle speed

ap_start = 0e-3;  % (m)
ap_end   = 3e-3;  % (m)
ss_start = 4.5e3;   % (rpm)
ss_end   = 8.5e3;  % (rpm)

ss_discretization = (ss_end - ss_start) / ss_step;
ap_discretization = (ap_end - ap_start) / ap_step;

% Training Peparation
SS = zeros(ss_step + 1, ap_step+1);
AP = zeros(ss_step + 1, ap_step+1);
for i = 1 : (ss_step + 1)
    ss = ss_start + (i - 1) * ss_discretization;
    for j = 1 : (ap_step + 1)
        ap = ap_start + (j - 1) * ap_discretization;
        SS(i, j) = ss;
        AP(i, j) = ap;
    end
end
discretization_points = zeros(2, (ss_step + 1) * (ap_step + 1));
discretization_points(1, :) = reshape(SS', 1, []);
discretization_points(2, :) = reshape(AP', 1, []);

% Model Training
filename = 'LH';
folderPath = fullfile('.', 'Data_Generated', ['KrigingModel_', filename]);
if ~exist(folderPath, 'dir')
    mkdir(folderPath);
end

kriging_num = size(discretization_points, 2);

% Model Selection
covFcn = @corrgauss;
regrFcn = @regpoly0;

for i = 1 : kriging_num
    tic;
    ss = discretization_points(1, i);
    ap = discretization_points(2, i);
    num_ss = (ss - ss_start) / ss_discretization;
    num_ap = round((ap - ap_start) / ap_discretization);
    num = num_ss * (ap_step+1) + num_ap; 
    % File name check
    traingdata = cell2mat(struct2cell(load(sprintf('%s%s%s%d%s', '.\Data_Generated\TrainingData_', filename, '\KrigingData', num, '.mat'))));
    traingdata(:, 1:2) = traingdata(:, 1:2) / 1000;
    S_total = traingdata(:, 1:4); % model input X
    Y_total = traingdata(:, 11); % model input Y
    
    % Kriging paras setting
    theta = [1 1 0.1 0.1]; lob = [1E-2 1E-2 1E-2 1E-2]; upb = [5 5 5 5];
    [dmodel, perf] = dacefit(S_total, Y_total, regrFcn, covFcn, theta, lob, upb);
    save(sprintf('%s%s%s%s','.\Data_Generated\KrigingModel_', filename, '\kmodel', num2str(num)));
    elapsed_time = toc;
    fprintf('current round %d in total %d with time consume %d (s)\n', i, kriging_num, toc);
end
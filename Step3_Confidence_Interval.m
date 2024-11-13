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

% OMA result
coeff_wnx = [-1.17986e-05, 0.0943074, 1620.62];
coeff_wny = [-1.04515e-05, 0.10178, 1751.88];
coeff_zetax = [-4.02625e-08, 0.000741234, -0.0104724];
coeff_zetay = [-2.04545e-09, 8.94091e-05, 1.863];
coeff = [coeff_wnx; coeff_wny; coeff_zetax; coeff_zetay];

ss_axis = linspace(ss_start, ss_end, ss_step + 1);
OMA_w = zeros(4, ss_step + 1);
for i = 1 : size(OMA_w, 1)
    for j = 1 : size(OMA_w, 2)
        OMA_w(i, j) = quadraFunc(ss_axis(j), coeff(i, : ));
    end
end

% Training preparation
kriging_num = size(discretization_points, 2);
lambda = zeros(ss_step + 1, ap_step + 1);
standard = zeros(ss_step + 1, ap_step + 1);
spectral_radius = zeros(ss_step + 1, ap_step + 1);

%------------------------------------------------Kriging prediction------------------------------------------------
% filename = 'LH';
filename = 'sobol';

% confidence interval para loading
samplenum = 500;
pred_X = [1.9565, 1.9545, 0.0227, 0.0228, 9.44, 10.18, 11.31, 4.10]';
dim_krig = 4;
if dim_krig == 4
    pred_X = pred_X(1:4);
end

X_0 = (pred_X - mean(pred_X))/std(pred_X);

for i = 1 : kriging_num
    tic;
    ss = discretization_points(1, i);
    ap = discretization_points(2, i);
    num_ss = (ss - ss_start) / ss_discretization;
    num_ap = round((ap - ap_start) / ap_discretization);
    num = num_ss * (ap_step+1) + num_ap;
    %------------------------------------------------------------------------------------------------
    % data for confidence interval
    traingdata = cell2mat(struct2cell(load(sprintf('%s%s%s%d%s', '.\Data_Generated\TrainingData_', filename, '\KrigingData', num, '.mat'))));
    Y_total = traingdata(:,11);
    
    load(sprintf('%s%s%s%s','.\Data_Generated\KrigingModel_', filename,'\kmodel', num2str(num)));
    S = dmodel.S;
    [mY, sY] = deal(dmodel.Ysc(1), dmodel.Ysc(2));
    Y = (Y_total - mY)/sY;
    
    beta = dmodel.beta;
    corrFcn = dmodel.corr;
    regrFcn = dmodel.regr;
    F = regrFcn(S);
    Y_pred = F *beta;
    
    STD = pred_std_calcu(S, Y, Y_pred, X_0);
    %------------------------------------------------------------------------------------------------
    X = OMA_w( : , num_ss + 1)';
    Y_0 = predictor(X, dmodel);
    lambda(num_ss + 1, num_ap + 1) = Y_0;
    standard(num_ss + 1, num_ap + 1) = STD;
    elapsed_time = toc;
    fprintf('current round %d in total %d with time consume %d (s)\n', i, kriging_num, toc);
end

t = 1;
[c1,h1] = contour(SS, AP, lambda, [1,1], 'k', 'DisplayName', 'Kriging');
hold on;
[c2,h2] = contour(SS, AP, lambda + t * standard, [1,1], 'r', 'DisplayName', '1 STD deviation');
hold on
[c3,h3] = contour(SS, AP, lambda - t * standard, [1,1], 'b', 'DisplayName', '1 STD deviation');
hold on
xlabel('spindle speed(rpm)');
ylabel('depth of cut(m)');
hold on;
legend('show');


folderPath = '.\Data_Essential';
fileName1 = 'SS_Kriging.mat';
fileName2 = 'AP_Kriging.mat';
fileName3 = 'EI_Kriging.mat';
fileName4 = 'STD_Kriging.mat';
save(fullfile(folderPath, fileName1), 'SS');
save(fullfile(folderPath, fileName2), 'AP');
save(fullfile(folderPath, fileName3), 'lambda');
save(fullfile(folderPath, fileName4), 'standard');

function y = quadraFunc(x, coeff)
    y = coeff(1)*x^2 + coeff(2)*x + coeff(3);
end

% STD = pred_std_calcu(S, Y, Y_pred, X_0);
function standard = pred_std_calcu(X, Y_true, Y_pred, X_0)
    % X: m * n£»m ×é n Î¬input
    % Y_true, Y_pred : m * 1£»m ×é 1 Î¬output
    % X_0 : n * 1£»n Î¬´ýÔ¤²âµã
    m = size(X, 1);
    n = size(X, 2);
    delta = (Y_true - Y_pred)' * (Y_true - Y_pred)/(m - n);
    cov_beta = delta * inv(X' * X);
    var = X_0' * cov_beta * X_0;
    standard = sqrt(var);
end


%---------------------------------data loading-----------------------------------------
filename = 'FRF_x.xlsx';
% filename = 'FRF_y.xlsx';
currentFilePath = mfilename('fullpath');
[currentDir, ~, ~] = fileparts(currentFilePath);
parentDir = fileparts(currentDir);
filePath = fullfile(parentDir, 'Data_Experiment', filename);

[num, txt, raw] = xlsread(filePath,4);
freq_total = num(1, : );
real_total = num(2, : );
img_total = num(3, : );

freq = freq_total(2000 : 6000);
real = real_total(2000 : 6000);
img = img_total(2000 : 6000);


signal_length = length(freq);
real_disp = zeros(1,signal_length);
img_disp = zeros(1,signal_length);

for i = 1 : signal_length
    real_disp(i) = real(i)/  (2 * pi * freq(i))^2;
    img_disp(i) = img(i)/ (2 * pi * freq(i))^2;
end
%---------------------------------FRF plotting-----------------------------------------
figure('Position', [100, 100, 1000, 600]);
subplot(2, 2, 1);
plot(freq, real);
title('Real FRF of Acc');
xlabel('Frequency (Hz)');
ylabel('Real (m/s^2/N)');


subplot(2, 2, 2);
plot(freq, img);
title('Img FRF of Acc');
xlabel('Frequency (Hz)');
ylabel('Img (m/s^2/N)');

subplot(2, 2, 3);
plot(freq, real_disp);
title('Real FRF of Disp');
xlabel('Frequency (Hz)');
ylabel('Real (m/N)');

subplot(2, 2, 4);
plot(freq, img_disp);
title('Img FRF of Disp');
xlabel('Frequency (Hz)');
ylabel('Img (m/N)');

%---------------------------------FRF calcu-----------------------------------------
% Modal paras can be extracted based on the peaks in FRF of Acc
% Peak Picking Method

wx_low = 1912;
wx_high = 2001;
img_frf_x = 2.329e-6; %1962

wy_low = 1910;
wy_high = 1999;
img_frf_y = 2.158e-6; %1957

wx_low_2 = 4385;
wx_high_2 = 4506;
img_frf_x_2 = 182.7/ (2 * pi * 4457)^2; % 2.3297e-07

wy_low_2 = 4391;
wy_high_2 = 4466;
img_frf_y_2 = 112/ (2 * pi * 4466)^2; % 1.422397e-07

[wn_x, zeta_x, kai_x] = Modal_Calcu(wx_low, wx_high, img_frf_x);
[wn_y, zeta_y, kai_y] = Modal_Calcu(wy_low, wy_high, img_frf_y);

[wn_x_2, zeta_x_2, kai_x_2] = Modal_Calcu(wx_low_2, wx_high_2, img_frf_x_2);
[wn_y_2, zeta_y_2, kai_y_2] = Modal_Calcu(wx_low_2, wx_high_2, img_frf_x_2);

% fprintf('Peak 1\n')
fprintf('x direction: Natural Freq = %d (Hz), Damping ratio = %d (1), Stiffness = %d (N/m)\n', wn_x, zeta_x, kai_x);
fprintf('y direction: Natural Freq = %d (Hz), Damping ratio = %d (1), Stiffness = %d (N/m)\n', wn_y, zeta_y, kai_y);

% Altintas MANUFACTURING AUTOMATION
% Section 3.5/3.6/3.7

function [wn, zeta, kai] = Modal_Calcu(w_low, w_high, img_frf)
    wn = (w_low + w_high) / 2; % Natural Freq
    zeta = (w_high - w_low) / (2 * wn); % Damping ratio
    kai = 1 / (2 * zeta *  img_frf); % Stiffness
end
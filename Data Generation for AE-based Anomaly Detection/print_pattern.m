%% print_pattern.m
% Visualization of attack vectors.

function print_pattern(alpha_pattern)

a = alpha_pattern;

% Since measurements are expressed in per unit values, a threshold of 1e-4 is set such that any
% attack vector element attaining a value below such threshold is perceived as 0.

a(find(abs(a)<1e-4))=0;
a_l = (abs(a) >= 1e-4);
a_c = sum(a_l,2);
a_pool = alpha_pattern;
a_pool_l_init = (abs(a_pool) >= 1e-4);
a_pool_l = (abs(a_pool) >= 1e-4);
a_pool_c = sum(a_pool_l,2);
a_pool(find(abs(a_pool)<1e-4))=0;

% Since measurements are expressed in Cartesian coordinates, these coordinates are subsequently 
% combined in a logical way for each measurement in order to deduce the sparsity pattern of attack 
% vectors.

A = a_pool_l(1:2:end,:);
B = a_pool_l(2:2:end,:);

C = A | B;

figure; spy(C, 20); pbaspect([1 1 1]);

yt={'I_2_-_3';'I_2_-_4';'I_2_-_5';'I_2_-_1';'I_7_-_8';'I_7_-_9';'I_7_-_4';'I_1_1_-_6';'I_1_1_-_1_0';'I_1_3_-_1_4';'I_1_3_-_6';'I_1_3_-_1_2';'V_2';'V_7';'V_1_1';'V_1_3'}; 
set(gca,'ytick',1:1:size(C,1)); 
set(gca,'yticklabel',yt);
% set(gca,'yticklabel',[]);
set(gca,'xtick',1:1:size(C,2)); 
set(gca,'xticklabel',[]);
set(gca,'XTickLabelRotation',90);
grid on;
set(gca, 'GridLineStyle', ':');
set(gca, 'GridAlpha', 0.8);
%set(gca, 'LineWidth', 1);
set(gca, 'GridColor', 'black');
xlabel('sparse attack vectors');
title('Combinations of measurements in IEEE 14-bus system');
% title('Combinations of measurements in IEEE 30-bus system');

end
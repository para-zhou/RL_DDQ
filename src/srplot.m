x = res.x;
% 0-4: 1
% 5-9: 10
% 10-14: 20

% figure
% plot(x, DDQ_2, x, DDQ_5, x, DDQ_10, x, DDQ_20, x, DDQ_40);


ft = fittype( 'smoothingspline' );
opts = fitoptions( 'Method', 'SmoothingSpline' );
opts.SmoothingParam = 0.1;

figure( 'Name', 'Average turns' );

[xData, yData] = prepareCurveData( x, DDQ_40 );
[fitresult1, ~] = fit( xData, yData, ft, opts );
h1 = plot( fitresult1 );
h1.Color = [0.8500, 0.3250, 0.0980];
h1.LineWidth = 1;
hold on

% [xData, yData] = prepareCurveData( x, DDQ_20 );
% [fitresult2, ~] = fit( xData, yData, ft, opts );
% h2 = plot( fitresult2 );
% h2.Color = [0.6350, 0.0780, 0.1840];
% h2.LineWidth = 1;
% hold on

[xData, yData] = prepareCurveData( x, DDQ_10 );
[fitresult3, ~] = fit( xData, yData, ft, opts );
h3 = plot( fitresult3 );
h3.Color = [0, 0.4470, 0.7410];
h3.LineWidth = 1;
hold on

% [xData, yData] = prepareCurveData( x, DDQ_5 );
% [fitresult4, ~] = fit( xData, yData, ft, opts );
% h4 = plot( fitresult4 );
% h4.Color = [0.3010, 0.7450, 0.9330];
% h4.LineWidth = 1;
% hold on

[xData, yData] = prepareCurveData( x, DDQ_2 );
[fitresult5, ~] = fit( xData, yData, ft, opts );
h5 = plot( fitresult5 );
h5.Color = [0.4660, 0.6740, 0.1880];
h5.LineWidth = 1;
hold on


legend( 'DDQ(40)', 'DDQ(10)', 'DDQ(2)', 'Location', 'southEast' );
xlabel('Epoch')
ylabel('Average turns')
title('Average turns')
grid on

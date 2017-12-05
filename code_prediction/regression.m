clc;
clear;
close all;
temperature=[28,30];
humidity=[0.66,0.58];
% “ª¥Œ≤Â÷µ
datam=xlsread('dataset_for_simulation.xlsx');
Time0=[538,600];
Time=datam(:,3);
temperature1=interp1(Time0,temperature,Time);
Time0=[538,600];
Time=datam(:,3);
humidity1=interp1(Time0,humidity,Time);
subplot(2,1,1)
scatter(Time,temperature1,'r');
xlabel('Time [s]','Fontsize',10);
ylabel('Temperature [^oC]','Fontsize',10)
subplot(2,1,2)
scatter(Time,humidity1,'b');
xlabel('Time [s]','Fontsize',10);
ylabel('Humidity [%]','Fontsize',10)

Wind=[4,5];
Wind_mean=mean(Wind,2);
Wind_std=std(Wind,0,2);
%% Geneate Wind Speed randomly
Max_wind_mean=max(Wind_mean);
Max_Wind_std=max(Wind_std);
    for j=1:1:length(datam(:,10))
        k=(Wind_std/Wind_mean)^(-1.088);%%%%%%%%%%%-1.088
        c=Wind_mean/gamma(1+(1/k));
        uu(j)=wblrnd(c,k);
    end
U=uu';
Time=datam(:,3);
figure;
scatter(Time,uu','r');
xlabel('Time [s]','Fontsize',10);
ylabel('Wind Speed [m/s]','Fontsize',10)

%plot
figure
subplot(2,2,1)
scatter(datam(:,8),datam(:,13),'r');
xlabel('Temperature [^oC]','Fontsize',10);
ylabel('Velocity [km/h]','Fontsize',10);

subplot(2,2,2)
scatter(datam(:,9),datam(:,13),'b');%scatter
xlabel('Humidity [%]','Fontsize',10);
ylabel('Velocity [km/h]','Fontsize',10);

subplot(2,2,3)
scatter(datam(:,10),datam(:,13),'k');
xlabel('Wind Speed [m/s]','Fontsize',10);
ylabel('Velocity [km/h]','Fontsize',10);

subplot(2,2,4)
scatter(datam(:,12),datam(:,13),'g');
xlabel('Curvature [^oC]','Fontsize',10);
ylabel('Velocity [km/h]','Fontsize',10);




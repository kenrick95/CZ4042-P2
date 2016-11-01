load 'dataTeststore.mat';

lrs=[1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6];
mbss=[50,100,150,200,250,300,350,400,450,500];
l2rs=[1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6];
mmts=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95];

acc=zeros(numel(lrs),1);
for i=1:numel(lrs)
    lr=lrs(i);
    filename=sprintf('lrs-%d.mat', i);
    load(filename);
    YTest = classify(convnet, dataTeststore);
    TTest = dataTeststore.Labels;
    acc(i) = sum(YTest == TTest)/numel(YTest);
end
bar(acc);
xlabel('Learning rate', 'fontsize', 14);
ylabel('Accuracy', 'fontsize', 14);
set(gca, 'fontsize', 14);
xlim([0 numel(lrs)+1]);
xticks(1:numel(lrs));
xticklabels(arrayfun(@(n)(num2str(n,'%.1e')), lrs, 'UniformOutput', false));
disp('Press any key to continue.');
pause;

acc=zeros(numel(mbss),1);
for i=1:numel(mbss)
    mbs=mbss(i);
    filename=sprintf('mbss-%d.mat', i);
    load(filename);
    YTest = classify(convnet, dataTeststore);
    TTest = dataTeststore.Labels;
    acc(i) = sum(YTest == TTest)/numel(YTest);
end
plot(mbss, acc, 'o-');
xlabel('Mini-batch size', 'fontsize', 14);
ylabel('Accuracy', 'fontsize', 14);
set(gca, 'fontsize', 14);
disp('Press any key to continue.');
pause;

acc=zeros(numel(l2rs),1);
for i=1:numel(l2rs)
    l2r=l2rs(i);
    filename=sprintf('l2rs-%d.mat', i);
    load(filename);
    YTest = classify(convnet, dataTeststore);
    TTest = dataTeststore.Labels;
    acc(i) = sum(YTest == TTest)/numel(YTest);
end
bar(acc);
xlabel('L2 regularization', 'fontsize', 14);
ylabel('Accuracy', 'fontsize', 14);
set(gca, 'fontsize', 14);
ymin=0.86;
ymax=0.89;
ylim([ymin ymax]);
xlim([0 numel(l2rs)+1]);
xticks(1:numel(l2rs));
xticklabels(arrayfun(@(n)(num2str(n,'%.1e')), l2rs, 'UniformOutput', false));
disp('Press any key to continue.');
pause;

acc=zeros(numel(mmts),1);
for i=1:numel(mmts)
    mmt=mmts(i);
    filename=sprintf('mmts-%d.mat', i);
    load(filename);
    YTest = classify(convnet, dataTeststore);
    TTest = dataTeststore.Labels;
    acc(i) = sum(YTest == TTest)/numel(YTest);
end
plot(mmts, acc, 'o-');
xlabel('Momentum', 'fontsize', 14);
ylabel('Accuracy', 'fontsize', 14);
set(gca, 'fontsize', 14);

load 'dataTeststore.mat';

num_layers=20:10:70;
acc=zeros(numel(num_layers),1);
for i=1:numel(num_layers)
    filename=sprintf('conv1size-%d.mat',num_layers(i));
    load(filename);
    YTest = classify(convnet, dataTeststore);
    TTest = dataTeststore.Labels;
    acc(i) = sum(YTest == TTest)/numel(YTest);
end
plot(num_layers,acc,'-o');
xlabel('Conv_layer1 size', 'fontsize', 14, 'Interpreter', 'None');
ylabel('Accuracy', 'fontsize', 14);
set(gca, 'fontsize', 14);
xticks(20:10:70);
disp('Press any key to continue.');
pause;

num_layers=20:10:70;
acc=zeros(numel(num_layers),1);
for i=1:numel(num_layers)
    filename=sprintf('conv2size-%d.mat',num_layers(i));
    load(filename);
    YTest = classify(convnet, dataTeststore);
    TTest = dataTeststore.Labels;
    acc(i) = sum(YTest == TTest)/numel(YTest);
end
plot(num_layers,acc,'-o');
xlabel('Conv_layer2 size', 'fontsize', 14, 'Interpreter', 'None');
ylabel('Accuracy', 'fontsize', 14);
set(gca, 'fontsize', 14);
xticks(20:10:70);

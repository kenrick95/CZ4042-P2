[x,t] = house_dataset;
net1 = feedforwardnet(10);
net2 = train(net1,x,t);
y = net2(x);
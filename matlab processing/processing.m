A = xlsread('KNNoptimization_IEEE14.xlsx');

k = A(1:200,2);
acc = A(1:200,3);

m = max(acc)
t = k(acc==m)

figure(1)
plot(k,acc)
xlabel('Number of Neighbors (K)')
ylabel('Test Accuracy')
hold on
plot(t,m, '*')
print('knnOpt_IEEE14', '-dpng');

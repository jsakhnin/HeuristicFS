A = xlsread('ANNoptimization_IEEE14.xlsx');

k = A(:,2);
k = log(k);
acc = A(:,3);

m = max(acc)
t = k(acc==m)

figure(1)
plot(k,acc)
xlabel('Log (\alpha)')
ylabel('Test Accuracy')
hold on
grid on
ylim([0,1]);
plot(t,m, '*')
print('annOpt_IEEE14', '-dpng');
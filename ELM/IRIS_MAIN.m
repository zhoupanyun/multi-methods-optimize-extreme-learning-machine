%% 清空环境变量 elm
clc
clear
close all
format compact
%% 读取数据
load iris_data;%3分类
input=features;
output=classes;
%% 随机选择测试集与训练集 7:3划分
[~,n]=sort(rand(1,size(input,1)));
m=round(size(input,1)*0.7);

input_train=input(n(1:m),:)';
input_test=input(n(m+1:end),:)';
label_train=output(n(1:m),:)';
label_test=output(n(m+1:end),:)';

%输入数据归一化
[inputn_train,inputps]=mapminmax(input_train);
[inputn_test,inputtestps]=mapminmax('apply',input_test,inputps);
%% ELM参数设置
inputnum=size(input_train,1);%输入层节点
hiddennum=5;%隐含层节点
activation='sin';%激活函数sin,sig  hardlim
TYPE=1;%1-分类 0-回归
%% 没有优化的ELM
[IW,B,LW,TF,TYPE] = elmtrain(inputn_train,label_train,hiddennum,activation,TYPE);
%% ELM仿真测试
Tn_sim = elmpredict(inputn_test,IW,B,LW,TF,TYPE);
test_accuracy=(sum(label_test==Tn_sim))/length(label_test)
stem(label_test,'*')
hold on
plot(Tn_sim,'p')
title('没有优化的ELM')
legend('期望输出','实际输出')
xlabel('样本数')
ylabel('类别标签')
%% 节点个数

[bestchrom,trace]=gaforelm(inputnum,hiddennum,TYPE,activation,inputn_train,label_train,inputn_test,label_test);%遗传算法


%% 优化后结果分析
figure
[r c]=size(trace);
plot(trace,'b--');
title('适应度曲线图')
xlabel('进化代数');ylabel('诊断正确率');
x=bestchrom;
%% 把最优初始阀值权值赋予ELM重新训练与预测
TYPE=1;
if TYPE  == 1
    T1  = ind2vec(label_train);
end
w1=x(1:inputnum*hiddennum);
B1=x(inputnum*hiddennum+1:inputnum*hiddennum+hiddennum)';
%% train
W=reshape(w1,hiddennum,inputnum);
Q=size(inputn_train,2);
BiasMatrix = repmat(B1,1,Q);
tempH = W * inputn_train + BiasMatrix;
switch activation
    case 'sig'
        H = 1 ./ (1 + exp(-tempH));
    case 'sin'
        H = sin(tempH);
    case 'hardlim'
        H = hardlim(tempH);
end
LW = pinv(H') * T1';
%% test
Tn_sim = elmpredict(inputn_test,W,B1,LW,activation,TYPE);

youhua_test_accuracy=sum(Tn_sim==label_test)/length(label_test)
figure
stem(label_test,'*')
hold on
plot(Tn_sim,'p')
title('优化后的ELM')
legend('期望输出','实际输出')


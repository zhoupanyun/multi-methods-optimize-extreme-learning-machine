function [IW,B,LW,TF,TYPE,T] = elmtrain(P_train,T,N,TF,TYPE)
%显然在这个程序中elmtrain跟了6个参数，正常情况下nargin=6，但是有些省略的时候，就要用到下面几句程序进行默认赋值
if nargin < 2
    error('ELM:Arguments','Not enough input arguments.');
end  %输入参数必须大于等于2个，否则无法进行建模，因为至少要有输出输入参数才行
if nargin < 3
    N = size(P_train,2); %如果只有输入输出，默认隐含层神经元为样本数
end
if nargin < 4
    TF = 'sig';%如果只有输出输入，隐含层神经元数量，默认激活函数为sigmoid函数
end
if nargin < 5
    TYPE = 0;%如果没有定义函数的作用，默认为回归拟合
end
% 其中基本上都是在主程序中定义了那几个参数的，所以上面这几句一般用不到。。。。。。。。。
%%%%%%%%%%%%*****************************

if size(P_train,2) ~= size(T,2)
    error('ELM:Arguments','The columns of P and T must be same.');
end
%输出样本数量必须与输出样本数量一致。
[R,Q] = size(P_train);%
if TYPE  == 1
    T1  = ind2vec(T);
end
%如果定义的是分类，就将训练输出转为向量索引   http://blog.csdn.net/u011314012/article/details/51191006
[S,Q] = size(T1); %

% 随机产生输入权重矩阵
IW = rand(N,R) * 2 - 1;

% 随机产生隐层偏置
B = rand(N,1);
BiasMatrix = repmat(B,1,Q);

% 计算隐层输出H
tempH = IW * P_train + BiasMatrix;
switch TF
    case 'sig'
        H = 1 ./ (1 + exp(-tempH));
    case 'sin'
        H = sin(tempH);
    case 'hardlim'
        H = hardlim(tempH);
end
% 计算隐层到输出层之间的权重
LW = pinv(H') * T1';
TY=(H'*LW)';

if TYPE  == 1
    temp_Y=zeros(1,size(TY,2));
    for n=1:size(TY,2)
        [max_Y,index]=max(TY(:,n));
        temp_Y(n)=index;
    end
    Y_train=temp_Y;
    train_accuracy=sum(Y_train==T)/length(T);
end
if TYPE==0
    train_accuracy=mse(TY,T);
end
end
%pinv与inv都是用来求矩阵的逆矩阵，但是inv是知道存在逆矩阵的情况下用的
%当不清楚矩阵是否存在逆矩阵的情况时，或者根本不存在逆矩阵，就用PINV来伪逆矩阵
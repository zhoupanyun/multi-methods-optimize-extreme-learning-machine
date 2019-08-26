function y = fun(x,inputnum,hiddennum,TYPE,activation,inputn_train,label_train,inputn_test,label_test)
%该函数用来计算适应度值
if TYPE  == 1
    T1  = ind2vec(label_train);
else
    T1=label_train;
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
%%
Tn_sim = elmpredict(inputn_test,W,B1,LW,activation,TYPE);
test_accuracy=sum(Tn_sim==label_test)/length(label_test);
y=test_accuracy;%以分类准确率作为适应度值


function [y,trace]=afforelm(inputnum,hiddennum,TYPE,activation,inputn_train,label_train,inputn_test,label_test)
%%  参数设置
fishnum=20;    %生成50只人工鱼
MAXGEN=100;     %最多迭代次数
try_number=100;%最多试探次数
visual=1;      %感知距离
delta=0.618;   %拥挤度因子
step=0.1;      %步长
xmin=-1;
xmax=2;%寻优范围
x=[xmin;xmax];
d=inputnum*hiddennum+hiddennum;
gen=1;
%% 初始化鱼群
for i=1:fishnum
  X(:,i)=(xmin+(xmax-xmin).*rand(1,d));
  Y(1,i)=fun(X(:,i)',inputnum,hiddennum,TYPE,activation,inputn_train,label_train,inputn_test,label_test); 
end
[fit,I]=max(Y); % fim 初始极值
BestX=[]; 
BestY=[];   %每步中最优的函数值

besty=0;                %最优函数值
%% 
while gen<=MAXGEN
    gen
    for i=1:fishnum
        [Xi1,Yi1]=AF_swarm(X,i,visual,step,delta,try_number,x,Y,inputnum,hiddennum,TYPE,activation,inputn_train,label_train,inputn_test,label_test);     %聚群行为 得到新位置和新适应度
        [Xi2,Yi2]=AF_follow(X,i,visual,step,delta,try_number,x,Y,inputnum,hiddennum,TYPE,activation,inputn_train,label_train,inputn_test,label_test);    %追尾行为
        if Yi1>Yi2
            X(:,i)=Xi1;
            Y(1,i)=Yi1;
        else
            X(:,i)=Xi2;
            Y(1,i)=Yi2;%进行比较更新
        end
    end
    [Ymax,index]=max(Y);

    if Ymax>besty
        besty=Ymax;
        bestx=X(:,index);
        BestY(gen)=Ymax;
        BestX(:,gen)=X(:,index);
    else
        BestY(gen)=BestY(gen-1);
        BestX(:,gen)=BestX(:,gen-1);
    end    
    gen=gen+1;
end
y=bestx';
trace=BestY;
end%出售各类算法优化深度极限学习机代码392503054

function [h,trace]=saforelm(inputnum,hiddennum,TYPE,activation,inputn_train,label_train,inputn_test,label_test)
%% 参数设定  
%%%冷却表参数%%%%%%%%%%
L=10;      %马尔科夫链长度
K=0.9;    %衰减因子
S=0.01;     %步长因子
T=100;      %初始温度
P=0;        %Metroppolis过程中总接受点
YZ=1E-2;    %容差，相邻两次温度的差值
max_iter=100;%最大退火次数                  %while循环用容差作为终止条件，for循环用最大退火次数作为终止条件
%随机产生10个初始值，并从10个初值中产生1个处置最优解
Xs=1;
Xx=0;
pop=20;
D=inputnum*hiddennum+hiddennum;
Prex=(rand(D,pop)*(Xs-Xx)+Xx);
for i=1:pop
   funt(i)=fun(Prex(:,i)',inputnum,hiddennum,TYPE,activation,inputn_train,label_train,inputn_test,label_test); 
end
[sort_val,index_val] = sort(funt,'descend');
Prebestx=Prex(:,index_val(end));
Prex=Prex(:,index_val(end-1));
Bestx=Prex;
bestfit=zeros(1,max_iter);
%每迭代一次退火一次(降温)，直到满足迭代条件为止
for iter=1:max_iter
    T=K*T;%在当前温度T下迭代次数
    for i=1:L
        %在附近随机选下一点
        Nextx=Prex+S*(rand(D,1)*(Xs-Xx)+Xx);
        %边界条件处理
        for ii=1:D
            if Nextx(ii)>Xs | Nextx(ii)<Xx
                Nextx(ii)=rand*(Xs-Xx)+Xx;
            end
        end
        %%是否全局最优解
        a=fun(Bestx',inputnum,hiddennum,TYPE,activation,inputn_train,label_train,inputn_test,label_test); 
        b=fun(Nextx',inputnum,hiddennum,TYPE,activation,inputn_train,label_train,inputn_test,label_test); 
        if a<b
           prebest=a;
           Prebestx=Bestx;%保留上一个最优解
           Bestx=Nextx;%更新最优解
           a=b;
        end%如果新解更好，用新解替代最优解，原最优解变为前最优解
%%%%%%%%%%%%Metropolis过程
        c=fun(Prex',inputnum,hiddennum,TYPE,activation,inputn_train,label_train,inputn_test,label_test); 
        if c<b 
            %%%接受新解
            Prex=Nextx;
            P=P+1;
        else
            changer=-1*(b-c)/T;
            p1=exp(changer);
            %%%以一定概率接受较差的解
            if p1>rand
                Prex=Nextx;
                P=P+1;
            end
        end
       trace(P+1)=a;    
    end
    

   %deta=abs(a-prebest);
end
h=Bestx';
end
%出售各类算法优化深度极限学习机代码392503054

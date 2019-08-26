function [y,trace]=antforelm(inputnum,hiddennum,TYPE,activation,inputn_train,label_train,inputn_test,label_test);%蚁群算法%%%%%%%%%%%%%%%%%%%%蚁群算法求函数极值%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%初始化%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
m=20;                    %蚂蚁个数
G_max=100;               %最大迭代次数
Rho=0.5;                 %信息素蒸发系数
P0=0.5;                  %转移概率常数
XMAX= 1;                 %搜索变量x最大值
XMIN=0;                %搜索变量x最小值
d=inputnum*hiddennum+hiddennum;;     
%%%%%%%%%%%%%%%%%随机设置蚂蚁初始位置%%%%%%%%%%%%%%%%%%%%%%
for i=1:m
    X(i,:)=(XMIN+(XMAX-XMIN).*rand(1,d));
    Tau(i)=fun(X(i,:),inputnum,hiddennum,TYPE,activation,inputn_train,label_train,inputn_test,label_test); 
end
step=0.1;                %局部搜索步长
for NC=1:G_max
    NC
    lamda=1/NC;
    [Tau_best,BestIndex]=max(Tau);
    %%%%%%%%%%%%%%%%%%计算状态转移概率%%%%%%%%%%%%%%%%%%%%
    for i=1:m
        P(NC,i)=(Tau(BestIndex)-Tau(i))/Tau(BestIndex);
    end
    %%%%%%%%%%%%%%%%%%%%%%位置更新%%%%%%%%%%%%%%%%%%%%%%%%
    for i=1:m
           %%%%%%%%%%%%%%%%%局部搜索%%%%%%%%%%%%%%%%%%%%%%
        if P(NC,i)<P0
            temp1=X(i,:)+(rand(1,d))*step*lamda;
           
        else
            %%%%%%%%%%%%%%%%全局搜索%%%%%%%%%%%%%%%%%%%%%%%
             temp1=X(i,:)+(XMAX-XMIN)*(rand(1,d));
        end
        %%%%%%%%%%%%%%%%%%%%%边界处理%%%%%%%%%%%%%%%%%%%%%%%
        if temp1<XMIN
            temp1=rand;
        end
        if temp1>XMAX
            temp1=rand;
        end
        %%%%%%%%%%%%%%%%%%蚂蚁判断是否移动%%%%%%%%%%%%%%%%%%
        if fun(temp1,inputnum,hiddennum,TYPE,activation,inputn_train,label_train,inputn_test,label_test)>fun(X(i,:),inputnum,hiddennum,TYPE,activation,inputn_train,label_train,inputn_test,label_test)
            X(i,:)=temp1;
        end
    end
    %%%%%%%%%%%%%%%%%%%%%%%更新信息素%%%%%%%%%%%%%%%%%%%%%%%
    for i=1:m
        Tau(i)=(1-Rho)*Tau(i)+fun(X(i,:),inputnum,hiddennum,TYPE,activation,inputn_train,label_train,inputn_test,label_test); 
    end
    [value,index]=max(Tau);
    trace(NC)=fun(X(index,:),inputnum,hiddennum,TYPE,activation,inputn_train,label_train,inputn_test,label_test); 
    
end
[min_value,max_index]=max(Tau);
minX=X(max_index,:);                           %最优变量
minValue=fun(X(max_index,:),inputnum,hiddennum,TYPE,activation,inputn_train,label_train,inputn_test,label_test);   %最优值 

y=minX;
end
%出售各类算法优化深度极限学习机代码392503054
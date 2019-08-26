function [Xnext,Ynext]=AF_swarm(X,i,visual,step,deta,try_number,LBUB,lastY,inputnum,hiddennum,TYPE,activation,inputn_train,label_train,inputn_test,label_test)
% 聚群行为
%输入：
%X           所有人工鱼的位置
%i           当前人工鱼的序号
%visual      感知范围
%step        最大移动步长
%deta        拥挤度
%try_number  最大尝试次数
%LBUB        各个数的上下限
%lastY       上次的各人工鱼位置的食物浓度

%输出：
%Xnext       Xi人工鱼的下一个位置  
%Ynext       Xi人工鱼的下一个位置的食物浓度
Xi=X(:,i);
D=dist(Xi,X);
index=find(D>0 & D<visual);
nf=length(index);
if nf>0
    for j=1:size(X,1)
        Xc(j,1)=mean(X(j,index));
    end
    Yc=fun(Xc',inputnum,hiddennum,TYPE,activation,inputn_train,label_train,inputn_test,label_test); 
    Yi=lastY(i);
    if Yc/nf>deta*Yi
        Xnext=Xi+rand*step*(Xc-Xi)/norm(Xc-Xi);
        for i=1:length(Xnext)
            if  Xnext(i)>LBUB(2)
                Xnext(i)=rand;
            end
            if  Xnext(i)<LBUB(1)
                Xnext(i)=rand;
            end
        end
        Ynext=fun(Xnext',inputnum,hiddennum,TYPE,activation,inputn_train,label_train,inputn_test,label_test); 
    else
        [Xnext,Ynext]=AF_prey(Xi,i,visual,step,try_number,LBUB,lastY,inputnum,hiddennum,TYPE,activation,inputn_train,label_train,inputn_test,label_test);
    end
else
    [Xnext,Ynext]=AF_prey(Xi,i,visual,step,try_number,LBUB,lastY,inputnum,hiddennum,TYPE,activation,inputn_train,label_train,inputn_test,label_test);
end%出售各类算法优化深度极限学习机代码392503054
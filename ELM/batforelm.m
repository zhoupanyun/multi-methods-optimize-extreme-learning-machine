function [y,trace]=batforelm(inputnum,hiddennum,TYPE,activation,inputn_train,label_train,inputn_test,label_test)

para=[20 100 0.5 0.5 0 2 0 1];  
%%
n=para(1);            % Population size, typically 10 to 40 种群大小，一共20只蝙蝠
N_gen=para(2);        % Number of generations  迭代数
A=para(3);            % Loudness  (constant or decreasing) 发送超声脉冲的响度
r=para(4);            % Pulse rate (constant or decreasing)发送超声脉冲的间隔，发射率
Fmin=para(5);         % Frequency minimum 脉冲频率最小值
Fmax=para(6);         % Frequency maximum脉冲频率最大值
Xmin=para(7);
Xmax=para(8);         % 边界大小，就是优化的隐含层节点数范围边界
N_iter=0;             % Total number of function evaluations
d=inputnum*hiddennum+hiddennum;;          % Number of dimensions 每个蝙蝠有d个属性，就是在d维空间中搜索
F=zeros(n,1);         % 初始化脉冲频率 Frequency
v=zeros(n,d);         % 初始化速度    Velocities
trace=[];
%% 定义初始极值，与初始极值位置
for i=1:n
  X(i,:)=(Xmin+(Xmax-Xmin).*rand(1,d));
  Fitness(i)=fun(X,inputnum,hiddennum,TYPE,activation,inputn_train,label_train,inputn_test,label_test); 
end
[fit,I]=max(Fitness); % fim 初始极值
best=X(I,:);           % 初始全局最优解的位置

%% 按照公式依次迭代直到满足精度或者迭代次数
%for t=1:N_gen, %1000代   也可以设计成满足一个阈值while (fmin>1e-5)....end
for  t=1:N_gen
      t
        for i=1:n  %10
          F(i)=Fmin+(Fmin-Fmax)*rand;
          v(i,:)=v(i,:)+(X(i,:)-best)*F(i);
          S(i,:)=X(i,:)+v(i,:);
          % 更新位置，速度，频率
          % Pulse rate
          if rand>r  %从最佳解中选择一个,围绕选择的解产生一个局部解
          sigmoid=0.001;% 缩放因子
              S(i,:)=best+sigmoid*randn(1,d);%%局部解
          end
          
          S(i,:)=boundary(S(i,:));%%看看产生的这个局部最优解有没有在寻优范围外
          
     % Evaluate new solutions
             Fnew=fun(S(i,:),inputnum,hiddennum,TYPE,activation,inputn_train,label_train,inputn_test,label_test); 
     % Update if the solution improves, or not too loud
           if (Fnew>=Fitness(i)) && (rand<A) ,
                X(i,:)=S(i,:);
                Fitness(i)=Fnew;
                %%如果新解更优，此时接受新解，并且增大加快发射声波的次数r，以及减小每个声波的响度A
                %这个程序为了方便，省略了此步骤
           end

          % Update the current best solution
          if Fnew>=fit
                best=S(i,:);
                fit=Fnew;
          end%将局部新解更新到全局最优解中
        end
        N_iter=N_iter+n;
        trace(t)=fit;
end
y=best;
%出售各类算法优化深度极限学习机代码392503054
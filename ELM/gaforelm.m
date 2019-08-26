function [bestchrom ,trace]=gaforelm(inputnum,hiddennum,TYPE,activation,inputn_train,label_train,inputn_test,label_test)
%% 遗传算法参数初始化
maxgen=100;                         %进化代数，即迭代次数
sizepop=20;                        %种群规模
pcross=[0.7];                       %交叉概率选择，0和1之间
pmutation=[0.05];                    %变异概率选择，0和1之间
%节点总数
numsum=inputnum*hiddennum+hiddennum;
lenchrom=ones(1,numsum);       
bound=[-1*ones(numsum,1) 1*ones(numsum,1)];    %数据范围
%------------------------------------------------------种群初始化------------------------------%------------------
individuals=struct('fitness',zeros(1,sizepop), 'chrom',[]);  %将种群信息定义为一个结构体
avgfitness=[];                      %每一代种群的平均适应度
bestfitness=[];                     %每一代种群的最佳适应度
bestchrom=[];                       %适应度最好的染色体
trace=0;
%初始化种群
for i=1:sizepop
    %随机产生一个种群
    individuals.chrom(i,:)=Code(lenchrom,bound);    %编码
    x=individuals.chrom(i,:);
    %计算适应度
    individuals.fitness(i)=fun(x,inputnum,hiddennum,TYPE,activation,inputn_train,label_train,inputn_test,label_test);   %染色体的适应度
end

%找最好的染色体
[bestfitness bestindex]=max(individuals.fitness);
bestchrom=individuals.chrom(bestindex,:);  %最好的染色体
%% 迭代求解最佳初始阀值和权值
% 进化开始
for i=1:maxgen
    
    %选择
    individuals=select(individuals,sizepop);
    %交叉
    individuals.chrom=Cross(pcross,lenchrom,individuals.chrom,sizepop,bound);
    %变异
    individuals.chrom=Mutation(pmutation,lenchrom,individuals.chrom,sizepop,i,maxgen,bound);
    % 计算适应度
    for j=1:sizepop
        x=individuals.chrom(j,:); %解码
        individuals.fitness(j)=fun(x,inputnum,hiddennum,TYPE,activation,inputn_train,label_train,inputn_test,label_test);
    end
  %找到最小和最大适应度的染色体及它们在种群中的位置
    [newbestfitness,newbestindex]=max(individuals.fitness);
    [worestfitness,worestindex]=min(individuals.fitness);
    % 代替上一次进化中最好的染色体
    if bestfitness<newbestfitness
        bestfitness=newbestfitness;
        bestchrom=individuals.chrom(newbestindex,:);
    end
    individuals.chrom(bestindex,:)=bestchrom;
    individuals.fitness(bestindex)=bestfitness;
    trace(i)=bestfitness ;%记录每一代进化中最好的适应度和平均适应度
    
end

end
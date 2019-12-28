% SURF特征文件作为算法的输入。主要包含 4 个.mat 文件：Caltech.mat, amazon.mat,
% webcam.mat, dslr.mat。对应 4 个不同的领域。彼此之间两两一组，就是一个迁移
% 学习任务。每个数据文件包含两个部分： fts 为 800 维的特征， labels 为对应的标注
% 我们选择由 Caltech.mat 作为源域，由 amazon.mat 作为目标域


%% 首先对数据进行加载并做简单地归一化，将最后的数据存入Xs,Ys,Xt,Yt
%这四个变量分别对应源域的特征、标注和目标域的特征、标注
load('.\surf\Caltech10_SURF_L10.mat');     %加载源域数据
%由数据可知800维的特征，那么对于每个样本都有800个特征元素，先将这每个样本的特征
%元素加起来再除，得到每个特征元素的归一化处理。
%语法有B = repmat(A,m,n)，将矩阵 A 复制 m×n 块，即把 A 作为 B 的元素，
%B 由 m×n 个 A 平铺而成
fts = fts ./ repmat(sum(fts, 2), 1, size(fts, 2));  %将fts中的元素按列数800归一化处理
Xs = zscore(fts, 1); %z=(x-mean(x))./std(x) 标准化
Ys = labels;
clear fts; clear labels;
load('.\surf\amazon_SURF_L10.mat');        %加载目标域数据
fts = fts ./ repmat(sum(fts, 2), 1, size(fts, 2));  %将fts中的元素按列数800归一化处理
Xt = zscore(fts, 1); %z=(x-mean(x))./std(x) 标准化
Yt = labels;
clear fts; clear labels;

%初始化参数
options.gamma = 2;
options.kernel_type = 'linear';
options.lambda = 1.0;
options.dim = 20;
[X_src_new, X_tar_new, A] = TCA(Xs, Xt, options);

% Use knn to predict the target label
%这里是一个分类器的模型，先对源数据进行训练，然后直接用于目标域的测试
knn_model = fitcknn (X_src_new, Ys , 'NumNeighbors', 100) ;
Y_tar_pseudo = knn_model.predict(X_tar_new) ;
acc = length(find(Y_tar_pseudo==Yt))/length(Yt);
fprintf('Acc=%0.4f \n', acc);

%%可以发现这里并不是，没有时间或硬件训练，而是有标注的数据集有限，需要对目标域数据进行标注。

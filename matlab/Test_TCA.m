% SURF�����ļ���Ϊ�㷨�����롣��Ҫ���� 4 ��.mat �ļ���Caltech.mat, amazon.mat,
% webcam.mat, dslr.mat����Ӧ 4 ����ͬ�����򡣱˴�֮������һ�飬����һ��Ǩ��
% ѧϰ����ÿ�������ļ������������֣� fts Ϊ 800 ά�������� labels Ϊ��Ӧ�ı�ע
% ����ѡ���� Caltech.mat ��ΪԴ���� amazon.mat ��ΪĿ����


%% ���ȶ����ݽ��м��ز����򵥵ع�һ�������������ݴ���Xs,Ys,Xt,Yt
%���ĸ������ֱ��ӦԴ�����������ע��Ŀ�������������ע
load('.\surf\Caltech10_SURF_L10.mat');     %����Դ������
%�����ݿ�֪800ά����������ô����ÿ����������800������Ԫ�أ��Ƚ���ÿ������������
%Ԫ�ؼ������ٳ����õ�ÿ������Ԫ�صĹ�һ������
%�﷨��B = repmat(A,m,n)�������� A ���� m��n �飬���� A ��Ϊ B ��Ԫ�أ�
%B �� m��n �� A ƽ�̶���
fts = fts ./ repmat(sum(fts, 2), 1, size(fts, 2));  %��fts�е�Ԫ�ذ�����800��һ������
Xs = zscore(fts, 1); %z=(x-mean(x))./std(x) ��׼��
Ys = labels;
clear fts; clear labels;
load('.\surf\amazon_SURF_L10.mat');        %����Ŀ��������
fts = fts ./ repmat(sum(fts, 2), 1, size(fts, 2));  %��fts�е�Ԫ�ذ�����800��һ������
Xt = zscore(fts, 1); %z=(x-mean(x))./std(x) ��׼��
Yt = labels;
clear fts; clear labels;

%��ʼ������
options.gamma = 2;
options.kernel_type = 'linear';
options.lambda = 1.0;
options.dim = 20;
[X_src_new, X_tar_new, A] = TCA(Xs, Xt, options);

% Use knn to predict the target label
%������һ����������ģ�ͣ��ȶ�Դ���ݽ���ѵ����Ȼ��ֱ������Ŀ����Ĳ���
knn_model = fitcknn (X_src_new, Ys , 'NumNeighbors', 100) ;
Y_tar_pseudo = knn_model.predict(X_tar_new) ;
acc = length(find(Y_tar_pseudo==Yt))/length(Yt);
fprintf('Acc=%0.4f \n', acc);

%%���Է������ﲢ���ǣ�û��ʱ���Ӳ��ѵ���������б�ע�����ݼ����ޣ���Ҫ��Ŀ�������ݽ��б�ע��

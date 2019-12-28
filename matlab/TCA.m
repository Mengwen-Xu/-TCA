function [X_src_new, X_tar_new, A] = TCA(X_src, X_tar, options)
%TCA 这是转移成分分析的实现。
% 参考文献:Sinno Pan e t al。通过转移成分分析实现领域自适应

% Iuputs:
%%% X_src: source feature matrix, ns ? n_feature
%%% X_tar: target feature matrix, nt ? n_feature
%%% options: option struct
%%%%% lambda: regularization parameter 正则化参数
%%%%% dim: dimensionality after adaptation (dim <= n_feature )适应后de维度
%%%%% kernel_tpye: kernel name, choose from 'primal' | 'linear' | 'rbf'
%%%%%内核名，从'primal' | 'linear' | 'rbf'中选择
%%%%% gamma: bandwidth for rbf kernel , can be missed for other kernels

% Outputs:
%%% X_src_new: transformed source feature matrix , ns ? dim
%%% X_tar_new: transformed target feature matrix , nt ? dim
%%% A: adaptation matrix , ( ns + nt ) ? ( ns + nt )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %% Set options
    lambda = options.lambda;              
	dim = options.dim;                    
	kernel_type = options.kernel_type;    
	gamma = options.gamma;                

    %% Calculate
	X = [X_src',X_tar'];
    %sparse从常规矩阵转换稀疏矩阵，如果 A 是矩阵，则 sum(A) 将返回包含每列总和的行向量
    X = X*diag(sparse(1./sqrt(sum(X.^2))));%sparse创建稀疏矩阵，只会存储非零的值
    %一顿操作之后，其实就是对每个列向量，即每个样本的特征向量做单位向量化
	[m,n] = size(X);
	ns = size(X_src,1); %源域样本数量
	nt = size(X_tar,1); %目标域样本数量
	e = [1/ns*ones(ns,1);-1/nt*ones(nt,1)];
	M = e * e';%这里的M矩阵是文档中的L矩阵
	M = M / norm(M,'fro');% norm(X,'fro') 返回矩阵 X 的 Frobenius 范数,是一个值相当于所有元素平方和开方
	H = eye(n)-1/(n)*ones(n,n);
	if strcmp(kernel_type,'primal')%strcmp(s1,s2) 比较 s1 和 s2，如果二者相同，则返回 1 (true)，否则返回 0 (false)
        %表示核函数方法选的是不用计算K矩阵了，直接把X当作K
		[A,~] = eigs(X*M*X'+lambda*eye(m),X*H*X',dim,'SM');
        %d = eigs(A,B,___) 解算广义特征值问题 A*V = B*V*D。
        %您可以选择指定 k、sigma、opts 或名称-值对组作为额外的输入参数
		Z = A' * X;%将原始数据变换到另一个空间
        Z = Z * diag(sparse(1./sqrt(sum(Z.^2))));%再做单位向量化
		X_src_new = Z(:,1:ns)';
		X_tar_new = Z(:,ns+1:end)';
	else
	    K = TCA_kernel(kernel_type,X,[],gamma);%按照选定的方法核函数方法，求出核矩阵
	    [A,~] = eigs(K*M*K'+lambda*eye(n),K*H*K',dim,'SM');
	    Z = A' * K;
        Z = Z*diag(sparse(1./sqrt(sum(Z.^2))));
        X_src_new = Z(:,1:ns)';
		X_tar_new = Z(:,ns+1:end)';
	end
end




%% 
function K = TCA_kernel(ker,X,X2,gamma)
% With Fast Computation of the RBF kernel matrix
% To speed up the computation, we exploit a decomposition of the Euclidean distance (norm)
%
% Inputs:
%       ker:    'linear','rbf','sam'
%       X:      data matrix (features * samples)
%       gamma:  bandwidth of the RBF/SAM kernel
% Output:
%       K: kernel matrix
    switch ker
        case 'linear'

            if isempty(X2)
                K = X'*X;
            else
                K = X'*X2;
            end

        case 'rbf'

            n1sq = sum(X.^2,1);
            n1 = size(X,2);

            if isempty(X2)
                D = (ones(n1,1)*n1sq)' + ones(n1,1)*n1sq -2*X'*X;
            else
                n2sq = sum(X2.^2,1);
                n2 = size(X2,2);
                D = (ones(n2,1)*n1sq)' + ones(n1,1)*n2sq -2*X'*X2;
            end
            K = exp(-gamma*D); 

        case 'sam'

            if isempty(X2)
                D = X'*X;
            else
                D = X'*X2;
            end
            K = exp(-gamma*acos(D).^2);

        otherwise
            error(['Unsupported kernel ' ker])
    end
end


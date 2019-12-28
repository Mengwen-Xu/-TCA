function [X_src_new, X_tar_new, A] = TCA(X_src, X_tar, options)
%TCA ����ת�Ƴɷַ�����ʵ�֡�
% �ο�����:Sinno Pan e t al��ͨ��ת�Ƴɷַ���ʵ����������Ӧ

% Iuputs:
%%% X_src: source feature matrix, ns ? n_feature
%%% X_tar: target feature matrix, nt ? n_feature
%%% options: option struct
%%%%% lambda: regularization parameter ���򻯲���
%%%%% dim: dimensionality after adaptation (dim <= n_feature )��Ӧ��deά��
%%%%% kernel_tpye: kernel name, choose from 'primal' | 'linear' | 'rbf'
%%%%%�ں�������'primal' | 'linear' | 'rbf'��ѡ��
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
    %sparse�ӳ������ת��ϡ�������� A �Ǿ����� sum(A) �����ذ���ÿ���ܺ͵�������
    X = X*diag(sparse(1./sqrt(sum(X.^2))));%sparse����ϡ�����ֻ��洢�����ֵ
    %һ�ٲ���֮����ʵ���Ƕ�ÿ������������ÿ��������������������λ������
	[m,n] = size(X);
	ns = size(X_src,1); %Դ����������
	nt = size(X_tar,1); %Ŀ������������
	e = [1/ns*ones(ns,1);-1/nt*ones(nt,1)];
	M = e * e';%�����M�������ĵ��е�L����
	M = M / norm(M,'fro');% norm(X,'fro') ���ؾ��� X �� Frobenius ����,��һ��ֵ�൱������Ԫ��ƽ���Ϳ���
	H = eye(n)-1/(n)*ones(n,n);
	if strcmp(kernel_type,'primal')%strcmp(s1,s2) �Ƚ� s1 �� s2�����������ͬ���򷵻� 1 (true)�����򷵻� 0 (false)
        %��ʾ�˺�������ѡ���ǲ��ü���K�����ˣ�ֱ�Ӱ�X����K
		[A,~] = eigs(X*M*X'+lambda*eye(m),X*H*X',dim,'SM');
        %d = eigs(A,B,___) �����������ֵ���� A*V = B*V*D��
        %������ѡ��ָ�� k��sigma��opts ������-ֵ������Ϊ������������
		Z = A' * X;%��ԭʼ���ݱ任����һ���ռ�
        Z = Z * diag(sparse(1./sqrt(sum(Z.^2))));%������λ������
		X_src_new = Z(:,1:ns)';
		X_tar_new = Z(:,ns+1:end)';
	else
	    K = TCA_kernel(kernel_type,X,[],gamma);%����ѡ���ķ����˺�������������˾���
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


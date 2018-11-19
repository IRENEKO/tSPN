function [core,nz,data,testdata]=tspn_iden(tensor,weight,sample_train,sample_test,opts)
switch opts.case
    case 0
        rd(1)=false;
        rd(2)=false;
    case 1
        rd(1)=true;
        rd(2)=false;
    case 2
        rd(1)=false;
        rd(2)=true;
    otherwise
        rd(1)=true;
        rd(2)=true;
end

[N,d,~]=size(tensor);

% 1. Construct the high-rank tensor train and compute the probabilities of
% samples and non samples for later comparison.
% Step 1.1. Construct the high-rank tensor train
core.core{1}(1,:,:)=fliplr(reshape(tensor(:,1,:),[N,2]))'.*weight;
core.n(1,:)=[1,2,N];
for i=2:d-1
    for j=1:2
        data=tensor(:,i,j);
        core.core{i}(:,3-j,:)=diag(data);
    end
    core.n(i,:)=[N,2,N];
end
core.core{d}(:,:,1)=fliplr(reshape(tensor(:,d,:),[N,2]));
core.n(d,:)=[N,2,1];

% Step 1.2. Construct 'Training Samples' and 'Testing Samples'
I=sample_train;
[N,d]=size(I);
Data=cell(1,d);
testData=cell(1,d);
for i=1:d
    Data{i}=zeros(size(I,1),2);
    testData{i}=zeros(size(sample_test,1),2);
    for j=1:size(I,1)
        Data{i}(j,2-I(j,i))=1;
    end
    for j=1:size(sample_test,1)
        testData{i}(j,2-sample_test(j,i))=1;
    end
end

% Step 1.3. Find 'Training Non Samples' and 'Testing Non Samples' 
tempI=[I;sample_test];
[kk,~]=nonrepeated(tempI);
C=findnonsample(tempI,kk+100);

oData=cell(1,d);
for i=1:d
    oData{i}=zeros(size(C,1),2);    
    for j=1:size(C,1)
        oData{i}(j,2-C(j,i))=1;
    end
end

% Step 1.4. Compute the probabilities for 'Training Samples', 'Testing
% Samples', 'Training Non Samples' and 'Testing Non Samples'.
data=Data{1}*reshape(core.core{1},[2,core.n(1,end)]);
testdata=testData{1}*reshape(core.core{1},[2,core.n(1,end)]);
odata=oData{1}*reshape(core.core{1},[2,core.n(1,end)]);
for i=2:d
    data=dotkron(data,Data{i})*reshape(core.core{i},[prod(core.n(i,1:2)),core.n(i,end)]);
    testdata=dotkron(testdata,testData{i})*reshape(core.core{i},[prod(core.n(i,1:2)),core.n(i,end)]);
    odata=dotkron(odata,oData{i})*reshape(core.core{i},[prod(core.n(i,1:2)),core.n(i,end)]);
end

% Step 1.5. Plot the probabilities.
figure;
subplot(1,4,1);
plot(data);
subplot(1,4,2);
plot(testdata)
subplot(1,4,3);
plot(odata(1:kk))
subplot(1,4,4);
plot(odata(kk+1:end))

figure;
subplot(1,2,1);
samples=[data;testdata];
edges=[0:0.005:0.05 0.05:0.01:0.15 0.15:0.01:max(samples)+0.01];
histogram(samples,edges)
subplot(1,2,2);
nonsamples=odata;
histogram(nonsamples)



% 2. Construct the training inputs/outputs for system identification like
% training. Identify a TT using the training I/O. Compute the probabilities
% of samples and non samples for comparison.
% 2.1. Construct training inputs ('Training Samples' and 'Training Non
% Samples') and outputs.
I=[I;C(1:kk,:)];
[N,d]=size(I);
Data=cell(1,size(I,2));
for i=1:size(I,2)
    Data{i}=zeros(size(I,1),2);
    for j=1:size(I,1)
        Data{i}(j,2-I(j,i))=1;
    end
end

output=[data;odata(1:kk)];



if rd(1)
% 2.2. Initialize a TT.
Itr=3;
r=[1,5*ones(1,d-1),1];
for i=1:d
    core.core{i}=exp(rand(r(i),2,r(i+1)));
    core.core{i}=core.core{i}./norm(core.core{i}(:));
    core.n(i,:)=[r(i),2,r(i+1)];
end
tempr{d}=ones(1,N);
tempr{d-1}=k_mode_product(core.core{d},Data{d},2);
for i=d-1:-1:2
    sz=size(tempr{i});
    tempr{i-1}=(dotkron(Data{i},tempr{i}')*reshape(core.core{i},[core.n(i,1),prod(core.n(i,2:3))])')';
end
templ{1}=ones(N,1);

% 2.3. Idenfity a TT using training inputs and outputs.
count=1;
for itr=1:Itr
    % % left to right
    for i=1:d-1
        temp=dotkron(templ{i},Data{i},tempr{i}');
        core.core{i}=reshape(lsqnonneg(temp,output),core.n(i,:));
        t=temp*core.core{i}(:);    
        re(count)=norm(t-output)/norm(output);
        count=count+1;
        temp=reshape(sum(sum(core.core{i},2),1),[1,core.n(i,end)]);
        k=find(temp~=0);
        if length(k)~=core.n(i,end)
            core.core{i}=core.core{i}(:,:,k);
            core.core{i+1}=core.core{i+1}(k,:,:);
            core.n(i,:)=[core.n(i,1:2),length(k)];
            core.n(i+1,:)=[length(k),core.n(i+1,2:3)];
        end
        tempa=diag(temp(k));
        tempb=diag(1./temp(k));
        if length(tempa)==1
            core.core{i}=tempb.*core.core{i};
        else
            core.core{i}=k_mode_product(core.core{i},tempb,3);
        end
        core.core{i+1}=k_mode_product(core.core{i+1},tempa,1);
        templ{i+1}=dotkron(templ{i},Data{i})*reshape(core.core{i},[prod(core.n(i,1:2)),core.n(i,3)]);
    end
    
    % % right to left
    for i=d:-1:2
        temp=dotkron(templ{i},Data{i},tempr{i}');
        core.core{i}=reshape(lsqnonneg(temp,output),core.n(i,:));
        t=temp*core.core{i}(:);    
        re(count)=norm(t-output)/norm(output);
        count=count+1;
        temp=reshape(sum(sum(core.core{i},3),2),[1,core.n(i,1)]);
        k=find(temp~=0);
        if length(k)~=core.n(i,1)
            core.core{i}=core.core{i}(k,:,:);
            core.core{i-1}=core.core{i-1}(:,:,k);
            core.n(i,:)=[length(k),core.n(i,2:3)];
            core.n(i-1,:)=[core.n(i-1,1:2),length(k)];
        end
        tempa=diag(temp(k));
        tempb=diag(1./temp(k));
        if length(tempa)==1
            core.core{i}=tempb.*core.core{i};
            core.core{i-1}=tempa.*core.core{i-1};
        else
            core.core{i}=k_mode_product(core.core{i},tempb,1);
            core.core{i-1}=k_mode_product(core.core{i-1},tempa,3);
        end
        tempr{i-1}=(dotkron(Data{i},tempr{i}')*reshape(core.core{i},[core.n(i,1),prod(core.n(i,2:3))])')';
    end
    if opts.norm
        scale=sum(core.core{1}(:));
        core.core{1}=core.core{1}./scale;
    end
end
% figure;
% plot(re)
% re(end)

% Step 2.4. Find 'Testing Samples' and 'Testing Non Samples' 
testData=cell(1,size(sample_test,2));
oData=cell(1,size(C,2));
for i=1:size(C,2)
    testData{i}=zeros(size(sample_test,1),2);
    oData{i}=zeros(size(C,1),2);
    for j=1:size(sample_test,1)
        testData{i}(j,2-sample_test(j,i))=1;
    end
    for j=1:size(C,1)
        oData{i}(j,2-C(j,i))=1;
    end
end

% Step 2.5. Compute the probabilities for 'Training Samples', 'Testing
% Samples', 'Training Non Samples' and 'Testing Non Samples'.
data=Data{1}*reshape(core.core{1},[2,core.n(1,end)]);
testdata=testData{1}*reshape(core.core{1},[2,core.n(1,end)]);
odata=oData{1}*reshape(core.core{1},[2,core.n(1,end)]);
for i=2:d
    data=dotkron(data,Data{i})*reshape(core.core{i},[prod(core.n(i,1:2)),core.n(i,end)]);
    testdata=dotkron(testdata,testData{i})*reshape(core.core{i},[prod(core.n(i,1:2)),core.n(i,end)]);
    odata=dotkron(odata,oData{i})*reshape(core.core{i},[prod(core.n(i,1:2)),core.n(i,end)]);
end

% Step 2.6. Plot the probabilities.
figure;
subplot(1,4,1);
plot(data(1:size(sample_train,1)));
subplot(1,4,2);
plot(testdata)
subplot(1,4,3);
plot(odata(1:kk))
subplot(1,4,4);
plot(odata(kk+1:end))

figure;
subplot(1,2,1);
samples=[data(1:size(sample_train,1));testdata];
edges=[0:0.005:0.05 0.05:0.01:0.15 0.15:0.01:max(samples)+0.01];
histogram(samples,edges)
subplot(1,2,2);
nonsamples=odata;
histogram(nonsamples)

nz(1)=0;
num=0;
for i=1:d
    nz(1)=nz(1)+nnz(core.core{i}(:));
    num=num+numel(core.core{i});
end
SP=100-nz./num*100;
end

if rd(2)
% NNLasso + NNLS retrain
Itr=2;
r=[1,5*ones(1,d-1),1];
for i=1:d
    core.core{i}=exp(rand(r(i),2,r(i+1)));
    core.core{i}=core.core{i}./norm(core.core{i}(:));
    core.n(i,:)=[r(i),2,r(i+1)];
end
tempr{d}=ones(1,N);
tempr{d-1}=k_mode_product(core.core{d},Data{d},2);
for i=d-1:-1:2
    sz=size(tempr{i});
    tempr{i-1}=(dotkron(Data{i},tempr{i}')*reshape(core.core{i},[core.n(i,1),prod(core.n(i,2:3))])')';
end
templ{1}=ones(N,1);

count=1;
% Non negative lasso with different lambda (determine the sparsities)
lambda=0.005;
opts=[];
opts.tFlag=5;       % run .maxIter iterations
opts.maxIter=1000;   % maximum number of iterations
opts.nFlag=0;       % without normalization
opts.rFlag=1;       % the input parameter 'lambda' is a ratio in (0, 1]
opts.fName = 'nnLeastR';
for itr=1:Itr
    % % left to right
    for i=1:d-1
        temp=dotkron(templ{i},Data{i},tempr{i}');
        g=DPC_nnLasso(temp, output, lambda, opts);
        core.core{i}=reshape(g,core.n(i,:));
        t=temp*core.core{i}(:);    
        re(count)=norm(t-output)/norm(output);
        count=count+1;
        templ{i+1}=dotkron(templ{i},Data{i})*reshape(core.core{i},[prod(core.n(i,1:2)),core.n(i,3)]);
    end
    
    % % right to left
    for i=d:-1:2
        temp=dotkron(templ{i},Data{i},tempr{i}');
        g=DPC_nnLasso(temp, output, lambda, opts);
        core.core{i}=reshape(g,core.n(i,:));
        t=temp*core.core{i}(:);    
        re(count)=norm(t-output)/norm(output);
        count=count+1;
        tempr{i-1}=(dotkron(Data{i},tempr{i}')*reshape(core.core{i},[core.n(i,1),prod(core.n(i,2:3))])')';
    end
end
% For the sparsity determined by the above steps, retrain the tensor
% network to optimize the relative error
for itr=1:Itr
    % % left to right
    for i=1:d-1
        temp=dotkron(templ{i},Data{i},tempr{i}');
%         I=find(core.core{i}>1e-16);
        I=find(core.core{i}~=0);
        core.core{i}=zeros(core.n(i,:));
        core.core{i}(I)=lsqnonneg(temp(:,I),output);
        t=temp*core.core{i}(:);    
        re(count)=norm(t-output)/norm(output);
        count=count+1;
        templ{i+1}=dotkron(templ{i},Data{i})*reshape(core.core{i},[prod(core.n(i,1:2)),core.n(i,3)]);
    end
    
    % % right to left
    for i=d:-1:2
        temp=dotkron(templ{i},Data{i},tempr{i}');
%         I=find(core.core{i}>1e-16);
        I=find(core.core{i}~=0);
        core.core{i}=zeros(core.n(i,:));
        core.core{i}(I)=lsqnonneg(temp(:,I),output);
        t=temp*core.core{i}(:);    
        re(count)=norm(t-output)/norm(output);
        count=count+1;
        tempr{i-1}=(dotkron(Data{i},tempr{i}')*reshape(core.core{i},[core.n(i,1),prod(core.n(i,2:3))])')';
    end
end
figure;
plot(re)
% RE(ko)=re(end);

testData=cell(1,size(sample_test,2));
oData=cell(1,size(C,2));
for i=1:size(C,2)
    testData{i}=zeros(size(sample_test,1),2);
    oData{i}=zeros(size(C,1),2);
    for j=1:size(sample_test,1)
        testData{i}(j,2-sample_test(j,i))=1;
    end
    for j=1:size(C,1)
        oData{i}(j,2-C(j,i))=1;
    end
end

data=Data{1}*reshape(core.core{1},[2,core.n(1,end)]);
testdata=testData{1}*reshape(core.core{1},[2,core.n(1,end)]);
odata=oData{1}*reshape(core.core{1},[2,core.n(1,end)]);
for i=2:d
    data=dotkron(data,Data{i})*reshape(core.core{i},[prod(core.n(i,1:2)),core.n(i,end)]);
    testdata=dotkron(testdata,testData{i})*reshape(core.core{i},[prod(core.n(i,1:2)),core.n(i,end)]);
    odata=dotkron(odata,oData{i})*reshape(core.core{i},[prod(core.n(i,1:2)),core.n(i,end)]);
end

figure;
subplot(1,4,1);
plot(data(1:size(sample_train,1)));
subplot(1,4,2);
plot(testdata)
subplot(1,4,3);
plot(odata(1:kk))
subplot(1,4,4);
plot(odata(kk+1:end))

figure;
subplot(1,2,1);
samples=[data(1:size(sample_train,1));testdata];
edges=[0:0.005:0.05:0.01:0.15 0.15:0.01:max(samples)+0.01];
histogram(samples,edges)
subplot(1,2,2);
nonsamples=odata;
figure;
histogram(nonsamples)

nz(2)=0;
num=0;
for i=1:d
    nz(2)=nz(2)+nnz(core.core{i}(:));
    num=num+numel(core.core{i});
end
% NZ(ko)=nz;
SP=100-nz./num*100;
end


end
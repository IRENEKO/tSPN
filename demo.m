% nltcs
clear all
load nltcs_tspn
load nltcsData

sample_train=[nltcs_ts;nltcs_valid];
sample_test=nltcs_test;
% case 0 refers to compute only the original distribution; case 1 refers to
% compute original distribution and tSPN distribution; can ignore other
% cases
opts.case=1;
opts.norm=1;
[core,nz]=tspn_iden(tensor,weight,sample_train,sample_test,opts);

% KL divergence
new=contract(core);
new=new(:);
clear core
[N,d,~]=size(tensor);
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
ori=contract(core);
ori=ori(:);
Idx=ori>0 & new>0;
KLdiv=sum(ori(Idx).*log(ori(Idx)./new(Idx)))





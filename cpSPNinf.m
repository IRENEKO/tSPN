function inference = cpSPNinf(tensor,weight,samples)
% 2018, Ching-Yun Ko

[N,d]=size(samples);
Data=cell(1,d);
for i=1:d
    Data{i}=zeros(N,2);
    for j=1:N
        Data{i}(j,2-samples(j,i))=1;
    end
end

[N,d,~]=size(tensor);
CP=fliplr(reshape(tensor(:,1,:),[N,2]))'.*weight;
inference=Data{1}*CP;
for i=2:d
    CP=fliplr(reshape(tensor(:,i,:),[N,2]))';
    inference=inference.*(Data{i}*CP);
end
inference=sum(inference,2);

end
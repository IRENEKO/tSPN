function [n,temp]=nonrepeated(A)
N=size(A,1);
temp=A(1,:);
for i=2:N
    iden=ismember(A(i,:),temp,'rows');
    if ~iden
        temp=[temp;A(i,:)];
    end
end
n=size(temp,1);
end
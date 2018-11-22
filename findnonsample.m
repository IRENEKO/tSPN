function B=findnonsample(A,n)
% 2018, Ching-Yun Ko
d=size(A,2);
B=[];
while size(B,1)<n
    if size(B,1)==0
        b=double(randn(1,d)<0);
        iden1=ismember(b,A,'rows');
        if ~iden1
            B=b;
        end
    else
        b=double(randn(1,d)<0);
        iden1=ismember(b,A,'rows');
        iden2=ismember(b,B,'rows');
        if ~(iden1||iden2)
            B=[B;b];
        end
    end
end

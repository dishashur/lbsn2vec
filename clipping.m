ele = 20;
k = 1;
smol_checkins = [];
temp = sortrows(selected_checkins,1);
for i=1:length(temp)
    if temp(i,1)<=ele
        smol_checkins(k,:) = temp(i,:);
        k=k+1;
    end
end


smol_user_IDs = selected_users_IDs(1:ele);

k=1;
fren_new = [];
for i=1:length(smol_user_IDs)
    temp = find(friendship_new(:,1)==i);
    for j=1:length(temp)
        fren_new(k,:)=friendship_new(temp(j),:);
        k=k+1;
    end
    temp = find(friendship_new(:,2)==i);
    for j=1:length(temp)
        fren_new(k,:)=friendship_new(temp(j),:);
        k=k+1;
    end
end
fren_new = unique(fren_new,'rows');

for i=1:length(fren_new)
    if fren_new(i,2) > ele
        fren_new(i,:) = 0;
    end
end
fren_new = fren_new(fren_new(:,1)>0,:);


k=1;
fren_old = [];
for i=1:length(smol_user_IDs)
    temp = find(friendship_old(:,1)==i);
    for j=1:length(temp)
        fren_old(k,:)=friendship_old(temp(j),:);
        k=k+1;
    end
    temp = find(friendship_old(:,2)==i);
    for j=1:length(temp)
        fren_old(k,:)=friendship_old(temp(j),:);
        k=k+1;
    end
end
fren_old = unique(fren_old,'rows');

for i=1:length(fren_old)
    if fren_old(i,2) > ele
        fren_old(i,:) = 0;
    end
end
fren_old = fren_old(fren_old(:,1)>0,:);

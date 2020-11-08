
toy_users_IDs = randi([0,10000],5,1);

toy_friendship_old = [1,5;2,3;3,4];

toy_venue = randi([100,10000],1,7);

toy_checkins = [1,randi([1,168]),toy_venue(randi([1,numel(toy_venue)])),randi([1,7]);...
    2,randi([1,168]),toy_venue(randi([1,numel(toy_venue)])),randi([1,7]);...
    3,randi([1,168]),toy_venue(randi([1,numel(toy_venue)])),randi([1,7]);...
    4,randi([1,168]),toy_venue(randi([1,numel(toy_venue)])),randi([1,7]);...
    5,randi([1,168]),toy_venue(randi([1,numel(toy_venue)])),randi([1,7])];

%clipped data_set

qty = 500;

new_users_IDs = selected_users_IDs(1:qty);

k = 1;
for i=1:size(friendship_old,1)
     if (friendship_old(i,1)<=qty)||(friendship_old(i,2)<=qty)
            new_friendship_old(k,:)= friendship_old(i,:);
            k=k+1;
     end
end

%make sure all qty user ids are accounted for atleast
%once in the friendship matrix
found = ismember([1:qty],new_friendship_old);
found = found';
temp = find(found == 0); %ensure that all users are accounted for atleast once
for i=1:numel(new_friendship_old)
    if new_friendship_old(i)>qty
        new_friendship_old(i)=randi([1,qty]);
    end
end



k = 1;
for i=1:size(selected_checkins,1)
    if selected_checkins(i,1)<=qty
        new_checkins(k,:) = selected_checkins(i,:);
        k=k+1;
    end
end

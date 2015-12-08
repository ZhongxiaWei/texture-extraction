% % Advanced Texture classification/pca and svm after feature extraction
% % Name:Zhongxia Wei
% % ID:9443118655
% % email:zhongxiw@usc.edu
% % Compiled on mac with Matlab
clear all;
all_pic = load ('/Users/weizhongxia/Desktop/courses/569/hw/hw2/HW2_images/P1/part b/character.csv');
all_pic = reshape(all_pic,[25,192]);
all_pic = all_pic';
label(1:48)=0;  %grass
label(49:96)=1;  %leather
label(97:144)=2  %sand
label(145:192)=3  %straw
all_pic_lab = [all_pic label'];
grass=all_pic_lab(1:48,:);
leather=all_pic_lab(49:96,:);
sand=all_pic_lab(97:144,:);
straw=all_pic_lab(145:192,:);
[data_w,score,latent] = pca(all_pic);
all_pic_3=all_pic*data_w(:,1:3);
figure(1);
plot3(all_pic_3(1:48,1),all_pic_3(1:48,2),all_pic_3(1:48,3),'g*');
hold on
plot3(all_pic_3(49:96,1),all_pic_3(97:144,2),all_pic_3(145:192,3),'r*');
hold on
plot3(all_pic_3(97:144,1),all_pic_3(97:144,2),all_pic_3(97:144,3),'b*');
hold on
plot3(all_pic_3(145:192,1),all_pic_3(145:192,2),all_pic_3(145:192,3),'c*');


%% grass vs nongrass
p=randperm(48);
for i=1:36
    grass_train(i,:)=grass(p(i),:);
end
for i=37:48
    grass_test(i-36,:)=grass(p(i),:);
end
%nongrass
p=randperm(48);
for i=1:12
    grass_train(i+36,:)=leather(p(i),:);
end
p=randperm(48);
for i=1:12
    grass_train(i+48,:)=sand(p(i),:);
end
p=randperm(48);
for i=1:12
    grass_train(i+60,:)=straw(p(i),:);
end

%% leather vs nonleather
p=randperm(48);
for i=1:36
    leather_train(i,:)=leather(p(i),:);
end
for i=37:48
    leather_test(i-36,:)=leather(p(i),:);
end
%nonleather
p=randperm(48);
for i=1:12
    leather_train(i+36,:)=grass(p(i),:);
end
p=randperm(48);
for i=1:12
    leather_train(i+48,:)=sand(p(i),:);
end
p=randperm(48);
for i=1:12
    leather_train(i+60,:)=straw(p(i),:);
end

%% sand vs nonsand
p=randperm(48);
for i=1:36
    sand_train(i,:)=sand(p(i),:);
end
for i=37:48
    sand_test(i-36,:)=sand(p(i),:);
end
%nongrass
p=randperm(48);
for i=1:12
    sand_train(i+36,:)=leather(p(i),:);
end
p=randperm(48);
for i=1:12
    sand_train(i+48,:)=grass(p(i),:);
end
p=randperm(48);
for i=1:12
    sand_train(i+60,:)=straw(p(i),:);
end

%% straw vs nonstraw
p=randperm(48);
for i=1:36
    straw_train(i,:)=straw(p(i),:);
end
for i=37:48
    straw_test(i-36,:)=straw(p(i),:);
end
%nongrass
p=randperm(48);
for i=1:12
    straw_train(i+36,:)=leather(p(i),:);
end
p=randperm(48);
for i=1:12
    straw_train(i+48,:)=sand(p(i),:);
end
p=randperm(48);
for i=1:12
    straw_train(i+60,:)=grass(p(i),:);
end

test=[grass_test;leather_test;sand_test;straw_test];

%% train
[grass_w,score,latent] = pca(grass_train(:,1:24));
grass_pca=grass_train(:,1:24)*grass_w(:,1:3);
grass_new=grass_pca(1:36,:);
nongrass_new=grass_pca(37:72,:);
grass_mean=mean(grass_new);
nongrass_mean=mean(nongrass_new);
grass_cov=cov(grass_new);
nongrass_cov=cov(nongrass_new);
for i=1:72
    disG=(grass_pca(i,:)-grass_mean)*inv(grass_cov)*(grass_pca(i,:)-grass_mean)';
    disN=(grass_pca(i,:)-nongrass_mean)*inv(nongrass_cov)*(grass_pca(i,:)-nongrass_mean)';
    if disG<disN
        gVS(i)= 1;
    end
    if disG>disN
        gVS(i)= 0;
    end
end
grassClaRe(1:36)=1;
grassClaRe(37:72)=0;
errRate_g_n = mean(gVS ~= grassClaRe);

[leather_w,score,latent] = pca(leather_train(:,1:24));
leather_pca=leather_train(:,1:24)*leather_w(:,1:3);
leather_new=leather_pca(1:36,:);
nonleather_new=leather_pca(37:72,:);
leather_mean=mean(leather_new);
nonleather_mean=mean(nonleather_new);
leather_cov=cov(leather_new);
nonleather_cov=cov(nonleather_new);
for i=1:72
    disL=(leather_pca(i,:)-leather_mean)*inv(leather_cov)*(leather_pca(i,:)-leather_mean)';
    disNL=(leather_pca(i,:)-nonleather_mean)*inv(nonleather_cov)*(leather_pca(i,:)-nonleather_mean)';
    if disL<disNL
        lVS(i)= 1;
    end
    if disL>disNL
        lVS(i)= 0;
    end
end
leatherClaRe(1:36)=1;
leatherClaRe(37:72)=0;
errRate_l_n = mean(lVS ~= leatherClaRe);

[sand_w,score,latent] = pca(sand_train(:,1:24));
sand_pca=sand_train(:,1:24)*sand_w(:,1:3);
sand_new=sand_pca(1:36,:);
nonsand_new=sand_pca(37:72,:);
sand_mean=mean(sand_new);
nonsand_mean=mean(nonsand_new);
sand_cov=cov(sand_new);
nonsand_cov=cov(nonsand_new);
for i=1:72
    disS=(sand_pca(i,:)-sand_mean)*inv(sand_cov)*(sand_pca(i,:)-sand_mean)';
    disNS=(sand_pca(i,:)-nonsand_mean)*inv(nonsand_cov)*(sand_pca(i,:)-nonsand_mean)';
    if disS<disNS
        sVS(i)= 1;
    end
    if disS>disNS
        sVS(i)= 0;
    end
end
sandClaRe(1:36)=1;
sandClaRe(37:72)=0;
errRate_s_n = mean(sVS ~= sandClaRe);

[straw_w,score,latent] = pca(straw_train(:,1:24));
straw_pca=straw_train(:,1:24)*straw_w(:,1:3);
straw_new=straw_pca(1:36,:);
nonstraw_new=straw_pca(37:72,:);
straw_mean=mean(straw_new);
nonstraw_mean=mean(nonstraw_new);
straw_cov=cov(straw_new);
nonstraw_cov=cov(nonstraw_new);
for i=1:72
    disS=(straw_pca(i,:)-straw_mean)*inv(straw_cov)*(straw_pca(i,:)-straw_mean)';
    disNS=(straw_pca(i,:)-nonstraw_mean)*inv(nonstraw_cov)*(straw_pca(i,:)-nonstraw_mean)';
    if disS<disNS
        strVS(i)= 1;
    end
    if disS>disNS
        strVS(i)= 0;
    end
end
strawClaRe(1:36)=1;
strawClaRe(37:72)=0;
errRate_str_n = mean(strVS ~= strawClaRe);

%% test data
[test_w,score,latent] = pca(test(:,1:24));
test_pca=test(:,1:24)*test_w(:,1:3);

for i=1:48
    dis_G=(test_pca(i,:)-grass_mean)*inv(grass_cov)*(test_pca(i,:)-grass_mean)';
    dis_L=(test_pca(i,:)-leather_mean)*inv(leather_cov)*(test_pca(i,:)-leather_mean)';
    dis_S=(test_pca(i,:)-sand_mean)*inv(sand_cov)*(test_pca(i,:)-sand_mean)';
    dis_Str=(test_pca(i,:)-straw_mean)*inv(straw_cov)*(test_pca(i,:)-straw_mean)';
    minDis=min(min(dis_G,dis_L),min(dis_S,dis_Str));
    if dis_G==minDis
        labeled(i)= 0;
    end
    if dis_L==minDis
        labeled(i)= 1;
    end
    if dis_S==minDis
        labeled(i)= 2;
    end
    if dis_Str==minDis
        labeled(i)= 3;
    end
end
errRate_test = mean(labeled ~= test(:,26)');


%% svm
label=[ones(1,36), -1*ones(1,36)]';
params = ['-t 0 -s 0'];
modelG=svmtrain(label,grass_train(:,1:24),params);
[predicted_label,accuracy,decision_values]=svmpredict(label,grass_train(:,1:24),modelG);
grass_svm=accuracy;

modelL=svmtrain(label,grass_train(:,1:24),params);
[predicted_label,accuracy,decision_values]=svmpredict(label,leather_train(:,1:24),modelL);
leather_svm=accuracy;

modelS=svmtrain(label,grass_train(:,1:24),params);
[predicted_label,accuracy,decision_values]=svmpredict(label,sand_train(:,1:24),modelS);
sand_svm=accuracy;
                    
modelStr=svmtrain(label,grass_train(:,1:24),params);
[predicted_label,accuracy,decision_values]=svmpredict(label,straw_train(:,1:24),modelStr);
straw_svm=accuracy;

label(1:12)=1;
label(13:48)=-1;
[predicted_label,accuracy,decision_values]=svmpredict(label,test(:,1:24),modelG);
grass_svm_test=accuracy;

label(1:48)=-1;
label(13:24)=1;
[predicted_label,accuracy,decision_values]=svmpredict(label,test(:,1:24),modelL);
leather_svm_test=accuracy;

label(1:48)=-1;
label(25:36)=1;
[predicted_label,accuracy,decision_values]=svmpredict(label,test(:,1:24),modelS);
sand_svm_test=accuracy;

label(1:48)=-1;
label(37:48)=1;
[predicted_label,accuracy,decision_values]=svmpredict(label,test(:,1:24),modelStr);
straw_svm_test=accuracy;

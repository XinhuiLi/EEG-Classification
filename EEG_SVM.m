clc;clear;close all;

%% initialize parameters
parpool open 2

sub_id = 1;
load(['Subject_',num2str(sub_id),'.mat']);
load(['./ICA_all/EEG_',num2str(sub_id),'_ICArej.mat']);
n_chan = size(X_EEG_TRAIN,1);
n_sp = n_TRAIN + n_TEST;
n_ic = size(EEG.icaweights,1);
% t_start = 501;
t_len = 50;
t_range = 300 + (1:10:350);
% t_end = t_start + t_len - 1;
datareshape = reshape(EEG.data,n_chan,n_sp*1200);
datareshape = EEG.icaweights * EEG.icasphere * datareshape;
dataICA = reshape(datareshape,n_ic,1200,n_sp);

acc = zeros(size(t_range));
box = zeros(size(t_range));
sclae = zeros(size(t_range));
model = cell(size(t_range));
test_zscore = cell(size(t_range));

%% train SVM classifier 

tic;
parfor t_ind = 1:length(t_range)
    % data segmentation
    t_start = t_range(t_ind);
    t_end = t_start + t_len - 1;
    total_data = zeros(n_ic, n_sp);
    dataICA_mean = mean(dataICA(:,t_start:t_end,:),2);
    total_data(:,:) = dataICA_mean(:,1,:);
    total_svm_data = zscore(total_data');
    train_svm_data = total_svm_data(1:n_TRAIN,:);
    test_zscore{t_ind} = total_svm_data(n_TRAIN+1:end,:);
    % SVM classifier
    cvp = cvpartition(n_TRAIN, 'LeaveOut');
    rng default;
    Mdl = fitcsvm(train_svm_data,Y_EEG_TRAIN,'OptimizeHyperparameters','auto',...
        'KernelFunction','rbf','HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
        'expected-improvement-plus', 'CVPartition',cvp));
    %Mdl = fitcsvm(train_svm_data, Y_EEG_TRAIN,'Standardize',true, 'KernelFunction','linear','BoxConstraint',1,'KernelScale',3.16,'CVPartition',cvp);
    CVSVM = crossval(Mdl,'Leaveout','on');
    accuracy = 1 - kfoldLoss(CVSVM);
    %accuracy = 1 - kfoldLoss(Mdl);
    acc(t_ind) = accuracy;
    box(t_ind) = Mdl.BoxConstraints(1);
    scale(t_ind) = Mdl.KernelParameters.Scale;
    model{t_ind} = Mdl;
    
%     if(acc>max_acc)
%         max_acc = acc;
%         box = Mdl.BoxConstraints(1);
%         scale = Mdl.KernelParameters.Scale;
%         max_time = t_start;
%     end
end

t_parfor = toc;
acc_max = max(acc);
max_ind = find(acc==acc_max);
box_max = box(max_ind);
scale_max = scale(max_ind);
t_max = t_range(max_ind);
model_max = model{max_ind};
test_zscored_data = test_zscore{max_ind};
save(['./finalresults/Subject',num2str(sub_id),'model.mat'],'model_max','t_max','acc_max');
fprintf('Sub:%d. Max Acc:%.2f. Max time:%d.',sub_id, acc_max, t_max);

%% predict accuracy

[pred_y,pred_score] = predict(model_max, test_zscored_data);
save(['./finalresults/Subject',num2str(sub_id),'prediction.mat'],'pred_y');
t_start = t_max;
t_end = t_start + t_len - 1;
dataICA_mean = mean(dataICA(:,t_start:t_end,:),2);
total_data(:,:) = dataICA_mean(:,1,:);
total_svm_data = zscore(total_data');
train_svm_data = total_svm_data(1:n_TRAIN,:);
[pred_train_label,pred_train_score] = predict(model_max, train_svm_data);

%% plot ROC

[X,Y,AUC] = perfcurve(Y_EEG_TRAIN,pred_train_score(:,2),1);
figure;
plot(X,Y,'LineWidth',2);
title('ROC'); xlabel('FP rate'); ylabel('TP rate');
legend(['AUC=',num2str(AUC)]);

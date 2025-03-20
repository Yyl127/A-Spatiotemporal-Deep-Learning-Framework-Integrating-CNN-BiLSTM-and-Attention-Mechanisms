function [output1,output2,output3,output4,output5]= CNN_BILSTM(inputArg1)

Data=inputArg1;
Data(find(isinf(Data)==1)) = 0;
Data(find(isnan(Data)==1)) = 0;

row_to_delete = []; % 用于存储需要删除的行索引
for i = 1:size(Data, 1)
    % 判断该行是否有10个以上的零
    if sum(Data(i, 1:10) == 0) >= 8
        row_to_delete = [row_to_delete; i];
    end
end

% 删除符合条件的行
Data(row_to_delete, :) = [];
data1 = Data;



TotalSamples = size(data1, 1);
shuffledIdx = randperm(TotalSamples);
Data = data1(shuffledIdx, :);

%% 划分训练集、验证集和测试集
Train_Size = round(0.6 * TotalSamples);
Val_Size = round(0.2 * TotalSamples);
Test_Size = TotalSamples - Train_Size - Val_Size;

% 训练集
Train_InPut = Data(1:Train_Size, 1:11);
Train_OutPut = Data(1:Train_Size, end);

% 验证集
Val_InPut = Data(Train_Size+1:Train_Size+Val_Size, 1:11);
Val_OutPut = Data(Train_Size+1:Train_Size+Val_Size, end);

% 测试集
Test_InPut = Data(Train_Size+Val_Size+1:end, 1:11);
Test_OutPut = Data(Train_Size+Val_Size+1:end, end);

%% 数据归一化
% 输入归一化
[~, Ps.Input] = mapminmax(Train_InPut', -1, 1);
Train_InPut = mapminmax('apply', Train_InPut', Ps.Input);
Val_InPut = mapminmax('apply', Val_InPut', Ps.Input);
Test_InPut = mapminmax('apply', Test_InPut', Ps.Input);

% 输出归一化
[~, Ps.Output] = mapminmax(Train_OutPut', -1, 1);
Train_OutPut = mapminmax('apply', Train_OutPut', Ps.Output);
Val_OutPut = mapminmax('apply', Val_OutPut', Ps.Output);
Test_OutPut = mapminmax('apply', Test_OutPut', Ps.Output);

%% 调整数据维度
% 转换为CNN输入格式 [11, 1, 1, batchSize]
  Train_InPut_CNN = reshape(Train_InPut, [11,1,1,1,size(Train_InPut,2)]); % [C=11, H=1, W=1, T=1, B]
 Train_InPut_CNN = permute(Train_InPut_CNN, [1,2,3,5,4]); % [11,1,1,B,1]
%Train_InPut_CNN = reshape(Train_InPut, 11, 1, 1, []);
Val_InPut_CNN = reshape(Val_InPut, 11, 1, 1, []);
Test_InPut_CNN = reshape(Test_InPut, 11, 1, 1, []);

% 转换为LSTM输入格式 [11, 1, batchSize]
Train_InPut_LSTM = reshape(Train_InPut, 11, 1, []);
Val_InPut_LSTM = reshape(Val_InPut, 11, 1, []);
Test_InPut_LSTM = reshape(Test_InPut, 11, 1, []);

%% 构建增强网络结构
spatialBranch = [
    imageInputLayer([11 1 1], 'Name', 'spatial_input')
    
    % 第一个卷积块
    convolution2dLayer([3,1], 512, 'Padding','same', 'Name', 'conv1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')
    dropoutLayer(0.3, 'Name', 'dropout1')
    
    % 第二个卷积块
    convolution2dLayer([3,1], 512, 'Padding','same', 'Name', 'conv2')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')
    dropoutLayer(0.3, 'Name', 'dropout2')
    
    % === 空间注意力机制 ===
    ChannelPoolingLayer('Name', 'channel_pool')  % 显式命名池化层
    convolution2dLayer([1,1], 1, 'Padding','same', 'Name', 'spatial_att_conv')
    sigmoidLayer('Name', 'spatial_att_sigmoid')
    multiplicationLayer(2, 'Name', 'spatial_attention')
    
    % 后续层
    maxPooling2dLayer([2,1], 'Stride', [2 1], 'Padding','same', 'Name', 'maxpool1')
    fullyConnectedLayer(512, 'WeightL2Factor', 0.02, 'Name', 'fc1')
    reluLayer('Name', 'relu3')
    dropoutLayer(0.4, 'Name', 'dropout3')
    
    fullyConnectedLayer(256, 'WeightL2Factor', 0.02, 'Name', 'fc2')
    reluLayer('Name', 'relu4')
    dropoutLayer(0.4, 'Name', 'dropout4')
    
    fullyConnectedLayer(50, 'Name', 'fc_out')
    flattenLayer('Name','spatial_flatten')
];

% 时间分支（BiLSTM）
temporalBranch = [
    sequenceInputLayer(11, 'Name', 'temporal_input')
    
    bilstmLayer(512, 'OutputMode','sequence')
    dropoutLayer(0.3)
    
    bilstmLayer(256, 'OutputMode','sequence')
    dropoutLayer(0.3)
    
    bilstmLayer(64, 'OutputMode','last')
    
    fullyConnectedLayer(128, 'WeightL2Factor', 0.02)
    reluLayer
    dropoutLayer(0.4)
    
    fullyConnectedLayer(50)
    flattenLayer('Name','temporal_flatten')
];

% 融合分支
fusionBranch = [
    concatenationLayer(1, 2, 'Name','fusion_concat')
    
    SpatioTemporalAttention1D('stt_att')

    



    fullyConnectedLayer(256, 'WeightL2Factor', 0.02)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.4)
    
    fullyConnectedLayer(128, 'WeightL2Factor', 0.02)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.4)
    
    fullyConnectedLayer(64)
    reluLayer
    
    fullyConnectedLayer(1)  % 单输出
];

% 构建网络
lgraph = layerGraph(spatialBranch);
lgraph = addLayers(lgraph, temporalBranch);
lgraph = addLayers(lgraph, fusionBranch);
lgraph = connectLayers(lgraph, 'relu2', 'spatial_attention/in2');                % 原始特征图

%lgraph = connectLayers(lgraph, 'fusion_concat', 'add/in2');

lgraph = connectLayers(lgraph, 'spatial_flatten', 'fusion_concat/in1');
lgraph = connectLayers(lgraph, 'temporal_flatten', 'fusion_concat/in2');
net = dlnetwork(lgraph);

%% 训练参数优化
numEpochs = 1000;
miniBatchSize = 64;
initialLearnRate = 0.001;
warmupEpochs = 50;
patience = 50;

%% 训练循环（学习率预热）
% figure;
% lineLossTrain = animatedline('Color', [0.85 0.325 0.098], 'LineWidth', 1.5);
% lineLossVal = animatedline('Color', [0 0.447 0.741], 'LineWidth', 1.5);
% xlabel("Epoch");
% ylabel("Loss");
% legend('Training Loss','Validation Loss');
% grid on;

numIterationsPerEpoch = floor(Train_Size / miniBatchSize);
bestLoss = inf;
epochsWithoutImprovement = 0;
averageGrad = [];
averageSqGrad = [];

for epoch = 1:numEpochs
    % 学习率预热 + 余弦衰减
    if epoch <= warmupEpochs
        learnRate = initialLearnRate * (epoch / warmupEpochs);
    else
        progress = (epoch - warmupEpochs) / (numEpochs - warmupEpochs);
        learnRate = 0.5 * initialLearnRate * (1 + cos(pi * progress));
    end
    
    % 训练阶段（添加输入噪声）
    for i = 1:numIterationsPerEpoch
        idx = (i-1)*miniBatchSize+1 : i*miniBatchSize;
        
        % 添加高斯噪声增强
        noiseScale = 0.02;
        X1 = dlarray(Train_InPut_CNN(:, :, :, idx) + noiseScale*randn(size(Train_InPut_CNN(:, :, :, idx))), 'SSCB');
        X2 = dlarray(Train_InPut_LSTM(:, :, idx) + noiseScale*randn(size(Train_InPut_LSTM(:, :, idx))), 'CTB');
        Y = dlarray(Train_OutPut(:, idx), 'CB');
        
        [gradients, totalLoss] = dlfeval(@modelGradients, net, X1, X2, Y);
        [net.Learnables, averageGrad, averageSqGrad] = ...
            adamupdate(net.Learnables, gradients, averageGrad, averageSqGrad, epoch, learnRate);
    end
    
    % 验证阶段
    XVal1 = dlarray(Val_InPut_CNN, 'SSCB');
    XVal2 = dlarray(Val_InPut_LSTM, 'CTB');
    YVal = dlarray(Val_OutPut, 'CB');
    
    YValPred = forward(net, XVal1, XVal2);
    valLoss = mean(huberLoss(YValPred, YVal, 0.5));
    
    % 记录损失
    % addpoints(lineLossTrain, epoch, double(extractdata(totalLoss)));
    % addpoints(lineLossVal, epoch, double(extractdata(valLoss)));
     
    % 早停判断
    if valLoss < bestLoss
        bestLoss = valLoss;
        epochsWithoutImprovement = 0;
        bestNet = net;
    else
        epochsWithoutImprovement = epochsWithoutImprovement + 1;
    end
    
    if epochsWithoutImprovement >= patience
        fprintf('Early stopping at epoch %d\n', epoch);
        break;
    end
    
    fprintf('Epoch: %d, LR: %.5f, TrainLoss: %.4f, ValLoss: %.4f\n',...
        epoch, learnRate, double(extractdata(totalLoss)), double(extractdata(valLoss)));
    drawnow;
end

%% 测试与评估
net = bestNet;
XTest1 = dlarray(Test_InPut_CNN, 'SSCB');
XTest2 = dlarray(Test_InPut_LSTM, 'CTB');
YPred = forward(net, XTest1, XTest2);

True_Test = mapminmax('reverse', Test_OutPut, Ps.Output)';
Predicted_Test = mapminmax('reverse', extractdata(YPred), Ps.Output)';

% 评估指标
evalMetrics = @(true, pred) struct(...
    'R2', 1 - sum((true - pred).^2)/sum((true - mean(true)).^2),...
    'RMSE', sqrt(mean((true - pred).^2)),...
    'MAE', mean(abs(true - pred)));

metrics = evalMetrics(True_Test, Predicted_Test);


output1=metrics.R2;
output2=metrics.RMSE;
output3=metrics.MAE;
output4=Ps;
output5=net;

end

clear;
load TWS_data.mat
data=merged_matrix;
load predict_data.mat
data_2=pre_data;
clear pre_data;

a=[7,11,12,12,12,12,12,12,12,10,10,9,9,9,9,5,5,12,12,12,12,12,0];
z=a.*156;
C = zeros(22,9);
pre_data=zeros(120,130,228);
x=1;
y=1092;
i=1;
flag=0;
while i<23
[output1,output2,output3,output4,output5]= CNN_BILSTM(data(x:y,:));
C(i,1)=output1;
C(i,2)=output2;
C(i,3)=output3;

C(i,9)=y-x+1;
x=x+z(i);
y=y+z(i+1);
Ps=output4;
net=output5;

            if i==1
                for j=1:9
                    new_data=data_2(:,:,(i-1)*12+j);
                    % 输入归一化（使用训练时的参数）
                    Norm_Input = mapminmax('apply', new_data', Ps.Input);

                    %% 数据维度转换
                    % CNN分支输入格式 [11,1,1,batchSize]
                    input_cnn = reshape(Norm_Input, 11, 1, 1, []);

                    % LSTM分支输入格式 [11,1,batchSize]
                    input_lstm = reshape(Norm_Input, 11, 1, []);

                    %% 创建dlarray
                    X1 = dlarray(input_cnn, 'SSCB');
                    X2 = dlarray(input_lstm, 'CTB');

                    %% 进行预测
                    YPred = forward(net, X1, X2);

                    %% 反归一化输出
                    predicted_values = mapminmax('reverse', extractdata(YPred), Ps.Output);
 
                    Predicted_New = double(predicted_values);  % 转换为双精度
                    [output]= to3(Predicted_New);
                    pre_data(:,:,j)=output;
                    fprintf(' 已预测个数： %d .\n', (i-1)*12+j);
                    
                end
            else
                for j=1:12
                    new_data=data_2(:,:,(i-1)*12+j-3);
        
                     Norm_Input = mapminmax('apply', new_data', Ps.Input);

                    %% 数据维度转换
                    % CNN分支输入格式 [11,1,1,batchSize]
                    input_cnn = reshape(Norm_Input, 11, 1, 1, []);

                    % LSTM分支输入格式 [11,1,batchSize]
                    input_lstm = reshape(Norm_Input, 11, 1, []);

                    %% 创建dlarray
                    X1 = dlarray(input_cnn, 'SSCB');
                    X2 = dlarray(input_lstm, 'CTB');

                    %% 进行预测
                    YPred = forward(net, X1, X2);

                    %% 反归一化输出
                    predicted_values = mapminmax('reverse', extractdata(YPred), Ps.Output);
                    Predicted_New = double(predicted_values);  % 转换为双精度
                    [output]= to3(Predicted_New);
                    pre_data(:,:,(i-1)*12+j-3)=output;
                    fprintf(' 已预测个数： %d .\n', (i-1)*12+j-3);
                    
                end
            end
     



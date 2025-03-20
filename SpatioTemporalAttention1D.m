classdef SpatioTemporalAttention1D < nnet.layer.Layer
    properties (Learnable)
        W_spatial   % [C,1,T] 每个时间步独立的空间权重
        W_temporal  % [C,1,T] 每个通道独立的时间权重
        alpha       % 可学习缩放因子
    end
    
    methods
        function layer = SpatioTemporalAttention1D(name)
            layer.Name = name;
            layer.alpha = dlarray(0.1); % 初始值
        end
        
        function Z = predict(layer, X)
            % X: [C,1,B,T]
            
            % 空间注意力（每个时间步独立）
            spatial_att = sigmoid( sum(X .* layer.W_spatial, 2, 'native') ); % [C,1,B,T]
            
            % 时间注意力（每个通道独立）
            temporal_att = sigmoid( sum(X .* layer.W_temporal, [1,3], 'native') ); % [C,1,B,T]
            
            % 结合方式（元素乘积）
            combined_att = spatial_att .* temporal_att;
            
            % 残差连接（可学习缩放）
            Z = X + layer.alpha * (X .* combined_att);
        end
        
        function layer = initialize(layer, X)
            [C, ~, ~, T] = size(X);
            
            % 初始化参数
            layer.W_spatial = dlarray(randn(C,1,T) * sqrt(2/(C+T))); % He初始化
            layer.W_temporal = dlarray(randn(C,1,T) * sqrt(2/(C+T)));
            layer.alpha = dlarray(0.1); 
        end
    end
end

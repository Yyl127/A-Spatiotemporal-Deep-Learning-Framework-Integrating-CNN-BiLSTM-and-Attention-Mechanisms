classdef ChannelPoolingLayer < nnet.layer.Layer
    methods
        % 添加构造函数处理Name参数
        function this = ChannelPoolingLayer(varargin)
            % 解析输入参数（支持名称-值对）
            p = inputParser;
            addParameter(p, 'Name', '', @ischar);
            parse(p, varargin{:});
            
            % 设置层属性
            this.Name = p.Results.Name;
            this.Description = "Channel Pooling Layer";
        end
        
        function Z = predict(~, X)
            avg = mean(X, 3);    % 沿通道平均
            max_val = max(X, [], 3); % 沿通道最大值
            Z = cat(3, avg, max_val); % 拼接结果
        end
    end
end

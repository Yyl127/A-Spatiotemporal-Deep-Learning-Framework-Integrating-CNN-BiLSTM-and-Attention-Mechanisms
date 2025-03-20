function [gradients, totalLoss] = modelGradients(net, X1, X2, Y)
    YPred = forward(net, X1, X2);
    delta = 0.5; % Huber损失参数
    totalLoss = mean(huberLoss(YPred, Y, delta));
    gradients = dlgradient(totalLoss, net.Learnables);
end


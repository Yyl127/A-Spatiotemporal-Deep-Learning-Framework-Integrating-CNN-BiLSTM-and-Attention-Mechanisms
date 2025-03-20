function loss = huberLoss(predictions, targets, delta)
    residual = abs(predictions - targets);
    quadratic = min(residual, delta);
    linear = residual - quadratic;
    loss = 0.5 * quadratic.^2 + delta * linear;
end
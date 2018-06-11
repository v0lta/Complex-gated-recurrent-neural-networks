close all
clear all
x = linspace(-10,10, 50);
y = linspace(-10,10, 50);
a = 0.1;
b = 0.5;
for i = 1:length(x)
    for j = 1:length(x)
        % sigx = 1./(1 + exp(-x(i)));
        % sigy = 1./(1 + exp(-y(j)));
        % sig = (a.*sigx + b.*sigy)./2
        sig = 1./(1 + exp(-(a * x(i) + b * y(j))));
        % sig = 1./(1 + exp(-a * x(i))) +  1./(1 + exp(-b * y(j)));
        %% tan = tanh(x(i).*x(i) + y(j).*y(j));
        g(i,j) = (sig);
    end
end
surf(x,y,g)

if 0
    for i = 1:1000
        A = randn(50, 50);
        B = randn(50, 50);
        [Ua, Sa, Va] = svd(A);
        [Ub, Sb, Vb] = svd(B);
        Oa = Ua*Va';
        Ob = Ub*Vb';
        norms(i) = norm((Oa + Ob)/2);
    end
    mean(norms)
end
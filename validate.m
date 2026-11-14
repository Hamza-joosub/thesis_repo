% Cross-validation harness: run Matthews's functions verbatim on toy inputs,
% write outputs to CSV for comparison against the Python implementation.
% Clone github.com/nematthews/Meucci-HS-FP-framework-SA-context next to this repository first.
addpath('../Meucci-HS-FP-framework-SA-context/Toolboxes/hsfp/functions');

% Case 1: exponential time decay, T=100, tau=10 periods
T = 100; tau = 10;
sig = zeros(T,1);                  % es_probs only uses height() of this
p1 = es_probs(sig, tau);
csvwrite('mat_p_T100_tau10.csv', p1');

% Case 2: T=1060, tau=118 (his sample and half-life)
T2 = 1060; tau2 = 118;
sig2 = zeros(T2,1);
p2 = es_probs(sig2, tau2);
csvwrite('mat_p_T1060_tau118.csv', p2');

% ENS on both, standard and generalised gamma=2
e1  = ens(p1);
e2  = ens(p2);
e1g = ens(p1, 'gen', 2);
e2g = ens(p2, 'gen', 2);
csvwrite('mat_ens.csv', [e1; e2; e1g; e2g]);

% Case 3: ENS on a hand-made ragged probability vector (non-exponential)
p3 = [0.4 0.3 0.15 0.1 0.05];
csvwrite('mat_ens_ragged.csv', [ens(p3); ens(p3,'gen',2)]);
printf('done\n');

% Exercise 3 - prolog programming
% Kacper Kamieniarz (293065)
% Jakub Szumski (295432)


% returns true if the sum of sizes of papers A0, ... , A5 is greater than size of A0
paper_size(A0, A1, A2, A3, A4, A5) :- 
	A0 + 1/2 * A1 + 1/4 * A2 + 1/8 * A3 + 1/16 * A4 + 1/32 * A5 >= 1.


% when called with (A,B,C) it returns all the triplets such that 
% A < B < C and A^2 + B^2 = C^2 and A,B,C are in range (1, 20)
pythagorean_b(A,B,C) :-
	between(1,20,A),
	between(A,20,B),
	between(B,20,C),
	C*C =:= A*A + B*B.


% when called with (A,B,C) it returns all the triplets
% such that A < B < C and A + B + C = 1000 and A^2 + B^2 = C^2
% this predicate checks the above conditions on all combinations
% of triplets from ranges s.t. A is between 1 and 332,
% B is between A and 499 and C is between B and 1000 - (A + B)
pythagorean_c(A,B,C) :-
	between(1,332,A),
	between(A,499,B),
	reduce(A,B,H),  
	between(B,H,C),
	A + B + C =:= 1000,
	C*C =:= A*A + B*B.

% puts the value of operation A - B in variable H
reduce(A,B,H) :- 
	H is 1000 - A - B.



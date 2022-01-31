%% staggevent2.m

% Modifications from default staggevent.m found in MatDyn1.2 archive [1], within 
% \MatDyn\Cases\Events.
% [1] https://www.esat.kuleuven.be/electa/teaching/matdyn/MatDyn1.2
% Three phase fault with fault resistance Rf is applied at virtual bus 3, which represents the 
% fault point in one of the transmission line and whose location is controlled in branch data of 
% casestagg2.m

function [event,buschange,linechange] = staggevent2

global Rf;
global t1 t2;

% The three phase fault starts at t1 and ends at t2.

% event = [time type]
event=[t1       1; 
       t1       1;
       t2       1;
       t2       1];

% In order to emulate the occurance of a fault, we have the following:
% when the fault emerges at t1, attribute 5 which pertains to shunt resistance changes value from 0 
% to fault resistance Rf;
% at the same time, attribute 6 corresponds to susceptance changes from 0 to a significantly large 
% negative value.
   
% buschange = [time bus(row)  attribute(col) new_value]
buschange   = [t1   3         6              -1e10;
               t1   3         5              Rf;
               t2   3         6              0;
               t2   3         5              0];
                      
% linechange = [time  line(row)  attribute(col) new_value]
linechange = [];

return;
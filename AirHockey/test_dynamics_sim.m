
TableLength = 2; % X

TableWidth = 1; % Y

GoalWidth = 0.2;

PuckMass = 1.0;
        
PuckRadius = 0.06;

StrikerMass = 0.1;

StrikerRadius = 0.08;

% Max Force the input can apply (Fx and Fy)
MaxForce = 10;

% Sample time
Ts = 0.02;

puckState = [TableLength/2; TableWidth/2; 0; 0];

% Visualization
fig = figure('Visible','on','HandleVisibility','off');
            
ha = gca(fig);
ha.XLimMode = 'manual';
ha.YLimMode = 'manual';

margin = 0.1;
ha.XLim = [-TableLength/2-margin, TableLength/2+margin];
ha.YLim = [-TableWidth/2-margin, TableWidth/2+margin];

hold(ha, 'on');

tablepoly = polyshape([-TableLength/2 TableLength/2 TableLength/2 -TableLength/2],[-TableWidth/2 -TableWidth/2 TableWidth/2 TableWidth/2]);
                  
tableplot = plot(ha,tablepoly,'FaceColor',[1 1 1]);